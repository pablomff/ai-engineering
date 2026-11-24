"""Ingesta: lee la documentación de httpx, la fragmenta y la sube a Pinecone.

Uso::

    uv run python -m pipelines.rag_hibrido.ingesta               # indexa si falta algo (o todo)
    uv run python -m pipelines.rag_hibrido.ingesta --reindexar   # vuelve a subir todo

Flujo::

    data/httpx-docs/**/*.md
      -> cargar_documentos   un Document por archivo, con fuente y categoría
      -> fragmentar          RecursiveCharacterTextSplitter (el de pipelines/rag):
                             hasta 500 tokens, 50 de solapamiento
      -> descartar_vacios    fuera los pedazos que son solo sintaxis (por ejemplo "```")
      -> armar_metadata      texto + fuente + sección + categoría por fragmento
      -> embed_documents     Pinecone calcula un vector de 1024 dimensiones por fragmento
      -> subir               upsert en lotes de 100 al namespace
      -> borrar_ids          fuera los ids viejos que ya no corresponden a ningún fragmento

Por qué el texto va en la metadata: Pinecone guarda vectores, y un vector
solo no se puede leer. Si el texto quedara en otro lado (una base
relacional, los archivos), cada búsqueda necesitaría una segunda consulta
para traerlo. Guardándolo junto al vector, una sola consulta devuelve todo.
También es lo que permite armar el corpus de BM25 leyendo solo Pinecone.
"""

import argparse
import logging
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone, PineconeError
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

from pipelines.rag.ingesta import fragmentar, id_de
from pipelines.rag.vectorstore import RAIZ_PROYECTO
from pipelines.rag_hibrido.config import ConfigRAG
from pipelines.rag_hibrido.embeddings import crear_contador_tokens, crear_embeddings
from pipelines.rag_hibrido.exceptions import IngestaIncompletaError, RAGHibridoError
from pipelines.rag_hibrido.indice import (
    IndiceVectorial,
    MetadataFragmento,
    VectorPinecone,
    asegurar_indice,
    configurar_logs,
    conectar,
    crear_cliente,
    dimension_embeddings,
)

logger = logging.getLogger(__name__)

CARPETA_POR_DEFECTO = RAIZ_PROYECTO / "data" / "httpx-docs"
"""Documentación de httpx (Markdown) y los RFC de HTTP que implementa (PDF, en
``rfc/``). Ver ``FUENTE.txt``."""

EXTENSIONES = frozenset({".md", ".pdf"})
"""Formatos que se indexan: Markdown y PDF."""

EXCLUIDOS = frozenset({"LICENSE.md"})
"""Archivos de la carpeta que no son documentación y no se indexan."""

TAMANO_LOTE = 100
"""Vectores por request de upsert.

Pinecone limita cada request a 1000 vectores y 2 MB. Con 1024 floats por
vector y hasta ~2 KB de texto en la metadata, 100 vectores ocupan unos
800 KB (en el JSON, cada float ocupa unos 8 caracteres): queda holgado y, si un lote falla, se reintenta solo ese lote.
"""

TAMANO_LOTE_BORRADO = 1000
"""Ids por request de borrado: el máximo que acepta Pinecone."""

# Un título de Markdown: de 1 a 6 '#' al principio de la línea y un espacio.
_TITULO = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Inicio o fin de un bloque de código (``` o ~~~).
_CERCO_CODIGO = re.compile(r"^\s*(```|~~~)")
# Título de sección de un RFC en el cuerpo: "5.2. Flow Control" o "Appendix A. ...".
# En el índice de contenidos, después del número viene un espacio no separable
# (\xa0); en el cuerpo, un espacio común. Así el índice no se toma por títulos.
_TITULO_RFC = re.compile(r"^((?:\d+(?:\.\d+)*|Appendix [A-Z](?:\.\d+)*)\.) (\S.*?)\s*$")
# Entrada del índice de contenidos: el número va seguido de un espacio no separable.
_INDICE_RFC = re.compile(r"^(?:\d+(?:\.\d+)*|Appendix [A-Z](?:\.\d+)*)\.\xa0|^Table of Contents\s*$")
# Encabezado y pie que se repiten en cada página de un RFC, por ejemplo
# "RFC 9112 HTTP/1.1 June 2022" y "Fielding, et al. Standards Track Page 20".
_ENCABEZADO_RFC = re.compile(r"^RFC \d{4,5} .+ (?:19|20)\d{2}$")
_PIE_RFC = re.compile(r"\sPage \d+$")


class ResultadoIngesta(BaseModel):
    """Resumen de una ingesta, para mostrar en consola y para los tests."""

    omitida: bool
    """``True`` si el namespace ya estaba completo y no se volvió a indexar."""
    namespace: str
    archivos: int
    fragmentos: int
    borrados: int = 0
    """Vectores viejos que se borraron porque ya no corresponden a ningún fragmento."""


def categoria_de(fuente: str) -> str:
    """Etiqueta de categoría a partir de la carpeta y el formato del archivo.

    - ``rfc``: los PDF de ``rfc/``, las especificaciones del protocolo.
    - ``avanzado``: la documentación de httpx en ``advanced/``.
    - ``general``: el resto de la guía de httpx.

    Con la categoría en la metadata se puede filtrar una búsqueda, por
    ejemplo ``{"categoria": {"$eq": "rfc"}}``.
    """
    if fuente.endswith(".pdf"):
        return "rfc"
    return "avanzado" if fuente.startswith("advanced/") else "general"


def limpiar_pagina(texto: str) -> tuple[list[str], list[str]]:
    """Limpia el texto de una página de RFC y detecta sus títulos de sección.

    Devuelve las líneas limpias y, en paralelo, el título que empieza en
    cada línea (``""`` si no es un título). Limpiar es:

    - Quitar el encabezado y el pie que el PDF repite en cada página: no
      dicen nada del contenido y, repetidos, ensucian la búsqueda.
    - Quitar el índice de contenidos: es una lista de títulos sin
      contenido, y aparecía en los resultados de casi cualquier pregunta
      sobre el RFC porque nombra todos los temas a la vez.
    - Normalizar con NFKC: el PDF usa ligaduras tipográficas (``ﬁ`` en vez
      de ``fi``) y espacios no separables. Sin normalizar, BM25 no
      encontraría "specified" en un texto que dice "speciﬁed".

    El título se detecta en la línea cruda, antes de normalizar, porque es
    el espacio no separable el que distingue el índice de contenidos de un
    título real.
    """
    lineas: list[str] = []
    titulos_linea: list[str] = []
    for cruda in texto.splitlines():
        limpia = unicodedata.normalize("NFKC", cruda).strip()
        if not limpia or _ENCABEZADO_RFC.match(limpia) or _PIE_RFC.search(limpia):
            continue
        if _INDICE_RFC.match(cruda.strip()):
            continue
        coincidencia = _TITULO_RFC.match(cruda.strip())
        titulo = ""
        # Un título real es corto y no termina en punto: así se descartan los
        # ítems de listas numeradas ("1. The receiver...").
        if coincidencia and len(coincidencia.group(2)) <= 80 and not limpia.endswith((".", ":")):
            titulo = unicodedata.normalize("NFKC", f"{coincidencia.group(1)} {coincidencia.group(2)}")
        lineas.append(limpia)
        titulos_linea.append(titulo)
    return lineas, titulos_linea


def leer_pdf(ruta: Path, fuente: str) -> Document:
    """Convierte un PDF en un único ``Document`` con el mapa de sus páginas y secciones.

    Se une todo el PDF en un solo texto (y no un ``Document`` por página)
    para que un párrafo que cruza de una página a la siguiente no quede
    partido en dos fragmentos. Para no perder la página, la metadata guarda
    dónde empieza cada una (``paginas``) y dónde empieza cada sección
    (``titulos``). Después, cada fragmento toma la página y la sección
    donde empieza.
    """
    partes: list[str] = []
    paginas: list[tuple[int, int]] = []
    titulos_pdf: list[tuple[int, str]] = []
    posicion = 0
    for numero, pagina in enumerate(PdfReader(ruta).pages, start=1):
        lineas, titulos_linea = limpiar_pagina(pagina.extract_text() or "")
        if not lineas:
            continue
        paginas.append((posicion, numero))
        for linea, titulo in zip(lineas, titulos_linea, strict=True):
            if titulo:
                titulos_pdf.append((posicion, titulo))
            partes.append(linea)
            posicion += len(linea) + 1  # +1 por el salto de línea que las une
    texto = "\n".join(partes)
    return Document(
        page_content=texto,
        metadata={"archivo": fuente, "paginas": paginas, "titulos": titulos_pdf},
    )


def cargar_documentos(carpeta: Path) -> list[Document]:
    """Lee los ``.md`` y ``.pdf`` de ``carpeta`` y sus subcarpetas, en orden alfabético.

    Cada archivo es un ``Document`` con ``archivo`` (la ruta relativa, por
    ejemplo ``advanced/timeouts.md`` o ``rfc/rfc9113.pdf``) en la metadata.
    Se llama ``archivo`` porque así lo espera ``fragmentar`` de
    ``pipelines.rag.ingesta``; después se publica en Pinecone como
    ``fuente``. Los PDF suman en la metadata el mapa de páginas y
    secciones (ver ``leer_pdf``).

    Raises:
        FileNotFoundError: si ``carpeta`` no existe.
    """
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {carpeta}")
    documentos: list[Document] = []
    for ruta in sorted(carpeta.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES or ruta.name in EXCLUIDOS or not ruta.is_file():
            continue
        fuente = ruta.relative_to(carpeta).as_posix()
        if ruta.suffix.lower() == ".pdf":
            documento = leer_pdf(ruta, fuente)
        else:
            documento = Document(
                page_content=ruta.read_text(encoding="utf-8").strip(), metadata={"archivo": fuente}
            )
        if not documento.page_content.strip():
            logger.warning("Se omite %s: no tiene texto", fuente)
            continue
        documentos.append(documento)
    return documentos


def titulos(texto: str) -> list[tuple[int, str]]:
    """Posición (en caracteres) y texto de cada título de Markdown del documento.

    Ignora lo que está dentro de bloques de código: en la documentación de
    httpx hay muchos ejemplos de Python, y un comentario como
    ``# Using a client instance:`` no es un título.
    """
    encontrados: list[tuple[int, str]] = []
    posicion = 0
    en_codigo = False
    for linea in texto.splitlines(keepends=True):
        if _CERCO_CODIGO.match(linea):
            en_codigo = not en_codigo
        elif not en_codigo and (coincidencia := _TITULO.match(linea.rstrip("\n"))):
            encontrados.append((posicion, coincidencia.group(2)))
        posicion += len(linea)
    return encontrados


def pagina_de(paginas: Sequence[tuple[int, int]], inicio: int) -> int:
    """Número de la página donde empieza un fragmento: la última que empieza antes de ``inicio``."""
    anteriores = [numero for posicion, numero in paginas if posicion <= inicio]
    return anteriores[-1] if anteriores else paginas[0][1]


def titulos_de(documento: Document) -> list[tuple[int, str]]:
    """Los títulos de sección de un documento: los del PDF, o los del Markdown."""
    if "titulos" in documento.metadata:
        return list(documento.metadata["titulos"])
    return titulos(documento.page_content)


def sin_mapas(documentos: Sequence[Document]) -> list[Document]:
    """Copia de los documentos solo con ``archivo`` en la metadata.

    El splitter copia la metadata del documento a cada fragmento. El mapa
    de páginas y secciones de un PDF no hace falta en cada fragmento: se
    consulta una vez, al armar la metadata.
    """
    return [
        Document(page_content=d.page_content, metadata={"archivo": d.metadata["archivo"]})
        for d in documentos
    ]


def recalcular_inicios(documentos: Sequence[Document], fragmentos: Sequence[Document]) -> None:
    """Corrige el ``start_index`` de cada fragmento: dónde empieza en el texto original.

    Por qué hace falta: el splitter de LangChain busca cada fragmento a partir
    de "inicio anterior + largo anterior - solapamiento", y resta el
    solapamiento como si fueran caracteres. Nuestro solapamiento son 50
    *tokens* (unos 200 caracteres), así que cuando dos fragmentos se solapan
    empieza a buscar demasiado adelante, no lo encuentra y deja -1. Con -1,
    la página y la sección del fragmento saldrían mal.

    Acá cada fragmento se busca desde el inicio del anterior más uno: los
    fragmentos de un documento están en orden y cada uno empieza después
    del anterior, así que ese punto nunca se pasa del lugar correcto.

    Raises:
        ValueError: si un fragmento no aparece en su documento (no debería
            pasar: el splitter solo recorta, no cambia el texto).
    """
    textos = {str(doc.metadata["archivo"]): doc.page_content for doc in documentos}
    anterior: dict[str, int] = {}
    for fragmento in fragmentos:
        fuente = str(fragmento.metadata["archivo"])
        inicio = textos[fuente].find(fragmento.page_content, anterior.get(fuente, -1) + 1)
        if inicio < 0:
            raise ValueError(f"No se encontró {id_de(fragmento)} en {fuente}")
        fragmento.metadata["start_index"] = inicio
        anterior[fuente] = inicio


def seccion_de(titulos_documento: Sequence[tuple[int, str]], inicio: int, fuente: str) -> str:
    """Título de la sección donde empieza un fragmento.

    Es el último título que aparece antes de ``inicio`` (o justo en
    ``inicio``, si el fragmento arranca con un título). Si no hay ninguno
    (documentos sin títulos, como ``event-hooks.md``), usa el nombre del
    archivo.

    Junto con la página (en los PDF), es lo que le dice a una persona
    dónde está el fragmento dentro del documento.
    """
    anteriores = [titulo for posicion, titulo in titulos_documento if posicion <= inicio]
    return anteriores[-1] if anteriores else Path(fuente).stem


def descartar_vacios(fragmentos: Sequence[Document]) -> list[Document]:
    """Quita los fragmentos sin ninguna letra ni número.

    El splitter corta por separadores de Markdown, y a veces un pedazo queda
    formado solo por sintaxis: en ``advanced/transports.md`` aparece un
    fragmento que es únicamente el cierre de un bloque de código. No
    responde ninguna pregunta y, como su embedding es casi aleatorio, puede
    colarse en los resultados. Los fragmentos que quedan conservan su número
    original, así el id sigue apuntando al mismo lugar del archivo.
    """
    utiles = [f for f in fragmentos if re.search(r"\w", f.page_content)]
    for fragmento in fragmentos:
        if not re.search(r"\w", fragmento.page_content):
            logger.info("Se descarta %s: no tiene texto", id_de(fragmento))
    return utiles


def armar_metadata(documentos: Sequence[Document], fragmentos: Sequence[Document]) -> list[MetadataFragmento]:
    """Arma la metadata que se guarda en Pinecone para cada fragmento.

    Campos:

    - ``texto``: el contenido del fragmento (ver docstring del módulo).
    - ``fuente``: archivo de origen, por ejemplo ``advanced/timeouts.md``.
      Es el "documento" que evalúa ``evaluate.py``.
    - ``seccion``: título de la sección donde empieza el fragmento.
    - ``pagina``: página del PDF donde empieza el fragmento. Solo en los
      PDF: un Markdown no tiene páginas, y Pinecone no acepta valores
      nulos en la metadata, así que en ese caso el campo no se incluye.
    - ``categoria``: ``rfc``, ``avanzado`` o ``general``, para filtrar.
    - ``fragmento``: número del fragmento dentro del archivo, desde 1.
    - ``tokens``: tamaño del fragmento en tokens del modelo de embeddings.
    """
    por_fuente = {str(doc.metadata["archivo"]): doc for doc in documentos}
    titulos_por_fuente = {fuente: titulos_de(doc) for fuente, doc in por_fuente.items()}
    metadatas: list[MetadataFragmento] = []
    for fragmento in fragmentos:
        fuente = str(fragmento.metadata["archivo"])
        inicio = int(fragmento.metadata["start_index"])
        metadata: MetadataFragmento = {
            "texto": fragmento.page_content,
            "fuente": fuente,
            "seccion": seccion_de(titulos_por_fuente[fuente], inicio, fuente),
            "categoria": categoria_de(fuente),
            "fragmento": int(fragmento.metadata["fragmento"]),
            "tokens": int(fragmento.metadata["tokens"]),
        }
        paginas = por_fuente[fuente].metadata.get("paginas")
        if paginas:
            metadata["pagina"] = pagina_de(paginas, inicio)
        metadatas.append(metadata)
    return metadatas


def armar_vectores(
    fragmentos: Sequence[Document], metadatas: Sequence[MetadataFragmento], embeddings: Embeddings
) -> list[VectorPinecone]:
    """Calcula el embedding de cada fragmento y arma las tuplas ``(id, valores, metadata)``.

    El id es estable (``advanced/timeouts.md#2``): reindexar el mismo
    archivo reemplaza sus vectores en vez de duplicarlos.
    """
    valores = embeddings.embed_documents([f.page_content for f in fragmentos])
    return [
        (id_de(fragmento), vector, metadata)
        for fragmento, vector, metadata in zip(fragmentos, valores, metadatas, strict=True)
    ]


def subir(index: IndiceVectorial, vectores: Sequence[VectorPinecone], namespace: str) -> int:
    """Sube los vectores en lotes de ``TAMANO_LOTE`` y devuelve cuántos se cargaron.

    Cada lote es una request; el SDK reintenta por su cuenta los lotes que
    fallan por red o por límite de tasa (429).

    Raises:
        IngestaIncompletaError: si algún lote no se cargó ni con los reintentos.
    """
    respuesta = index.upsert(
        vectors=vectores, namespace=namespace, batch_size=TAMANO_LOTE, show_progress=False
    )
    if respuesta.has_errors:
        raise IngestaIncompletaError(
            f"Fallaron {respuesta.failed_item_count} de {respuesta.total_item_count} "
            f"vectores ({respuesta.failed_batch_count} lotes). Volvé a correr la ingesta: "
            "detecta que faltan vectores y completa el namespace"
        )
    return respuesta.upserted_count


def vectores_en(index: IndiceVectorial, namespace: str) -> int:
    """Cantidad de vectores del namespace según las estadísticas del índice."""
    resumen = index.describe_index_stats().namespaces.get(namespace)
    return resumen.vector_count if resumen is not None else 0


def listar_ids(index: IndiceVectorial, namespace: str) -> set[str]:
    """Todos los ids del namespace, recorriendo las páginas de ``list``."""
    return {item.id for pagina in index.list(namespace=namespace) for item in pagina.vectors}


def borrar_ids(index: IndiceVectorial, ids: Sequence[str], namespace: str) -> None:
    """Borra los vectores ``ids`` del namespace, de a ``TAMANO_LOTE_BORRADO`` por request."""
    for inicio in range(0, len(ids), TAMANO_LOTE_BORRADO):
        index.delete(ids=ids[inicio : inicio + TAMANO_LOTE_BORRADO], namespace=namespace)


def indexar(
    index: IndiceVectorial,
    namespace: str,
    carpeta: Path,
    embeddings: Embeddings,
    contar_tokens: Callable[[str], int],
    *,
    reindexar: bool = False,
) -> ResultadoIngesta:
    """Indexa los documentos de ``carpeta`` en ``namespace``, salvo que ya estén completos.

    Primero fragmenta (es rápido: no calcula embeddings) para saber cuántos
    vectores debería haber. Después:

    - Si el namespace tiene exactamente esa cantidad y no se pidió
      ``reindexar``, no hace nada: subir de nuevo daría el mismo resultado.
    - Si tiene otra cantidad (por ejemplo, una ingesta anterior que falló a
      mitad de camino y dejó solo algunos lotes), reindexa para completarlo.
      Así, volver a correr la ingesta alcanza para reparar el namespace.
    - Con ``reindexar`` sube todo aunque la cantidad coincida. Hace falta
      cuando cambió el texto de un archivo pero no su cantidad de fragmentos.

    Cómo reemplaza sin dejar el namespace vacío: sube primero todos los
    vectores (los ids son estables, así que los que ya existían se
    reemplazan) y recién después borra los ids viejos que ya no
    corresponden a ningún fragmento. No usa ``delete_all``: en Pinecone
    serverless las escrituras se aplican de forma asíncrona, y un borrado
    total aplicado tarde podría llevarse vectores recién subidos. Borrando
    solo ids que no se subieron, el orden deja de importar.

    Recibe el índice, el modelo y el contador de tokens como parámetros para
    poder probarla con versiones falsas, sin red.

    Raises:
        FileNotFoundError: si ``carpeta`` no existe.
        ValueError: si ``carpeta`` no tiene archivos ``.md``.
        IngestaIncompletaError: si algún lote no se cargó.
    """
    documentos = cargar_documentos(carpeta)
    if not documentos:
        raise ValueError(f"No hay archivos .md ni .pdf en {carpeta}")
    todos = fragmentar(sin_mapas(documentos), contar_tokens)
    recalcular_inicios(documentos, todos)
    fragmentos = descartar_vacios(todos)

    existentes = vectores_en(index, namespace)
    if existentes == len(fragmentos) and not reindexar:
        logger.info(
            "El namespace %s ya tiene los %d fragmentos: no se reindexa (usá --reindexar)",
            namespace, existentes,
        )
        return ResultadoIngesta(
            omitida=True, namespace=namespace, archivos=len(documentos), fragmentos=existentes
        )
    if existentes > 0 and not reindexar:
        # Las estadísticas de Pinecone tardan unos segundos en actualizarse: justo
        # después de una ingesta pueden dar otro número. Reindexar igual es seguro,
        # porque reemplaza los mismos ids.
        logger.warning(
            "El namespace %s tiene %d vectores y se esperaban %d: se reindexa para completarlo",
            namespace, existentes, len(fragmentos),
        )

    ids_anteriores = listar_ids(index, namespace) if existentes > 0 else set()
    metadatas = armar_metadata(documentos, fragmentos)
    logger.info("Calculando %d embeddings", len(fragmentos))
    vectores = armar_vectores(fragmentos, metadatas, embeddings)
    cargados = subir(index, vectores, namespace)
    logger.info("Subidos %d fragmentos de %d archivos a %s", cargados, len(documentos), namespace)

    obsoletos = sorted(ids_anteriores - {id_vector for id_vector, _, _ in vectores})
    if obsoletos:
        logger.info("Borrando %d fragmentos que ya no existen en los documentos", len(obsoletos))
        borrar_ids(index, obsoletos, namespace)
    return ResultadoIngesta(
        omitida=False,
        namespace=namespace,
        archivos=len(documentos),
        fragmentos=cargados,
        borrados=len(obsoletos),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Corre la ingesta desde la línea de comandos y devuelve el código de salida."""
    parser = argparse.ArgumentParser(description="Sube la documentación de httpx a Pinecone.")
    parser.add_argument("--data", type=Path, default=CARPETA_POR_DEFECTO)
    parser.add_argument(
        "--reindexar",
        action="store_true",
        help="Vuelve a subir todo aunque el namespace esté completo (si cambió el texto de un archivo).",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    configurar_logs()
    try:
        config = ConfigRAG.from_env()
        pc: Pinecone = crear_cliente(config)
        # El setup va primero: si el índice no existe lo crea, y si es
        # incompatible frena antes de calcular ningún embedding.
        embeddings = crear_embeddings(pc)
        indice = asegurar_indice(pc, config.index_name, dimension_embeddings(embeddings))
        resultado = indexar(
            conectar(pc, indice.host),
            config.namespace,
            args.data,
            embeddings,
            crear_contador_tokens(),
            reindexar=args.reindexar,
        )
    except ValidationError as exc:
        print(f"Configuración inválida en el .env:\n{exc}", file=sys.stderr)
        return 1
    except (RAGHibridoError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except PineconeError as exc:
        print(f"Error de Pinecone ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(resultado.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
