"""Ingesta: carga documentos, los fragmenta y los guarda en ChromaDB.

Uso::

    uv run python -m pipelines.rag.ingesta                 # indexa ./data en ./vectorstore
    uv run python -m pipelines.rag.ingesta --reindexar     # rehace la base y reemplaza la anterior
    uv run python -m pipelines.rag.ingesta --data otra/carpeta --vectorstore otra/base

Flujo::

    data/*.md, data/*.txt
      -> cargar_documentos      (un Document por archivo, con su nombre en la metadata)
      -> fragmentar             (RecursiveCharacterTextSplitter: 500 tokens, 50 de solapamiento)
      -> Chroma.add_documents   (calcula el embedding de cada fragmento y lo persiste)

Si la base ya tiene documentos, no se vuelve a indexar: calcular los
embeddings de nuevo cuesta tiempo (y dinero, con un modelo pago) y daría el
mismo resultado. Para forzarlo está ``--reindexar``, que se usa cuando cambian
los documentos o el modelo de embeddings.

Las rutas por defecto (``data/`` y ``vectorstore/``) son las de la raíz del
repositorio, desde cualquier carpeta en que se corra.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel

from pipelines.rag.exceptions import EmbeddingsNoCoincidenError
from pipelines.rag.vectorstore import (
    DIRECTORIO_POR_DEFECTO,
    MODELO_EMBEDDINGS,
    NOMBRE_COLECCION,
    NOMBRE_COLECCION_TEMPORAL,
    RAIZ_PROYECTO,
    abrir_cliente,
    crear_contador_tokens,
    crear_embeddings,
    crear_vectorstore,
    leer_coleccion,
    modelo_de,
)

logger = logging.getLogger(__name__)

CARPETA_POR_DEFECTO = RAIZ_PROYECTO / "data"
"""Carpeta de donde se leen los documentos."""

EXTENSIONES = frozenset({".md", ".txt"})
"""Extensiones de archivo que se indexan."""

TOKENS_POR_FRAGMENTO = 500
"""Tamaño máximo de cada fragmento, en tokens del modelo de embeddings.

Sumando el prefijo ``"passage: "`` y los tokens especiales, queda por debajo
del límite de 512 del modelo, así que ningún fragmento se trunca. Es lo
bastante grande para que un fragmento tenga una idea completa (una sección
con su contexto) y lo bastante chico para que el embedding no mezcle
demasiados temas.
"""

TOKENS_DE_SOLAPAMIENTO = 50
"""Tokens que un fragmento puede repetir del anterior.

Sirve para que una idea que cae justo en el corte no quede partida en dos
fragmentos que, por separado, no la explican.
"""


class ResultadoIngesta(BaseModel):
    """Resumen de una ingesta, para mostrar y para los tests."""

    omitida: bool
    """``True`` si la base ya existía y no se volvió a indexar."""
    archivos: int
    """Cantidad de archivos distintos en la base."""
    fragmentos: int
    """Cantidad de fragmentos en la base."""


def cargar_documentos(carpeta: Path) -> list[Document]:
    """Lee los ``.md`` y ``.txt`` de ``carpeta`` (sin subcarpetas), en orden alfabético.

    Cada archivo se convierte en un ``Document`` con su nombre en la
    metadata (``archivo``), que después permite citar la fuente. Los archivos
    vacíos se omiten.

    Raises:
        FileNotFoundError: si ``carpeta`` no existe.
    """
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {carpeta}")
    documentos: list[Document] = []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES or not ruta.is_file():
            continue
        texto = ruta.read_text(encoding="utf-8").strip()
        if not texto:
            logger.warning("Se omite %s: está vacío", ruta.name)
            continue
        documentos.append(Document(page_content=texto, metadata={"archivo": ruta.name}))
    return documentos


def crear_splitter(
    contar_tokens: Callable[[str], int], tamano: int = TOKENS_POR_FRAGMENTO
) -> RecursiveCharacterTextSplitter:
    """Crea el splitter que parte los documentos en fragmentos de hasta ``tamano`` tokens.

    Es "estratégico" en dos sentidos:

    - **Mide en tokens, no en caracteres**, con ``contar_tokens`` (el
      tokenizer del modelo de embeddings).
    - **Corta por estructura antes que por tamaño.** Prueba los separadores
      en orden: primero títulos de Markdown (``#``, ``##``...), después
      párrafos, después líneas, después palabras. Solo baja al siguiente si
      el pedazo todavía no entra. Así un fragmento tiende a ser una sección
      entera y no termina a mitad de una oración.

    Recibe la función de conteo como parámetro para poder probarlo sin
    descargar el tokenizer.
    """
    return RecursiveCharacterTextSplitter(
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN),
        is_separator_regex=True,
        chunk_size=tamano,
        chunk_overlap=TOKENS_DE_SOLAPAMIENTO,
        length_function=contar_tokens,
        add_start_index=True,
    )


def partir_documento(documento: Document, contar_tokens: Callable[[str], int]) -> list[Document]:
    """Parte un documento garantizando que cada fragmento, ya unido, tenga a lo sumo 500 tokens.

    Por qué hace falta verificar después de partir: el splitter decide el
    tamaño sumando los tokens de cada pedazo por separado, pero los tokens
    no se suman exactamente. El texto unido puede tokenizarse distinto en
    los bordes entre pedazos y quedar unos tokens por encima del límite. Si
    pasa, se vuelve a partir el documento con un tamaño objetivo menor,
    hasta que todos los fragmentos entren de verdad.

    Raises:
        ValueError: si no logra fragmentos que entren (no debería ocurrir:
            el splitter puede bajar hasta cortar por caracteres).
    """
    tamano = TOKENS_POR_FRAGMENTO
    while tamano > TOKENS_DE_SOLAPAMIENTO:
        partes = crear_splitter(contar_tokens, tamano).split_documents([documento])
        exceso = max(contar_tokens(p.page_content) for p in partes) - TOKENS_POR_FRAGMENTO
        if exceso <= 0:
            return partes
        logger.info(
            "%s: un fragmento quedó %d tokens por encima del límite, se reparte",
            documento.metadata["archivo"],
            exceso,
        )
        tamano -= max(exceso, 10)
    raise ValueError(f"No se pudo partir {documento.metadata['archivo']} en fragmentos válidos")


def fragmentar(
    documentos: Sequence[Document], contar_tokens: Callable[[str], int]
) -> list[Document]:
    """Parte cada documento y numera sus fragmentos desde 1.

    Cada fragmento conserva el ``archivo`` de origen y suma ``fragmento``
    (su número dentro del archivo), ``start_index`` (dónde empieza en el
    texto original) y ``tokens`` (contados sobre el texto ya unido).
    ``archivo`` + ``fragmento`` es lo que el modelo usa para citar la fuente.
    """
    fragmentos: list[Document] = []
    for documento in documentos:
        partes = partir_documento(documento, contar_tokens)
        for numero, parte in enumerate(partes, start=1):
            parte.metadata["fragmento"] = numero
            parte.metadata["tokens"] = contar_tokens(parte.page_content)
        logger.info(
            "%s: %d fragmentos (%s tokens)",
            documento.metadata["archivo"],
            len(partes),
            ", ".join(str(p.metadata["tokens"]) for p in partes),
        )
        fragmentos.extend(partes)
    return fragmentos


def id_de(fragmento: Document) -> str:
    """ID estable de un fragmento en la base, por ejemplo ``"arquitectura.md#2"``."""
    return f"{fragmento.metadata['archivo']}#{fragmento.metadata['fragmento']}"


def _resumen_existente(coleccion: Collection) -> ResultadoIngesta:
    """Arma el resultado de una ingesta omitida a partir de lo que ya hay en la base."""
    metadatas = coleccion.get(include=["metadatas"])["metadatas"] or []
    archivos = {m.get("archivo") for m in metadatas}
    return ResultadoIngesta(omitida=True, archivos=len(archivos), fragmentos=len(metadatas))


def _borrar_si_existe(cliente: ClientAPI, nombre: str) -> None:
    """Borra la colección ``nombre`` si existe."""
    if leer_coleccion(cliente, nombre) is not None:
        cliente.delete_collection(nombre)


def indexar(
    carpeta: Path,
    cliente: ClientAPI,
    embeddings: Embeddings,
    modelo: str,
    contar_tokens: Callable[[str], int],
    *,
    reindexar: bool = False,
) -> ResultadoIngesta:
    """Indexa los documentos de ``carpeta`` en la base, salvo que ya estén indexados.

    Si la colección ya tiene fragmentos y no se pidió ``reindexar``, no hace
    nada (pero verifica que se hayan indexado con el mismo modelo). Si no,
    arma una indexación nueva.

    La indexación es atómica: se arma entera en una colección temporal y
    recién al final reemplaza a la anterior. Así, si algo falla a mitad de
    camino (sin conexión para bajar el modelo, Ctrl-C, falta de memoria):

    - la base anterior queda intacta, en vez de quedar borrada o vacía;
    - nunca queda una base a medio indexar con el nombre definitivo, que
      una corrida posterior tomaría por completa.

    Raises:
        FileNotFoundError: si ``carpeta`` no existe.
        ValueError: si ``carpeta`` no tiene archivos ``.md`` ni ``.txt``.
        EmbeddingsNoCoincidenError: si la base existente se indexó con otro
            modelo y no se pidió ``reindexar``.
    """
    coleccion = leer_coleccion(cliente)
    if coleccion is not None and coleccion.count() > 0 and not reindexar:
        modelo_indexado = modelo_de(coleccion)
        if modelo_indexado != modelo:
            raise EmbeddingsNoCoincidenError(
                f"La base existente se indexó con '{modelo_indexado}' y el modelo actual es "
                f"'{modelo}': usá --reindexar"
            )
        resultado = _resumen_existente(coleccion)
        logger.info(
            "La base ya tiene %d fragmentos de %d archivos: no se reindexa (usá --reindexar)",
            resultado.fragmentos,
            resultado.archivos,
        )
        return resultado

    documentos = cargar_documentos(carpeta)
    if not documentos:
        raise ValueError(f"No hay archivos .md ni .txt en {carpeta}")
    fragmentos = fragmentar(documentos, contar_tokens)

    # Restos de una indexación anterior que se cortó a mitad de camino.
    _borrar_si_existe(cliente, NOMBRE_COLECCION_TEMPORAL)
    temporal = crear_vectorstore(cliente, embeddings, modelo, NOMBRE_COLECCION_TEMPORAL)
    try:
        temporal.add_documents(fragmentos, ids=[id_de(f) for f in fragmentos])
    except BaseException:
        _borrar_si_existe(cliente, NOMBRE_COLECCION_TEMPORAL)
        raise

    if coleccion is not None:
        logger.info("Reemplazando la colección anterior")
        cliente.delete_collection(NOMBRE_COLECCION)
    cliente.get_collection(NOMBRE_COLECCION_TEMPORAL).modify(name=NOMBRE_COLECCION)
    logger.info("Indexados %d fragmentos de %d archivos", len(fragmentos), len(documentos))
    return ResultadoIngesta(omitida=False, archivos=len(documentos), fragmentos=len(fragmentos))


class _EmbeddingsPerezosos(Embeddings):
    """Delega en ``crear_embeddings()`` recién cuando hace falta un vector.

    Cargar el modelo tarda varios segundos. Si la base ya existe, la ingesta
    no calcula ningún embedding, así que no tiene sentido cargarlo.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Calcula los embeddings de los fragmentos con el modelo real."""
        return crear_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Calcula el embedding de una consulta con el modelo real."""
        return crear_embeddings().embed_query(text)


def main(argv: Sequence[str] | None = None) -> int:
    """Corre la ingesta desde la línea de comandos y devuelve el código de salida."""
    parser = argparse.ArgumentParser(description="Indexa documentos en ChromaDB para el RAG.")
    parser.add_argument("--data", type=Path, default=CARPETA_POR_DEFECTO)
    parser.add_argument("--vectorstore", type=Path, default=DIRECTORIO_POR_DEFECTO)
    parser.add_argument(
        "--reindexar", action="store_true", help="Vuelve a indexar todo y reemplaza la base."
    )
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Logs HTTP de HuggingFace (una línea por archivo del modelo): tapan los de la ingesta.
    for ruidoso in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    try:
        cliente = abrir_cliente(args.vectorstore)
        # Si la base ya existe, indexar no necesita el modelo ni el tokenizer:
        # se pasan como funciones perezosas para no cargarlos en vano.
        resultado = indexar(
            args.data,
            cliente,
            _EmbeddingsPerezosos(),
            MODELO_EMBEDDINGS,
            lambda texto: crear_contador_tokens()(texto),
            reindexar=args.reindexar,
        )
    except (ValueError, EmbeddingsNoCoincidenError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ChromaError) as exc:
        # OSError cubre la carpeta inexistente y las fallas al bajar o cargar el
        # modelo de HuggingFace. La base anterior, si había, queda intacta.
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print(resultado.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
