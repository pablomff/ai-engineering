"""Modelo de embeddings y base vectorial ChromaDB, compartidos por ingesta y consulta.

Por qué un módulo compartido: el error más común en un RAG es indexar con un
modelo de embeddings y consultar con otro. Los vectores de modelos distintos
viven en espacios distintos, así que la distancia entre ellos no significa
nada y la búsqueda devuelve fragmentos al azar, sin ningún error visible.

Para evitarlo:

- ``crear_embeddings`` es el único lugar donde se elige el modelo, y lo usan
  tanto la ingesta como la consulta.
- La colección guarda en su metadata el nombre del modelo con el que se
  indexó, y ``abrir_para_consulta`` lo compara con el actual antes de buscar.

El modelo es local (HuggingFace, vía ``sentence-transformers``): no necesita
API key, no tiene costo por llamada y los mismos textos dan siempre los mismos
vectores. Anthropic no ofrece embeddings, así que el LLM y los embeddings son
de proveedores distintos, y eso está bien: solo importa que indexar y
consultar usen el mismo modelo.
"""

import functools
import logging
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from pipelines.rag.exceptions import BaseNoIndexadaError, EmbeddingsNoCoincidenError

logger = logging.getLogger(__name__)

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
"""Modelo de embeddings local.

Se eligió porque es multilingüe (los documentos están en español), chico
(384 dimensiones, unos 470 MB) y acepta textos de hasta 512 tokens. Muchos
modelos populares, como ``all-MiniLM-L6-v2``, cortan en 128 o 256 tokens: con
fragmentos de 500 tokens, el final de cada fragmento quedaría afuera del
embedding sin ningún aviso.
"""

# Los modelos de la familia e5 se entrenaron con estos prefijos: "query: " para
# la pregunta y "passage: " para los textos donde se busca. Sin ellos la
# búsqueda funciona peor.
PREFIJO_CONSULTA = "query: "
PREFIJO_PASAJE = "passage: "

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
"""Raíz del repositorio. Las rutas por defecto se anclan acá y no al
directorio actual: si no, correr el pipeline desde otra carpeta crearía una
base vacía nueva en esa carpeta en lugar de usar la existente."""

DIRECTORIO_POR_DEFECTO = RAIZ_PROYECTO / "vectorstore"
"""Carpeta local donde ChromaDB persiste la base."""

NOMBRE_COLECCION = "documentos"
"""Colección de ChromaDB donde se guardan los fragmentos."""

NOMBRE_COLECCION_TEMPORAL = "documentos__indexando"
"""Colección donde se arma una indexación nueva antes de reemplazar a la
anterior. Solo se renombra a ``NOMBRE_COLECCION`` cuando terminó entera."""

CLAVE_MODELO = "modelo_embeddings"
"""Clave de la metadata de la colección que guarda el modelo usado al indexar."""


@functools.cache
def crear_embeddings() -> Embeddings:
    """Crea el modelo de embeddings, una sola vez por proceso.

    Cargar el modelo tarda unos segundos (y la primera vez lo descarga), por
    eso se cachea. Los vectores se normalizan: así la distancia coseno y el
    producto interno dan el mismo orden, y los puntajes son comparables
    entre consultas.
    """
    logger.info("Cargando el modelo de embeddings %s", MODELO_EMBEDDINGS)
    return HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        encode_kwargs={"normalize_embeddings": True, "prompt": PREFIJO_PASAJE},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": PREFIJO_CONSULTA},
    )


@functools.cache
def crear_contador_tokens() -> Callable[[str], int]:
    """Devuelve una función que cuenta tokens con el tokenizer del modelo de embeddings.

    Por qué este tokenizer y no otro: "500 tokens" depende de quién cuente.
    El mismo texto en español da distinta cantidad de tokens con el
    tokenizer de OpenAI que con el de este modelo. Lo que importa es que el
    fragmento entre entero en el modelo de embeddings, así que se mide con
    su propio tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODELO_EMBEDDINGS)

    def contar_tokens(texto: str) -> int:
        """Cantidad de tokens de ``texto``, sin los tokens especiales de inicio y fin."""
        return len(tokenizer.encode(texto, add_special_tokens=False))

    return contar_tokens


def abrir_cliente(directorio: Path) -> ClientAPI:
    """Abre (o crea) la base ChromaDB persistida en ``directorio``.

    Si la carpeta no existe la crea, así que es para la ingesta; para
    consultar se usa ``abrir_cliente_existente``. Se desactiva la telemetría
    de ChromaDB para que la base sea completamente local.
    """
    return chromadb.PersistentClient(
        path=str(directorio), settings=ChromaSettings(anonymized_telemetry=False)
    )


def abrir_cliente_existente(directorio: Path) -> ClientAPI:
    """Abre la base de ``directorio`` sin crearla.

    Raises:
        BaseNoIndexadaError: si la carpeta no existe. Abrirla con
            ``abrir_cliente`` crearía una carpeta vacía en un lugar
            inesperado y ocultaría el problema real (una ruta equivocada).
    """
    if not directorio.is_dir():
        raise BaseNoIndexadaError(
            f"No existe la base vectorial en {directorio}: corré primero "
            "'uv run python -m pipelines.rag.ingesta'"
        )
    return abrir_cliente(directorio)


def leer_coleccion(cliente: ClientAPI, nombre: str = NOMBRE_COLECCION) -> Collection | None:
    """Devuelve la colección ``nombre``, o ``None`` si no existe."""
    try:
        return cliente.get_collection(nombre)
    except NotFoundError:
        return None


def modelo_de(coleccion: Collection) -> str | None:
    """Devuelve el modelo de embeddings con el que se indexó ``coleccion``."""
    modelo = (coleccion.metadata or {}).get(CLAVE_MODELO)
    return modelo if isinstance(modelo, str) else None


def crear_vectorstore(
    cliente: ClientAPI, embeddings: Embeddings, modelo: str, nombre: str = NOMBRE_COLECCION
) -> Chroma:
    """Crea la colección ``nombre`` y devuelve el vectorstore de LangChain sobre ella.

    La colección se crea con distancia coseno y con el nombre del modelo de
    embeddings en su metadata, para poder verificarlo al consultar. Es solo
    para la ingesta: para consultar, ``abrir_para_consulta`` abre la
    colección existente sin pasar estos argumentos, así no la modifica.
    """
    return Chroma(
        client=cliente,
        collection_name=nombre,
        embedding_function=embeddings,
        collection_metadata={CLAVE_MODELO: modelo},
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def abrir_para_consulta(cliente: ClientAPI, embeddings: Embeddings, modelo: str) -> Chroma:
    """Abre la base para buscar, verificando antes que sirva.

    Raises:
        BaseNoIndexadaError: si la colección no existe o está vacía.
        EmbeddingsNoCoincidenError: si se indexó con otro modelo de embeddings.
    """
    coleccion = leer_coleccion(cliente)
    cantidad = coleccion.count() if coleccion is not None else 0
    if coleccion is None or cantidad == 0:
        raise BaseNoIndexadaError(
            "La base vectorial está vacía: corré primero "
            "'uv run python -m pipelines.rag.ingesta'"
        )
    modelo_indexado = modelo_de(coleccion)
    if modelo_indexado != modelo:
        raise EmbeddingsNoCoincidenError(
            f"La base se indexó con '{modelo_indexado}' y se está consultando con "
            f"'{modelo}': reindexá con 'uv run python -m pipelines.rag.ingesta --reindexar'"
        )
    logger.info("Base abierta: %d fragmentos indexados con %s", cantidad, modelo)
    # Solo lectura: sin metadata ni configuración de creación, que en el camino
    # "obtener o crear" de Chroma podrían reescribir la colección existente.
    return Chroma(
        client=cliente,
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
