"""Setup de Pinecone: crea el índice serverless si no existe y verifica que sirva.

Uso::

    uv run python -m pipelines.rag_hibrido.indice

Qué hace:

1. Mide la dimensión del modelo de embeddings de Pinecone (1024 para
   ``multilingual-e5-large``) en vez de escribirla a mano.
2. Si el índice ``INDEX_NAME`` no existe, lo crea en modo serverless
   (AWS ``us-east-1``, la región del plan gratuito) con esa dimensión y
   métrica coseno, y espera a que esté listo.
3. Si ya existe, verifica que la dimensión, la métrica y el modelo de
   embeddings coincidan. Es el error más común con Pinecone: un índice
   creado para OpenAI (1536) no acepta vectores de 1024.

Es idempotente: correrlo dos veces no crea nada nuevo.
"""

import logging
import sys
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv
from pinecone import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    Pinecone,
    PineconeError,
    QueryResponse,
    RetryConfig,
    ServerlessSpec,
    UpsertResponse,
)
from pydantic import BaseModel, ValidationError

from langchain_core.embeddings import Embeddings

from pipelines.rag_hibrido.embeddings import MODELO_EMBEDDINGS, crear_embeddings
from pipelines.rag_hibrido.config import ConfigRAG
from pipelines.rag_hibrido.exceptions import (
    IndiceIncompatibleError,
    IndiceNoListoError,
    RAGHibridoError,
)

logger = logging.getLogger(__name__)

METRICA = "cosine"
"""Métrica de similitud del índice.

El modelo de embeddings devuelve vectores normalizados (norma 1), así que
coseno y producto interno ordenan igual. Se elige coseno porque es la que
recomiendan los modelos de la familia e5 y porque sigue siendo correcta si
mañana se cambia a un modelo que no normaliza: compara solo la dirección de
los vectores, no su largo.
"""

NUBE = "aws"
REGION = "us-east-1"
"""Nube y región del índice serverless. ``aws``/``us-east-1`` es la única
combinación disponible en el plan gratuito de Pinecone."""

ETIQUETA_MODELO = "modelo_embeddings"
"""Tag del índice que guarda con qué modelo de embeddings se creó.

Dos modelos distintos pueden tener la misma dimensión (por ejemplo, varios
modelos de 1024) y la verificación de dimensión no los distinguiría. La
etiqueta sí: si no coincide, buscar devolvería resultados al azar.
"""

REINTENTOS = RetryConfig(max_retries=3, backoff_factor=0.5, max_wait=30.0)
"""Reintentos nativos del SDK: espera exponencial con jitter ante 408, 429
(límite de tasa) y 5xx. El proyecto no permite bucles de reintento propios."""

TIMEOUT_SEGUNDOS = 30.0
"""Tiempo máximo de cada request HTTP a Pinecone."""


MetadataFragmento = dict[str, str | int]
"""Metadata de un vector en Pinecone: texto, fuente, sección, categoría, etc."""

VectorPinecone = tuple[str, list[float], MetadataFragmento]
"""Un vector listo para el upsert: ``(id, valores, metadata)``."""


@runtime_checkable
class IndiceVectorial(Protocol):
    """Las operaciones del índice de Pinecone que usa este pipeline.

    Es un ``Protocol``: describe la forma, no la implementación. El ``Index``
    real de Pinecone la cumple, y los tests pasan un índice falso en memoria
    con los mismos métodos, sin red ni cuenta.
    """

    def upsert(
        self,
        *,
        vectors: Sequence[VectorPinecone],
        namespace: str = "",
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Inserta o reemplaza vectores en un namespace."""
        ...

    def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float] | None = None,
        namespace: str = "",
        include_metadata: bool = False,
    ) -> QueryResponse:
        """Devuelve los ``top_k`` vectores más parecidos a ``vector``."""
        ...

    def list(self, *, namespace: str = "") -> Iterator[ListResponse]:
        """Recorre los ids de un namespace, una página por iteración."""
        ...

    def fetch(self, *, ids: Sequence[str], namespace: str = "") -> FetchResponse:
        """Trae vectores (con su metadata) por id."""
        ...

    def describe_index_stats(self) -> DescribeIndexStatsResponse:
        """Cantidad de vectores del índice, total y por namespace."""
        ...

    def delete(self, *, ids: Sequence[str] | None = None, namespace: str = "") -> None:
        """Borra los vectores ``ids`` del namespace."""
        ...


class ResultadoIndice(BaseModel):
    """Resumen del setup, para mostrar en consola y para los tests."""

    nombre: str
    creado: bool
    """``True`` si esta corrida creó el índice; ``False`` si ya existía."""
    dimension: int
    metrica: str
    host: str


def valor_etiqueta(modelo: str) -> str:
    """Convierte el nombre del modelo en un valor válido para un tag de Pinecone.

    Los tags no aceptan ``/``: un modelo como ``intfloat/multilingual-e5-large``
    quedaría como ``intfloat_multilingual-e5-large``.
    """
    return modelo.replace("/", "_")


def crear_cliente(config: ConfigRAG) -> Pinecone:
    """Crea el cliente de Pinecone con timeout y reintentos explícitos.

    El constructor no hace llamadas de red: la key se valida recién en la
    primera operación.
    """
    return Pinecone(
        api_key=config.pinecone_api_key.get_secret_value(),
        timeout=TIMEOUT_SEGUNDOS,
        retry_config=REINTENTOS,
    )


def dimension_embeddings(embeddings: Embeddings) -> int:
    """Dimensión de los vectores del modelo de embeddings, medida y no escrita a mano.

    Se calcula pidiendo el embedding de un texto de prueba (una llamada a
    Pinecone, de pocos tokens). Si mañana se cambia el modelo, la dimensión
    del índice nuevo se ajusta sola y la verificación de un índice existente
    detecta la diferencia.
    """
    return len(embeddings.embed_query("dimension"))


def asegurar_indice(pc: Pinecone, nombre: str, dimension: int) -> ResultadoIndice:
    """Crea el índice ``nombre`` si no existe y verifica que sea compatible.

    ``create_index`` espera por defecto a que el índice esté listo antes de
    devolver, sondeando su estado cada pocos segundos.

    Raises:
        IndiceIncompatibleError: si existe con otra dimensión, otra métrica u
            otro modelo de embeddings.
        IndiceNoListoError: si existe pero no está listo.
        PineconeError: si falla la API después de los reintentos del SDK.
    """
    creado = False
    if nombre not in pc.list_indexes().names():
        logger.info(
            "Creando el índice %s (serverless %s/%s, %d dimensiones, %s)",
            nombre, NUBE, REGION, dimension, METRICA,
        )
        pc.create_index(
            name=nombre,
            dimension=dimension,
            metric=METRICA,
            spec=ServerlessSpec(cloud=NUBE, region=REGION),
            tags={ETIQUETA_MODELO: valor_etiqueta(MODELO_EMBEDDINGS)},
        )
        creado = True
    else:
        logger.info("El índice %s ya existe: se verifica que sea compatible", nombre)

    descripcion = pc.describe_index(nombre)
    verificar_compatible(
        nombre, descripcion.dimension, descripcion.metric, descripcion.tags, dimension
    )
    if not descripcion.status.ready:
        raise IndiceNoListoError(
            f"El índice {nombre} no está listo (estado: {descripcion.status.state})"
        )
    return ResultadoIndice(
        nombre=nombre,
        creado=creado,
        dimension=dimension,
        metrica=METRICA,
        host=descripcion.host,
    )


def conectar(pc: Pinecone, host: str) -> IndiceVectorial:
    """Abre la conexión al índice por su host.

    Se usa el host (que devuelve ``asegurar_indice``) y no el nombre: con el
    nombre, el SDK haría una llamada extra a la API para averiguar el host.
    """
    return pc.Index(host=host)


def verificar_compatible(
    nombre: str,
    dimension_indice: int | None,
    metrica_indice: str | None,
    tags: dict[str, str] | None,
    dimension_esperada: int,
) -> None:
    """Verifica que un índice existente sirva para el modelo de embeddings actual.

    Está separada de ``asegurar_indice`` para poder probarla sin Pinecone.

    Raises:
        IndiceIncompatibleError: si la dimensión, la métrica o el modelo no
            coinciden. Si el índice no tiene la etiqueta del modelo (por
            ejemplo, se creó desde la consola web), solo se avisa en el log.
    """
    if dimension_indice != dimension_esperada:
        raise IndiceIncompatibleError(
            f"El índice {nombre} tiene dimensión {dimension_indice} y el modelo "
            f"{MODELO_EMBEDDINGS} produce vectores de {dimension_esperada}: "
            "borralo desde la consola de Pinecone o usá otro INDEX_NAME"
        )
    if metrica_indice != METRICA:
        raise IndiceIncompatibleError(
            f"El índice {nombre} usa la métrica {metrica_indice} y se esperaba {METRICA}"
        )
    modelo_indice = (tags or {}).get(ETIQUETA_MODELO)
    if modelo_indice is None:
        logger.warning(
            "El índice %s no tiene la etiqueta %s: no se puede verificar el modelo",
            nombre, ETIQUETA_MODELO,
        )
    elif modelo_indice != valor_etiqueta(MODELO_EMBEDDINGS):
        raise IndiceIncompatibleError(
            f"El índice {nombre} se creó para el modelo '{modelo_indice}' y el actual es "
            f"'{MODELO_EMBEDDINGS}': los vectores no serían comparables"
        )


def configurar_logs() -> None:
    """Configura los logs de los scripts y silencia los de las librerías.

    Pinecone, httpx y HuggingFace loguean cada request en nivel INFO y
    taparían los mensajes del pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("pinecone", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    """Corre el setup desde la línea de comandos y devuelve el código de salida."""
    del argv  # Sin argumentos: todo sale del .env.
    load_dotenv()
    configurar_logs()
    try:
        config = ConfigRAG.from_env()
        pc = crear_cliente(config)
        dimension = dimension_embeddings(crear_embeddings(pc))
        resultado = asegurar_indice(pc, config.index_name, dimension)
    except ValidationError as exc:
        print(f"Configuración inválida en el .env:\n{exc}", file=sys.stderr)
        return 1
    except RAGHibridoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except PineconeError as exc:
        # No se imprime el mensaje completo: podría incluir datos de la request.
        print(f"Error de Pinecone ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(resultado.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
