"""Infraestructura vectorial en Pinecone Serverless.

Crea (si no existe) un índice serverless, carga vectores de prueba con
metadatos dentro de un namespace, devuelve las estadísticas del índice y
consulta ese namespace filtrando por metadatos.

Por qué ``asyncio.to_thread``: el cliente ``Pinecone`` es síncrono (sus llamadas
HTTP bloquean y, al crear el índice, espera con ``time.sleep``). Llamarlo
directo dentro de ``async def`` congelaría el event loop, así que cada llamada
que va a la red se ejecuta en un hilo aparte.

Requiere la variable de entorno ``PINECONE_API_KEY`` (se puede poner en ``.env``).

Uso::

    uv run python ejercicios/pinecone_setup.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pinecone import (
    PineconeError,
    Pinecone,
    RateLimitError,
    RetryConfig,
    ServerlessSpec,
)

if TYPE_CHECKING:
    # Solo para los type hints; no se importan en tiempo de ejecución.
    from pinecone import DescribeIndexStatsResponse, QueryResponse

logger = logging.getLogger(__name__)

NAMESPACE = "dev-environment"

# Métrica de similitud. Los embeddings de OpenAI (1536 dimensiones) vienen
# normalizados a norma 1, y OpenAI recomienda cosine para compararlos. Con
# vectores normalizados cosine y dot product dan el mismo ranking, pero cosine
# sigue siendo correcto si mañana se usa un modelo que no normaliza, porque
# compara solo la dirección de los vectores y no su magnitud. Euclidean
# también ordenaría igual con vectores normalizados, pero su "distancia" (menor
# es mejor) es menos directa de interpretar que un score de similitud.
METRIC = "cosine"

# Tamaño de lote para el upsert. Pinecone limita cada request en cantidad de
# vectores y en tamaño (2 MB); con 1536 floats por vector, 100 por lote queda
# holgado. Con 3 vectores es un solo lote, pero el código ya escala.
BATCH_SIZE = 100

# Reintentos nativos del SDK: backoff exponencial con jitter ante 408, 429
# (rate limit) y 5xx. El CLAUDE.md prohíbe reintentos manuales, así que la
# política se configura acá y no con un bucle propio.
RETRY_CONFIG = RetryConfig(max_retries=3, backoff_factor=0.5, max_wait=30.0)
TIMEOUT_SECONDS = 30.0


class IngestaIncompletaError(Exception):
    """Algunos lotes del upsert no se cargaron aunque el SDK agotó sus reintentos.

    Existe porque un upsert por lotes no lanza excepción ante un lote fallido:
    lo informa en la respuesta, y sin este chequeo la falla pasaría inadvertida.
    """


def crear_cliente() -> Pinecone:
    """Crea el cliente de Pinecone con timeout y política de reintentos explícitos.

    Por qué explícitos: dejan documentado en el código cuánto se espera y cuántas
    veces se reintenta ante errores transitorios o límites de tasa.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable de entorno PINECONE_API_KEY.")
    return Pinecone(api_key=api_key, timeout=TIMEOUT_SECONDS, retry_config=RETRY_CONFIG)


def _asegurar_indice(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Crea el índice si no existe y valida que sea coherente con los embeddings.

    Es síncrona a propósito: agrupa las llamadas al plano de control para
    correrlas juntas en un solo hilo con ``asyncio.to_thread``.

    Raises:
        ValueError: si el índice ya existe con otra métrica u otra dimensión.
            Cargar vectores ahí fallaría (dimensión) o daría scores que no
            corresponden al modelo de embeddings (métrica).
        RuntimeError: si el índice no quedó listo tras crearlo.
    """
    if index_name not in pc.list_indexes().names():
        logger.info("Creando índice %s (%s, %d dims)", index_name, METRIC, dimension)
        # create_index espera por defecto a que el índice esté listo.
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    descripcion = pc.describe_index(index_name)
    if descripcion.metric != METRIC or descripcion.dimension != dimension:
        raise ValueError(
            f"El índice {index_name} existe con metric={descripcion.metric} y "
            f"dimension={descripcion.dimension}; se esperaba metric={METRIC} y "
            f"dimension={dimension}."
        )
    # Verificación explícita del estado: si la creación terminó por timeout o el
    # índice venía de antes en otro estado, no tiene sentido seguir.
    if not descripcion.status.ready:
        raise RuntimeError(
            f"El índice {index_name} no está listo (estado: {descripcion.status.state})."
        )


async def setup_vector_infrastructure(
    index_name: str, dimension: int
) -> DescribeIndexStatsResponse:
    """Configura el índice serverless, hace una carga inicial y devuelve sus estadísticas.

    Es idempotente: si el índice existe no se vuelve a crear, y los vectores usan
    ids fijos, así que un segundo upsert los sobrescribe en vez de duplicarlos.
    """
    # 1. Inicializar el cliente de Pinecone (no hace llamadas de red).
    pc = crear_cliente()

    try:
        # 2. Crear el índice si no existe y verificar que esté listo.
        await asyncio.to_thread(_asegurar_indice, pc, index_name, dimension)

        # 3. Conectar al índice. Pinecone resuelve el host a partir del nombre,
        # lo que implica una llamada a la API.
        index = await asyncio.to_thread(pc.Index, index_name)

        # 4. Upsert por lotes en un namespace: tuplas (id, vector, metadata).
        vectores: list[tuple[str, list[float], dict[str, str]]] = [
            ("vec1", [0.1] * dimension, {"topic": "infraestructura", "priority": "high"}),
            ("vec2", [0.2] * dimension, {"topic": "seguridad", "priority": "medium"}),
            ("vec3", [0.3] * dimension, {"topic": "observabilidad", "priority": "low"}),
        ]
        respuesta = await asyncio.to_thread(
            index.upsert,
            vectors=vectores,
            namespace=NAMESPACE,
            batch_size=BATCH_SIZE,
            show_progress=False,
        )
        if respuesta.has_errors:
            raise IngestaIncompletaError(
                f"Fallaron {respuesta.failed_item_count} de "
                f"{respuesta.total_item_count} vectores "
                f"({respuesta.failed_batch_count} lotes)."
            )
        logger.info("Cargados %d vectores en %s", respuesta.upserted_count, NAMESPACE)

        # 5. Retornar las estadísticas. Pueden tardar unos segundos en reflejar
        # el upsert (consistencia eventual).
        return await asyncio.to_thread(index.describe_index_stats)
    except RateLimitError:
        # Llega acá solo si el SDK ya agotó sus reintentos con backoff.
        logger.error("Pinecone sigue limitando la tasa tras %d reintentos", RETRY_CONFIG.max_retries)
        raise
    except PineconeError as error:
        # Errores de red, timeouts o respuestas de error de la API.
        logger.error("Error de Pinecone: %s", type(error).__name__)
        raise


async def consultar_namespace(
    index_name: str,
    vector: list[float],
    namespace: str,
    filtro: dict[str, dict[str, str]] | None = None,
    top_k: int = 3,
) -> QueryResponse:
    """Busca los vectores más similares dentro de un único namespace.

    El namespace es el límite de aislamiento: la búsqueda nunca ve vectores de
    otros namespaces. El filtro por metadatos segmenta además dentro de él.
    """
    pc = crear_cliente()
    try:
        index = await asyncio.to_thread(pc.Index, index_name)
        return await asyncio.to_thread(
            index.query,
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filtro,
            include_metadata=True,
        )
    except PineconeError as error:
        logger.error("Error de Pinecone al consultar: %s", type(error).__name__)
        raise


async def main() -> None:
    """Arma la infraestructura y muestra estadísticas y consultas por namespace."""
    index_name = "mi-indice-serverless"
    dimension = 1536  # Ejemplo con dimensión 1536 (OpenAI)

    stats = await setup_vector_infrastructure(index_name, dimension)
    print(stats)
    print("Namespaces:", stats.namespaces)

    vector_consulta = [0.1] * dimension
    filtro_prioridad_alta = {"priority": {"$eq": "high"}}

    # Dentro de dev-environment y filtrando por prioridad: solo vec1.
    resultado = await consultar_namespace(
        index_name, vector_consulta, NAMESPACE, filtro_prioridad_alta
    )
    print(f"\nQuery en '{NAMESPACE}' con priority=high:")
    for match in resultado.matches:
        print(f"  {match.id}  score={match.score:.4f}  {match.metadata}")

    # La misma consulta en otro namespace no devuelve nada: los datos de
    # dev-environment quedan aislados.
    otro = await consultar_namespace(index_name, vector_consulta, "prod-environment")
    print(f"\nQuery en 'prod-environment': {len(otro.matches)} resultados")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # El SDK y httpx loguean cada request en INFO; se silencian para que la
    # salida muestre solo lo de este script.
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
