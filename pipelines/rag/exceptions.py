"""Excepciones propias del pipeline RAG.

Están en un módulo aparte porque las usan las dos etapas del pipeline: la
ingesta (base inexistente, modelo de embeddings distinto) y la consulta
(respuesta inválida, falla del proveedor). Quien usa el pipeline puede
manejar cualquier falla con un solo ``except RAGError``, sin conocer las
excepciones de LangChain, de ChromaDB ni de los SDKs.
"""

from llm_client import LLMError


class RAGError(Exception):
    """Error base del pipeline RAG."""


class RAGConfigError(RAGError):
    """La configuración no permite consultar (por ejemplo, falta
    ``ANTHROPIC_API_KEY``)."""


class BaseNoIndexadaError(RAGConfigError):
    """La base vectorial no existe o está vacía: hay que correr la ingesta.

    Es un error de configuración y no una respuesta "No lo sé": sin
    documentos indexados, el modelo no tendría ningún contexto y cualquier
    pregunta terminaría sin respuesta, ocultando el problema real.
    """


class EmbeddingsNoCoincidenError(RAGConfigError):
    """La base se indexó con un modelo de embeddings distinto al actual.

    Los vectores de dos modelos distintos no son comparables: la búsqueda
    devolvería fragmentos al azar. Se resuelve reindexando.
    """


class RecursosNoDisponiblesError(RAGConfigError):
    """No se pudo cargar el modelo de embeddings o abrir la base.

    Por ejemplo: la primera vez sin conexión (el modelo no está en la caché
    de HuggingFace) o una base persistida dañada. Envuelve el ``OSError`` o
    el error de ChromaDB original, que queda como causa.
    """


class RespuestaInvalidaError(RAGError):
    """El modelo respondió, pero la salida no es confiable.

    Cubre JSON mal formado, respuestas cortadas por el límite de tokens,
    respuestas sin referencias y referencias a fragmentos que no se
    recuperaron (señal de que el modelo inventó la fuente).
    """


class ModeloNegativaError(RAGError):
    """El modelo se negó a responder (``stop_reason == "refusal"``)."""


class RAGProveedorError(RAGError):
    """Falló la llamada a Anthropic después de los reintentos del SDK.

    Conserva la ``LLMError`` de ``llm_client`` en ``error``, con su ``kind``
    y si es reintentable más tarde.
    """

    def __init__(self, error: LLMError) -> None:
        """Guarda la ``LLMError`` traducida y usa su mensaje (sin datos sensibles)."""
        super().__init__(error.message)
        self.error: LLMError = error
