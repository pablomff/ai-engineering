"""Excepciones propias del RAG híbrido.

Quien usa el pipeline puede manejar cualquier falla con un solo
``except RAGHibridoError``, sin conocer las excepciones de Pinecone ni de
LangChain.
"""


class RAGHibridoError(Exception):
    """Error base del RAG híbrido."""


class IndiceIncompatibleError(RAGHibridoError):
    """El índice existe pero no sirve para este modelo de embeddings.

    Pasa si se creó con otra dimensión (por ejemplo 1536, la de OpenAI,
    cuando el modelo de Pinecone produce 1024) o con otra métrica. Subir vectores
    de otra dimensión falla en cada upsert; con otra métrica los puntajes
    no significan lo que esperamos. Se resuelve borrando el índice o usando
    otro ``INDEX_NAME``.
    """


class IndiceInexistenteError(RAGHibridoError):
    """No hay ningún índice con el ``INDEX_NAME`` configurado.

    Se distingue de una caída de Pinecone para que el mensaje diga qué
    hacer: correr el setup o revisar el nombre.
    """


class IndiceNoListoError(RAGHibridoError):
    """El índice existe pero Pinecone todavía no lo marca como listo."""


class NamespaceVacioError(RAGHibridoError):
    """El namespace no tiene vectores: hay que correr la ingesta primero.

    Sin documentos, BM25 no tiene corpus y la búsqueda vectorial no tiene
    dónde buscar: cualquier consulta devolvería una lista vacía y ocultaría
    el problema real.
    """


class IngestaIncompletaError(RAGHibridoError):
    """Algunos lotes del upsert no se cargaron aunque el SDK agotó sus reintentos.

    Un upsert por lotes no lanza excepción si falla un lote: lo informa en la
    respuesta. Sin este error, la falla pasaría inadvertida y la base
    quedaría con huecos.
    """


class PineconeNoDisponibleError(RAGHibridoError):
    """Falló una llamada a Pinecone después de los reintentos del SDK.

    Envuelve la excepción original de Pinecone, que queda como causa
    (``__cause__``).
    """
