"""Cadena RAG: recupera fragmentos de ChromaDB y genera una respuesta basada en ellos.

Flujo::

    "pregunta"
      -> RunnableParallel           documentos = retriever(pregunta)  (búsqueda por similitud)
                                    pregunta   = la misma pregunta
      -> .assign(contexto=...)      los fragmentos formateados con su archivo y número
      -> .assign(respuesta=...)     prompt -> ChatAnthropic -> verificar_fin
                                    -> PydanticOutputParser
      -> validar_referencias        las referencias tienen que ser fragmentos recuperados
      => RespuestaRAG

Por qué la cadena conserva los documentos hasta el final, en vez de pasarle al
prompt solo el texto: así ``validar_referencias`` puede comprobar que cada
fuente citada por el modelo esté entre los fragmentos que de verdad recibió.
Una referencia a un fragmento que no estaba en el contexto es una fuente
inventada.

Los errores de red o del proveedor los reintenta el SDK (``LLM_MAX_RETRIES``).
No hay reintentos ante una salida inválida: se informa con
``RespuestaInvalidaError``.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from operator import itemgetter

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import ValidationError

from llm_client.providers.langchain_anthropic import (
    MOTIVO_NEGATIVA,
    MOTIVO_TRUNCADO,
    crear_chat_anthropic_desde_entorno,
    traducir_error_anthropic,
)
from pipelines.rag.exceptions import (
    ModeloNegativaError,
    RAGConfigError,
    RAGError,
    RAGProveedorError,
    RecursosNoDisponiblesError,
    RespuestaInvalidaError,
)
from pipelines.rag.schemas import RespuestaRAG
from pipelines.rag.vectorstore import (
    DIRECTORIO_POR_DEFECTO,
    MODELO_EMBEDDINGS,
    abrir_cliente_existente,
    abrir_para_consulta,
    crear_embeddings,
)

logger = logging.getLogger(__name__)

TOP_K = 4
"""Cantidad de fragmentos que se recuperan por pregunta.

Pocos fragmentos pueden dejar afuera la respuesta; muchos llenan el prompt
de texto irrelevante y el modelo presta menos atención a lo importante (el
problema conocido como *lost in the middle*). Entre 3 y 5 es el rango
habitual.
"""

MAX_TOKENS = 1024
"""Límite de tokens de la respuesta. Sobra para una respuesta breve con sus
referencias; si se alcanza, la respuesta quedó cortada."""

SISTEMA = """\
Sos el asistente técnico interno del equipo de ingeniería. Respondés preguntas \
usando únicamente el CONTEXTO que te pasa el usuario: fragmentos de la \
documentación interna, cada uno con su archivo y su número de fragmento.

Reglas:
- Respondé solo con información que esté en el CONTEXTO. No uses conocimiento \
propio ni completes con suposiciones, aunque la respuesta te parezca obvia.
- Si el CONTEXTO no contiene la respuesta, respondé exactamente "No lo sé." y \
dejá las referencias vacías.
- En referencias, citá cada fragmento que usaste, con el archivo y el número de \
fragmento tal como aparecen en el CONTEXTO.
- Respondé en español, en forma breve y concreta.

{instrucciones_formato}"""

HUMANO = "CONTEXTO:\n\n{contexto}\n\nPREGUNTA: {pregunta}"


def crear_prompt(parser: PydanticOutputParser[RespuestaRAG]) -> ChatPromptTemplate:
    """Crea el prompt "filtro de veracidad" con roles System y Human.

    Recibe ``contexto`` y ``pregunta``. Las instrucciones de formato (el
    esquema JSON de ``RespuestaRAG``) las genera el parser y se fijan con
    ``partial``, así el prompt y el parser nunca quedan desincronizados.
    """
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    return prompt.partial(instrucciones_formato=parser.get_format_instructions())


def formatear_contexto(documentos: Sequence[Document]) -> str:
    """Une los fragmentos recuperados en un texto, cada uno con su encabezado.

    El encabezado (``[archivo: ... | fragmento: N]``) es lo que el modelo
    copia en las referencias, por eso usa los mismos nombres que
    ``Referencia``.
    """
    return "\n\n---\n\n".join(
        f"[archivo: {d.metadata['archivo']} | fragmento: {d.metadata['fragmento']}]\n"
        f"{d.page_content}"
        for d in documentos
    )


def verificar_fin(mensaje: BaseMessage) -> BaseMessage:
    """Revisa por qué terminó de generar el modelo antes de parsear la respuesta.

    Una respuesta cortada por el límite de tokens puede parecer válida si el
    corte cae en un lugar justo, así que se descarta antes de mirarla.

    Raises:
        ModeloNegativaError: si el modelo se negó a responder.
        RespuestaInvalidaError: si la respuesta quedó cortada.
    """
    motivo_fin = mensaje.response_metadata.get("stop_reason")
    if motivo_fin == MOTIVO_NEGATIVA:
        logger.warning("El modelo se negó a responder (stop_reason=refusal)")
        raise ModeloNegativaError("El modelo se negó a responder")
    if motivo_fin == MOTIVO_TRUNCADO:
        logger.warning("Respuesta cortada por el límite de tokens")
        raise RespuestaInvalidaError("La respuesta del modelo quedó incompleta")
    return mensaje


def validar_referencias(entrada: Mapping[str, object]) -> RespuestaRAG:
    """Comprueba que el modelo solo haya citado fragmentos que recibió.

    Recibe el diccionario que arma la cadena (``documentos``, ``pregunta``,
    ``contexto`` y ``respuesta``) y devuelve la ``RespuestaRAG``.

    Raises:
        RespuestaInvalidaError: si alguna referencia no corresponde a un
            fragmento recuperado.
    """
    documentos = entrada["documentos"]
    respuesta = entrada["respuesta"]
    if not isinstance(respuesta, RespuestaRAG) or not isinstance(documentos, list):
        raise TypeError("validar_referencias recibió una entrada inesperada")

    recuperados = {(d.metadata["archivo"], d.metadata["fragmento"]) for d in documentos}
    inventadas = [r for r in respuesta.referencias if (r.archivo, r.fragmento) not in recuperados]
    if inventadas:
        # Se loguean nombres de archivo y números, nunca el texto.
        logger.warning(
            "Referencias a fragmentos no recuperados: %s",
            ", ".join(f"{r.archivo}#{r.fragmento}" for r in inventadas),
        )
        raise RespuestaInvalidaError("El modelo citó fragmentos que no estaban en el contexto")

    logger.info(
        "Respuesta validada: encontrada=%s, referencias=[%s]",
        respuesta.encontrada,
        ", ".join(f"{r.archivo}#{r.fragmento}" for r in respuesta.referencias),
    )
    return respuesta


def _registrar_recuperados(documentos: list[Document]) -> list[Document]:
    """Loguea qué fragmentos devolvió la búsqueda y los deja pasar sin cambios."""
    logger.info(
        "Fragmentos recuperados: %s",
        ", ".join(f"{d.metadata['archivo']}#{d.metadata['fragmento']}" for d in documentos),
    )
    return documentos


def crear_retriever(vectorstore: Chroma, k: int = TOP_K) -> BaseRetriever:
    """Crea el retriever: convierte la pregunta en un embedding y busca los ``k`` más cercanos.

    Usa el mismo objeto de embeddings con el que se abrió el vectorstore, que
    es el mismo modelo con el que se indexó.
    """
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def crear_cadena(
    retriever: Runnable[str, list[Document]], modelo: BaseChatModel
) -> Runnable[str, RespuestaRAG]:
    """Arma la cadena LCEL completa: recuperación, generación y validación.

    Recibe el retriever y el modelo ya creados para poder probar la cadena
    con versiones falsas. La cadena recibe la pregunta como ``str``.
    """
    parser = PydanticOutputParser(pydantic_object=RespuestaRAG)
    generar = crear_prompt(parser) | modelo | RunnableLambda(verificar_fin) | parser

    return (
        RunnableParallel(
            documentos=retriever | RunnableLambda(_registrar_recuperados),
            pregunta=RunnablePassthrough(),
        )
        | RunnablePassthrough.assign(
            contexto=itemgetter("documentos") | RunnableLambda(formatear_contexto)
        )
        | RunnablePassthrough.assign(respuesta=generar)
        | RunnableLambda(validar_referencias)
    )


def crear_modelo() -> BaseChatModel:
    """Crea el ``ChatAnthropic`` con la configuración del entorno.

    Raises:
        RAGConfigError: si falta ``ANTHROPIC_API_KEY`` u otra configuración
            es inválida.
    """
    try:
        return crear_chat_anthropic_desde_entorno(max_tokens=MAX_TOKENS)
    except ValueError as exc:
        # El mensaje ya viene sin valores sensibles (ver providers/).
        raise RAGConfigError(str(exc)) from exc


def _crear_cadena_por_defecto() -> Runnable[str, RespuestaRAG]:
    """Arma la cadena del entorno: modelo, base de ``DIRECTORIO_POR_DEFECTO`` y retriever.

    Raises:
        RAGConfigError: si falta configuración o la base no sirve.
        RecursosNoDisponiblesError: si no se pudo cargar el modelo de
            embeddings o abrir la base.
    """
    modelo = crear_modelo()
    try:
        cliente = abrir_cliente_existente(DIRECTORIO_POR_DEFECTO)
        vectorstore = abrir_para_consulta(cliente, crear_embeddings(), MODELO_EMBEDDINGS)
    except (OSError, ChromaError) as exc:
        # OSError cubre las fallas de HuggingFace (modelo sin descargar y sin
        # conexión); ChromaError, una base persistida dañada o ilegible.
        logger.error("No se pudo preparar la búsqueda (%s)", type(exc).__name__)
        raise RecursosNoDisponiblesError(
            f"No se pudo cargar el modelo de embeddings o abrir la base ({type(exc).__name__})"
        ) from exc
    return crear_cadena(crear_retriever(vectorstore), modelo)


_cadena: Runnable[str, RespuestaRAG] | None = None
"""La cadena por defecto, una vez creada."""

_lock_cadena = threading.Lock()


def cadena_por_defecto() -> Runnable[str, RespuestaRAG]:
    """Devuelve la cadena con la base de ``vectorstore/`` y el modelo del ``.env``.

    Se crea una sola vez y se reutiliza: cargar el modelo de embeddings tarda
    unos segundos y así todas las consultas comparten el cliente HTTP. La
    primera vez es bloqueante (carga el modelo, abre la base), por eso desde
    código async se llama con ``asyncio.to_thread``. El lock evita que dos
    consultas simultáneas carguen el modelo dos veces. Si la creación falla
    no se guarda nada, así que el próximo intento vuelve a probar.

    Raises:
        RAGConfigError: si falta configuración, la base está vacía, se
            indexó con otro modelo de embeddings o no se pudo cargar.
    """
    global _cadena
    if _cadena is None:
        with _lock_cadena:
            if _cadena is None:
                _cadena = _crear_cadena_por_defecto()
    return _cadena


async def get_rag_response(
    query: str,
    cadena: Runnable[str, RespuestaRAG] | None = None,
) -> RespuestaRAG:
    """Responde ``query`` usando solo los documentos indexados.

    Args:
        query: la pregunta del usuario.
        cadena: cadena a usar. Si es ``None`` se usa ``cadena_por_defecto()``;
            recibirla como parámetro permite reemplazarla en tests.

    Returns:
        La respuesta con sus referencias. Si los documentos no tienen la
        respuesta, ``respuesta`` es "No lo sé." y ``encontrada`` es ``False``:
        eso es un resultado válido, no un error.

    Raises:
        ValueError: si ``query`` está vacía.
        RAGConfigError: si falta configuración, la base no sirve o no se
            pudo cargar el modelo de embeddings (``RecursosNoDisponiblesError``).
        ModeloNegativaError: si el modelo se negó a responder.
        RespuestaInvalidaError: si la respuesta no tiene el formato pedido, quedó
            cortada, no tiene referencias o cita fragmentos que no recibió.
        RAGProveedorError: si falló la llamada a Anthropic después de los
            reintentos del SDK.

    Cualquier otra excepción es un bug y se propaga sin envolver.
    """
    if not query.strip():
        raise ValueError("La pregunta no puede estar vacía")
    if cadena is None:
        # Si ya está creada se usa directo; solo la primera vez hace falta un
        # hilo, porque crearla bloquea.
        cadena = _cadena or await asyncio.to_thread(cadena_por_defecto)

    logger.info("Consulta de %d caracteres", len(query))
    inicio = time.perf_counter()
    try:
        respuesta = await cadena.ainvoke(query)
    except RAGError:
        raise
    except OutputParserException as exc:
        causa = exc.__cause__
        if isinstance(causa, ValidationError):
            # Solo los mensajes de Pydantic, sin los valores recibidos.
            detalle = "; ".join(e["msg"] for e in causa.errors(include_input=False))
            logger.warning("Salida inválida: no cumple el esquema (%s)", detalle)
        else:
            logger.warning("Salida inválida: JSON mal formado")
        raise RespuestaInvalidaError("La respuesta del modelo no cumple el formato pedido") from exc
    except Exception as exc:
        error = traducir_error_anthropic(exc)
        if error is None:
            raise
        logger.error("Falló la llamada a Anthropic (%s, kind=%s)", type(exc).__name__, error.kind)
        raise RAGProveedorError(error) from exc

    logger.info("Respuesta generada en %.0f ms", (time.perf_counter() - inicio) * 1000)
    return respuesta
