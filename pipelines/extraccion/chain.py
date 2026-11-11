"""Cadena LCEL que extrae entidades técnicas de un texto y las valida.

Flujo::

    {"texto": ...}
      -> ChatPromptTemplate                     (system + human, con variables)
      -> modelo.with_structured_output(Schema)  (el LLM devuelve el esquema)
      -> validar_salida                         (stop_reason, JSON, Pydantic)
      => ExtraccionTecnica
    toda la cadena envuelta en .with_retry()

Hay dos niveles de reintento, que no se pisan entre sí:

- **Errores de red o del proveedor** (timeout, 429, 5xx): los reintenta el
  SDK, configurado con ``LLM_MAX_RETRIES`` y ``LLM_TIMEOUT_SECONDS``, igual
  que en ``llm_client``.
- **Salida inválida** (JSON mal formado, incompleto o que no cumple el
  esquema): la reintenta ``with_retry``, que vuelve a pedirle la respuesta al
  modelo. Solo reacciona a ``ExtraccionInvalidaError``, así no multiplica los
  reintentos de red que ya hizo el SDK. Los rechazos que se repetirían igual
  en cada intento (texto sin tecnologías, negativa del modelo) usan otras
  excepciones y no se reintentan.

El modelo es un ``ChatAnthropic`` creado en ``llm_client/providers/`` con la
configuración de ``llm_client.Settings``; este módulo no importa nada del SDK.
"""

import functools
import logging
import time
from collections.abc import Mapping

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from llm_client import LLMError, Settings
from llm_client.providers.langchain_anthropic import (
    crear_chat_anthropic,
    traducir_error_anthropic,
)
from pipelines.extraccion.schemas import ERROR_SIN_TECNOLOGIAS, ExtraccionTecnica

logger = logging.getLogger(__name__)

MAX_INTENTOS = 3
"""Intentos totales ante una salida inválida (el primero más 2 reintentos)."""

ESPERA_INICIAL_S = 1.0
"""Espera antes del primer reintento, en segundos. Crece de forma exponencial
(con jitter) en los siguientes."""

MAX_TOKENS = 1024
"""Límite de tokens de salida. Sobra para el esquema; si se alcanza, la
respuesta quedó cortada y se trata como incompleta."""

# Valores de stop_reason (el finish_reason de Anthropic) que se tratan aparte.
_MOTIVO_TRUNCADO = "max_tokens"  # cortó por el límite de tokens
_MOTIVO_NEGATIVA = "refusal"  # el modelo se negó a responder

INSTRUCCIONES_FORMATO = """\
Reglas para completar cada campo:
- tecnologias: solo las que aparecen en el texto, con su nombre propio. No inventes ninguna. \
Si el texto no menciona ninguna tecnología, devolvé una lista vacía; no uses textos como \
"No identificada".
- nivel_de_criticidad:
  - "alta": caída del servicio, pérdida de datos, problemas de seguridad o cuellos de botella \
que afectan a usuarios en producción.
  - "media": degradación de rendimiento o errores intermitentes con workaround.
  - "baja": descripciones de arquitectura sin problemas, mejoras o tareas de mantenimiento.
- resumen_tecnico: una o dos oraciones en español, sin repetir el texto original."""


class ExtraccionError(Exception):
    """Error base del pipeline de extracción.

    Permite que quien llama a ``process_text`` maneje cualquier fallo con un
    solo ``except``, sin conocer las excepciones de LangChain ni de los SDKs.
    """


class ExtraccionConfigError(ExtraccionError):
    """La configuración del entorno no permite crear la cadena (por ejemplo,
    falta ``ANTHROPIC_API_KEY``)."""


class ExtraccionInvalidaError(ExtraccionError):
    """El modelo respondió, pero la salida no sirve y vale la pena reintentar.

    Cubre JSON mal formado, respuestas cortadas por el límite de tokens y
    objetos que no cumplen el esquema. Es la única excepción que dispara
    ``with_retry``.
    """


class TextoSinTecnologiasError(ExtraccionError):
    """El modelo indicó que el texto no menciona ninguna tecnología.

    No se reintenta: con el mismo texto, el modelo daría la misma respuesta.
    """


class ModeloNegativaError(ExtraccionError):
    """El modelo se negó a procesar el texto (``stop_reason == "refusal"``).

    No se reintenta: una negativa ante el mismo texto se repite.
    """


class ExtraccionProveedorError(ExtraccionError):
    """Falló la llamada a Anthropic después de los reintentos del SDK.

    Conserva la ``LLMError`` de ``llm_client`` en ``error``, con su ``kind`` y
    si es reintentable más tarde.
    """

    def __init__(self, error: LLMError) -> None:
        """Guarda la ``LLMError`` traducida y usa su mensaje (que no tiene datos sensibles)."""
        super().__init__(error.message)
        self.error: LLMError = error


def crear_prompt() -> ChatPromptTemplate:
    """Crea el prompt de extracción con roles System y Human.

    El template recibe dos variables: ``texto`` (la entrada del usuario) e
    ``instrucciones_formato``. Esta última se fija con ``partial`` para que
    quien invoca la cadena solo tenga que pasar ``{"texto": ...}``, pero se
    puede reemplazar sin tocar el resto del prompt.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sos un analista técnico de software. Tu tarea es extraer información "
                "técnica estructurada del texto que te pasa el usuario.\n\n"
                "{instrucciones_formato}",
            ),
            ("human", "Texto a analizar:\n\n{texto}"),
        ]
    )
    return prompt.partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


def crear_modelo() -> BaseChatModel:
    """Crea el ``ChatAnthropic`` con la configuración del entorno.

    Fuerza el proveedor ``anthropic``: el pipeline solo usa Anthropic, así que
    no depende de ``LLM_PROVIDER`` ni exige la key de otro proveedor.

    Raises:
        ExtraccionConfigError: si la configuración es inválida (por ejemplo,
            falta ``ANTHROPIC_API_KEY``).
    """
    try:
        settings = Settings.from_env(provider="anthropic")
        return crear_chat_anthropic(settings, max_tokens=MAX_TOKENS)
    except ValidationError as exc:
        # Solo los mensajes, sin los valores recibidos, para no exponer la key.
        mensajes = "; ".join(e["msg"] for e in exc.errors(include_input=False))
        raise ExtraccionConfigError(f"Configuración inválida: {mensajes}") from exc
    except ValueError as exc:
        raise ExtraccionConfigError(f"Configuración inválida: {exc}") from exc


def _tipos_de_error(error: ValidationError) -> set[str]:
    """Devuelve los tipos de error de una ``ValidationError`` (ej. ``"missing"``)."""
    return {e["type"] for e in error.errors()}


def validar_salida(salida: Mapping[str, object]) -> ExtraccionTecnica:
    """Verifica la salida de ``with_structured_output(..., include_raw=True)``.

    Con ``include_raw=True`` el modelo estructurado no lanza excepciones de
    parseo: devuelve ``{"raw": AIMessage, "parsed": ..., "parsing_error": ...}``.
    Eso permite revisar el ``stop_reason`` (el ``finish_reason`` de Anthropic)
    antes de confiar en el objeto, y decidir acá qué cuenta como salida
    inválida.

    Por qué se revisa el ``stop_reason`` primero: una respuesta cortada por el
    límite de tokens puede parsear igual (por ejemplo, con un resumen a medio
    escribir), y en ese caso Pydantic no detectaría el problema.

    Raises:
        ModeloNegativaError: si el modelo se negó a responder.
        TextoSinTecnologiasError: si el modelo devolvió una lista de
            tecnologías vacía y ese es el único problema.
        ExtraccionInvalidaError: si la respuesta está truncada, el JSON no se
            pudo parsear o no cumple el esquema por otro motivo.
    """
    raw = salida.get("raw")
    metadata = raw.response_metadata if isinstance(raw, BaseMessage) else {}
    motivo_fin = metadata.get("stop_reason")

    if motivo_fin == _MOTIVO_NEGATIVA:
        logger.warning("El modelo se negó a procesar el texto (stop_reason=refusal)")
        raise ModeloNegativaError("El modelo se negó a procesar el texto")
    if motivo_fin == _MOTIVO_TRUNCADO:
        logger.warning("Salida inválida: respuesta cortada por límite de tokens (%s)", motivo_fin)
        raise ExtraccionInvalidaError("La respuesta del modelo quedó incompleta")

    error = salida.get("parsing_error")
    # El parser de LangChain envuelve la ValidationError de Pydantic en una
    # OutputParserException; se desenvuelve para distinguir "no cumple el
    # esquema" de "no es un JSON válido".
    causa = error.__cause__ if isinstance(error, OutputParserException) else error
    if isinstance(causa, ValidationError):
        if _tipos_de_error(causa) == {ERROR_SIN_TECNOLOGIAS}:
            logger.warning("El texto no menciona tecnologías: no se reintenta")
            raise TextoSinTecnologiasError("El texto no menciona ninguna tecnología") from error
        # Se loguean solo los campos que fallaron, no los valores: pueden
        # contener fragmentos del texto del usuario.
        campos = sorted({".".join(str(p) for p in e["loc"]) or "<raíz>" for e in causa.errors()})
        logger.warning("Salida inválida: no cumple el esquema (campos: %s)", ", ".join(campos))
        raise ExtraccionInvalidaError("La respuesta no cumple el esquema") from error
    if isinstance(error, BaseException):
        logger.warning("Salida inválida: JSON mal formado (%s)", type(error).__name__)
        raise ExtraccionInvalidaError("La respuesta no es un JSON válido") from error

    parsed = salida.get("parsed")
    if not isinstance(parsed, ExtraccionTecnica):
        logger.warning("Salida inválida: el modelo no devolvió la estructura pedida")
        raise ExtraccionInvalidaError("El modelo no devolvió la estructura pedida")

    logger.info(
        "Salida validada: %d tecnologías, criticidad=%s, stop_reason=%s",
        len(parsed.tecnologias),
        parsed.nivel_de_criticidad,
        motivo_fin,
    )
    return parsed


def crear_cadena(modelo: BaseChatModel) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Arma la cadena LCEL completa, con reintentos ante salidas inválidas.

    Recibe el modelo ya creado (normalmente con ``crear_modelo``) para poder
    probar la cadena con un modelo falso. La cadena recibe ``{"texto": ...}``
    y devuelve un ``ExtraccionTecnica`` ya validado.
    """
    prompt = crear_prompt()

    cadena = (
        prompt
        # json_schema usa la salida estructurada nativa de la API. El método por
        # defecto de ChatAnthropic (forzar una tool call) no lo soportan los
        # modelos Claude actuales.
        | modelo.with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)
        | RunnableLambda(validar_salida)
    )
    # El reintento envuelve toda la cadena: cada intento vuelve a llamar al
    # modelo, que es lo que puede corregir un JSON mal formado. Con
    # include_raw=True los errores de parseo no se lanzan, llegan a
    # validar_salida, así que la única excepción a reintentar es la nuestra.
    return cadena.with_retry(
        retry_if_exception_type=(ExtraccionInvalidaError,),
        stop_after_attempt=MAX_INTENTOS,
        # LangChain solo acepta la clave "initial" (descarta "multiplier").
        exponential_jitter_params={"initial": ESPERA_INICIAL_S, "jitter": ESPERA_INICIAL_S},
    )


@functools.cache
def _cadena_por_defecto() -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Crea la cadena del entorno una sola vez y la reutiliza.

    Así todas las llamadas a ``process_text`` comparten el mismo cliente HTTP
    y su pool de conexiones, en vez de abrir uno nuevo por llamada. Si la
    configuración es inválida no se cachea nada: la excepción se repite.
    """
    return crear_cadena(crear_modelo())


async def process_text(
    text: str,
    cadena: Runnable[dict[str, str], ExtraccionTecnica] | None = None,
) -> ExtraccionTecnica:
    """Extrae las entidades técnicas de ``text`` de forma asíncrona.

    Args:
        text: texto sin procesar (descripción de arquitectura, log de error, etc.).
        cadena: cadena a usar. Si es ``None`` se usa la del entorno, creada una
            sola vez; recibirla como parámetro permite reemplazarla en tests.

    Raises:
        ValueError: si ``text`` está vacío.
        ExtraccionConfigError: si la configuración del entorno es inválida.
        TextoSinTecnologiasError: si el texto no menciona tecnologías.
        ModeloNegativaError: si el modelo se negó a procesar el texto.
        ExtraccionInvalidaError: si después de ``MAX_INTENTOS`` intentos el
            modelo no devolvió una salida válida.
        ExtraccionProveedorError: si falló la llamada a Anthropic (red,
            autenticación, etc.) después de los reintentos del SDK.

    Cualquier otra excepción es un bug y se propaga sin envolver.
    """
    if not text.strip():
        raise ValueError("El texto a procesar no puede estar vacío")
    if cadena is None:
        cadena = _cadena_por_defecto()

    logger.info("Procesando texto de %d caracteres", len(text))
    inicio = time.perf_counter()
    try:
        resultado = await cadena.ainvoke({"texto": text})
    except ExtraccionInvalidaError:
        logger.error("Sin salida válida después de %d intentos", MAX_INTENTOS)
        raise
    except ExtraccionError:
        raise
    except Exception as exc:
        error = traducir_error_anthropic(exc)
        if error is None:
            raise
        # Solo el tipo y el kind: el mensaje original podría tener datos del request.
        logger.error("Falló la llamada a Anthropic (%s, kind=%s)", type(exc).__name__, error.kind)
        raise ExtraccionProveedorError(error) from exc

    logger.info("Extracción completada en %.0f ms", (time.perf_counter() - inicio) * 1000)
    return resultado
