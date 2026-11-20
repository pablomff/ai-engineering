"""Chat model de LangChain para Anthropic, construido desde ``Settings``.

``langchain_anthropic`` envuelve el SDK ``anthropic`` y deja pasar sus
excepciones. Por eso vive en ``providers/``, el único paquete donde se toca el
SDK: fuera de acá, el código recibe un ``BaseChatModel`` genérico y traduce
los errores con ``traducir_error_anthropic`` a nuestra jerarquía ``LLMError``,
sin importar nada de ``anthropic``.
"""

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from llm_client.exceptions import LLMError
from llm_client.providers.anthropic_client import map_anthropic_error
from llm_client.settings import Settings

# Valores de stop_reason (el finish_reason de Anthropic) que los pipelines
# tratan aparte. Viven acá porque son propios de la API de Anthropic.
MOTIVO_TRUNCADO = "max_tokens"
"""El modelo cortó por el límite de tokens: la respuesta está incompleta."""
MOTIVO_NEGATIVA = "refusal"
"""El modelo se negó a responder."""


def crear_chat_anthropic_desde_entorno(*, max_tokens: int) -> BaseChatModel:
    """Crea un ``ChatAnthropic`` con la configuración de ``os.environ``.

    Fuerza el proveedor ``anthropic``: los pipelines LCEL solo usan
    Anthropic, así que no dependen de ``LLM_PROVIDER`` ni exigen la key de
    otro proveedor. Está acá, y no copiado en cada pipeline, para que la
    limpieza de los mensajes de error (que no deben mostrar la API key) viva
    en un solo lugar.

    Raises:
        ValueError: si la configuración es inválida (por ejemplo, falta
            ``ANTHROPIC_API_KEY``). El mensaje empieza con "Configuración
            inválida:" y nunca incluye los valores recibidos.
    """
    try:
        settings = Settings.from_env(provider="anthropic")
        return crear_chat_anthropic(settings, max_tokens=max_tokens)
    except ValidationError as exc:
        # Solo los mensajes, sin los valores recibidos, para no exponer la key.
        mensajes = "; ".join(e["msg"] for e in exc.errors(include_input=False))
        raise ValueError(f"Configuración inválida: {mensajes}") from exc
    except ValueError as exc:
        raise ValueError(f"Configuración inválida: {exc}") from exc


def crear_chat_anthropic(settings: Settings, *, max_tokens: int) -> BaseChatModel:
    """Crea un ``ChatAnthropic`` con la configuración de ``Settings``.

    Usa la misma API key, modelo, timeout y reintentos nativos del SDK que
    ``AnthropicClient``. No envía ``temperature``, porque los modelos actuales
    de Anthropic la rechazan con un error 400.

    Raises:
        ValueError: si falta ``ANTHROPIC_API_KEY``.
    """
    if settings.anthropic_api_key is None:
        raise ValueError("Falta ANTHROPIC_API_KEY para usar el proveedor anthropic")
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        max_tokens=max_tokens,
    )


def traducir_error_anthropic(exc: BaseException) -> LLMError | None:
    """Traduce una excepción del SDK de Anthropic a ``LLMError``.

    Devuelve ``None`` si la excepción no viene del SDK, para que quien llama
    decida qué hacer con ella (normalmente, dejarla propagar: es un bug).
    Las excepciones propias de ``langchain_anthropic`` (como
    ``AnthropicInvalidRequestError``) heredan de las del SDK, así que también
    se traducen.
    """
    if isinstance(exc, anthropic.AnthropicError):
        return map_anthropic_error(exc)
    return None
