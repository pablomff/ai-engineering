"""Cliente LLM para Anthropic.

Implementa ``BaseLLMClient`` con el SDK oficial asíncrono (``AsyncAnthropic``)
y la API de Messages. Sigue la misma estructura que ``OpenAIClient``, pero
maneja las diferencias de Anthropic:

- El mensaje ``system`` no va dentro de ``messages``, sino en el parámetro
  ``system`` aparte.
- ``max_tokens`` es obligatorio.
- No se envía ``temperature``: los modelos actuales de Anthropic la rechazan.
- La respuesta es una lista de bloques que pueden no ser de texto.

Los reintentos y el timeout los maneja el propio SDK (``max_retries`` y
``timeout``): acá no hay reintentos manuales.
"""

import logging
import time
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import AssistantMessageParam, MessageParam, UserMessageParam

from llm_client.base import BaseLLMClient
from llm_client.exceptions import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnknownError,
)
from llm_client.schemas import ChatRequest, ModelConfig, ModelResponse, TokenUsage
from llm_client.settings import Settings

logger = logging.getLogger(__name__)

# Valor por defecto de temperature en ModelConfig. Se usa para avisar solo
# cuando alguien la cambió a propósito.
_DEFAULT_TEMPERATURE: float = ModelConfig().temperature


class AnthropicClient(BaseLLMClient):
    """Implementación de ``BaseLLMClient`` para Anthropic.

    Cumple el contrato de la clase base: ``generate`` y ``stream`` nunca dejan
    escapar excepciones del SDK, siempre lanzan una subclase de ``LLMError``.
    """

    def __init__(self, settings: Settings) -> None:
        """Crea el cliente asíncrono del SDK a partir de la configuración.

        Raises:
            ValueError: si falta ``ANTHROPIC_API_KEY``.
        """
        super().__init__(settings)
        if settings.anthropic_api_key is None:
            # Settings ya lo valida si el proveedor es anthropic, pero se puede
            # construir este cliente directamente con otra configuración.
            raise ValueError("Falta ANTHROPIC_API_KEY para usar el proveedor anthropic")
        self._client: AsyncAnthropic = AsyncAnthropic(
            # El valor real de la key solo se extrae acá, al crear el cliente.
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def provider_name(self) -> str:
        """Nombre del proveedor: ``"anthropic"``."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Modelo configurado en ``ANTHROPIC_MODEL`` (por defecto ``claude-opus-5-5``)."""
        return self.settings.anthropic_model

    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Genera la respuesta completa con la API de Messages.

        La latencia se mide con ``time.perf_counter`` e incluye los
        reintentos que haya hecho el SDK.

        Raises:
            LLMError: alguna subclase, según el error del SDK.
        """
        model = self._resolve_model(request)
        system, messages = self._split_system(request)
        self._warn_if_temperature(request)
        logger.debug("Anthropic generate: model=%s, mensajes=%d", model, len(request.messages))
        inicio = time.perf_counter()
        try:
            message = await self._client.messages.create(
                model=model,
                # A diferencia de OpenAI, max_tokens es obligatorio en Anthropic.
                max_tokens=request.config.max_tokens,
                messages=messages,
                # El system va aparte; omit evita mandar el campo si no hay.
                system=system if system is not None else anthropic.omit,
                # temperature NO se envía: los modelos actuales la rechazan con 400.
            )
        except anthropic.AnthropicError as exc:
            raise self._map_error(exc) from exc
        latency_ms = (time.perf_counter() - inicio) * 1000

        # La respuesta puede traer bloques que no son de texto (por ejemplo,
        # de razonamiento): solo se concatenan los de tipo "text".
        content = "".join(block.text for block in message.content if block.type == "text")
        logger.info(
            "Anthropic generate ok: model=%s, latency_ms=%.0f, finish_reason=%s",
            message.model,
            latency_ms,
            message.stop_reason,
        )
        return ModelResponse(
            ok=True,
            provider=self.provider_name,
            model=message.model,
            content=content,
            # En Anthropic el motivo de fin se llama stop_reason.
            finish_reason=message.stop_reason,
            usage=TokenUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            latency_ms=latency_ms,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Genera la respuesta en fragmentos de texto a medida que llegan.

        Usa el helper ``messages.stream`` del SDK, cuyo ``text_stream`` ya
        filtra los eventos y entrega solo los fragmentos de texto.

        Raises:
            LLMError: alguna subclase, según el error del SDK. Puede ocurrir
                al iniciar o en medio del stream.
        """
        model = self._resolve_model(request)
        system, messages = self._split_system(request)
        self._warn_if_temperature(request)
        logger.debug("Anthropic stream: model=%s, mensajes=%d", model, len(request.messages))
        try:
            # async with cierra la conexión aunque quien consume corte antes.
            async with self._client.messages.stream(
                model=model,
                max_tokens=request.config.max_tokens,
                messages=messages,
                system=system if system is not None else anthropic.omit,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as exc:
            raise self._map_error(exc) from exc

    async def aclose(self) -> None:
        """Cierra el cliente del SDK y libera sus conexiones HTTP."""
        await self._client.close()

    def _split_system(self, request: ChatRequest) -> tuple[str | None, list[MessageParam]]:
        """Separa el mensaje ``system`` del resto de la conversación.

        Anthropic no acepta el system dentro de ``messages``: va en el
        parámetro ``system`` aparte. ``ChatRequest`` ya garantiza que hay como
        máximo uno y que es el primero.

        Se construye un tipo del SDK por rol en vez de un dict genérico para
        que el type checker valide la estructura sin recurrir a ``Any``.
        """
        system: str | None = None
        messages: list[MessageParam] = []
        for m in request.messages:
            if m.role == "system":
                system = m.content
            elif m.role == "user":
                messages.append(UserMessageParam(role="user", content=m.content))
            else:
                messages.append(AssistantMessageParam(role="assistant", content=m.content))
        return system, messages

    def _warn_if_temperature(self, request: ChatRequest) -> None:
        """Avisa si se pidió una temperature que Anthropic va a ignorar.

        Solo avisa si difiere del valor por defecto, para no llenar el log en
        cada llamada normal.
        """
        if request.config.temperature != _DEFAULT_TEMPERATURE:
            logger.warning(
                "Anthropic: se ignora temperature=%s porque los modelos actuales no la aceptan",
                request.config.temperature,
            )

    def _map_error(self, exc: anthropic.AnthropicError) -> LLMError:
        """Traduce una excepción del SDK de Anthropic a nuestra jerarquía.

        El orden importa: ``APITimeoutError`` es subclase de
        ``APIConnectionError``, y las específicas por status (auth, rate
        limit, bad request, overloaded) son subclases de ``APIStatusError``,
        así que van antes que la genérica.

        Los mensajes son propios y no copian el del SDK, para no filtrar datos
        sensibles.
        """
        provider = self.provider_name
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return LLMAuthError(
                f"Anthropic rechazó las credenciales (HTTP {exc.status_code})", provider=provider
            )
        if isinstance(exc, anthropic.RateLimitError):
            return LLMRateLimitError(
                "Anthropic: límite de requests o tokens superado", provider=provider
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMTimeoutError(
                f"Anthropic no respondió en {self.settings.timeout_seconds}s", provider=provider
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMConnectionError("No se pudo conectar con Anthropic", provider=provider)
        if isinstance(exc, anthropic.OverloadedError):
            # 529 (overloaded_error) es propio de Anthropic: el servicio está
            # sobrecargado. Se resuelve esperando, por eso es reintentable.
            return LLMProviderUnavailableError(
                "Anthropic está sobrecargado (HTTP 529)", provider=provider
            )
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                return LLMProviderUnavailableError(
                    f"Anthropic no está disponible (HTTP {exc.status_code})", provider=provider
                )
            if exc.status_code >= 400:
                return LLMBadRequestError(
                    f"Anthropic rechazó el request (HTTP {exc.status_code})", provider=provider
                )
        return LLMUnknownError(
            f"Error inesperado de Anthropic ({type(exc).__name__})", provider=provider
        )
