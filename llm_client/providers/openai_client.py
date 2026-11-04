"""Cliente LLM para OpenAI.

Implementa ``BaseLLMClient`` con el SDK oficial asíncrono (``AsyncOpenAI``) y
la API de Chat Completions. Traduce nuestros schemas al formato del SDK y las
excepciones del SDK a nuestras ``LLMError``, para que nada de ``openai`` salga
de este módulo.

Los reintentos y el timeout los maneja el propio SDK (``max_retries`` y
``timeout``): acá no hay reintentos manuales.
"""

import logging
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

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
from llm_client.schemas import ChatMessage, ChatRequest, ModelResponse, TokenUsage
from llm_client.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Implementación de ``BaseLLMClient`` para OpenAI.

    Cumple el contrato de la clase base: ``generate`` y ``stream`` nunca dejan
    escapar excepciones del SDK, siempre lanzan una subclase de ``LLMError``.
    """

    def __init__(self, settings: Settings) -> None:
        """Crea el cliente asíncrono del SDK a partir de la configuración.

        Falla al construir si no hay modelo configurado: es mejor detectarlo
        al arrancar que en la primera llamada.

        Raises:
            ValueError: si falta ``OPENAI_API_KEY`` u ``OPENAI_MODEL``.
        """
        super().__init__(settings)
        if settings.openai_model is None:
            raise ValueError("Falta OPENAI_MODEL para usar el proveedor openai")
        if settings.openai_api_key is None:
            # Settings ya lo valida si el proveedor es openai, pero se puede
            # construir este cliente directamente con otra configuración.
            raise ValueError("Falta OPENAI_API_KEY para usar el proveedor openai")
        self._model: str = settings.openai_model
        self._client: AsyncOpenAI = AsyncOpenAI(
            # El valor real de la key solo se extrae acá, al crear el cliente.
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def provider_name(self) -> str:
        """Nombre del proveedor: ``"openai"``."""
        return "openai"

    @property
    def default_model(self) -> str:
        """Modelo configurado en ``OPENAI_MODEL``."""
        return self._model

    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Genera la respuesta completa con Chat Completions.

        La latencia se mide con ``time.perf_counter`` e incluye los
        reintentos que haya hecho el SDK.

        Raises:
            LLMError: alguna subclase, según el error del SDK.
        """
        model = self._resolve_model(request)
        logger.debug("OpenAI generate: model=%s, mensajes=%d", model, len(request.messages))
        inicio = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(request.messages),
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
            )
        except openai.OpenAIError as exc:
            raise self._map_error(exc) from exc
        latency_ms = (time.perf_counter() - inicio) * 1000

        # Defensivo: la API siempre devuelve al menos una choice, pero el tipo
        # del SDK no lo garantiza.
        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice else None) or ""
        usage = (
            TokenUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            )
            if completion.usage
            else None
        )
        logger.info(
            "OpenAI generate ok: model=%s, latency_ms=%.0f, finish_reason=%s",
            completion.model,
            latency_ms,
            choice.finish_reason if choice else None,
        )
        return ModelResponse(
            ok=True,
            provider=self.provider_name,
            model=completion.model,
            content=content,
            finish_reason=choice.finish_reason if choice else None,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Genera la respuesta en fragmentos de texto a medida que llegan.

        Solo hace ``yield`` de fragmentos no vacíos. Algunos chunks llegan sin
        ``choices`` (por ejemplo, el de uso de tokens) o con ``content`` en
        ``None`` (el primero, que solo trae el rol), y se saltean.

        Raises:
            LLMError: alguna subclase, según el error del SDK. Puede ocurrir
                al iniciar o en medio del stream.
        """
        model = self._resolve_model(request)
        logger.debug("OpenAI stream: model=%s, mensajes=%d", model, len(request.messages))
        try:
            respuesta = await self._client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(request.messages),
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
                stream=True,
            )
            # async with cierra la conexión aunque quien consume corte antes.
            async with respuesta:
                async for chunk in respuesta:
                    if not chunk.choices:
                        continue
                    texto = chunk.choices[0].delta.content
                    if texto:
                        yield texto
        except openai.OpenAIError as exc:
            raise self._map_error(exc) from exc

    async def aclose(self) -> None:
        """Cierra el cliente del SDK y libera sus conexiones HTTP."""
        await self._client.close()

    def _map_error(self, exc: openai.OpenAIError) -> LLMError:
        """Traduce una excepción del SDK de OpenAI a nuestra jerarquía.

        El orden importa: ``APITimeoutError`` es subclase de
        ``APIConnectionError``, y las específicas por status (auth, rate
        limit, bad request) son subclases de ``APIStatusError``, así que van
        antes que la genérica.

        Los mensajes son propios y no copian el del SDK, para no filtrar datos
        sensibles (el SDK puede incluir fragmentos de la API key).
        """
        provider = self.provider_name
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return LLMAuthError(
                f"OpenAI rechazó las credenciales (HTTP {exc.status_code})", provider=provider
            )
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError("OpenAI: límite de requests o tokens superado", provider=provider)
        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(
                f"OpenAI no respondió en {self.settings.timeout_seconds}s", provider=provider
            )
        if isinstance(exc, openai.APIConnectionError):
            return LLMConnectionError("No se pudo conectar con OpenAI", provider=provider)
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                return LLMProviderUnavailableError(
                    f"OpenAI no está disponible (HTTP {exc.status_code})", provider=provider
                )
            if exc.status_code >= 400:
                return LLMBadRequestError(
                    f"OpenAI rechazó el request (HTTP {exc.status_code})", provider=provider
                )
        return LLMUnknownError(f"Error inesperado de OpenAI ({type(exc).__name__})", provider=provider)


def _to_openai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Convierte nuestros ``ChatMessage`` al formato de mensajes del SDK.

    Se construye un tipo del SDK por rol en vez de un dict genérico para que
    el type checker valide la estructura sin recurrir a ``Any``.
    """
    resultado: list[ChatCompletionMessageParam] = []
    for m in messages:
        if m.role == "system":
            resultado.append(ChatCompletionSystemMessageParam(role="system", content=m.content))
        elif m.role == "user":
            resultado.append(ChatCompletionUserMessageParam(role="user", content=m.content))
        else:
            resultado.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=m.content)
            )
    return resultado
