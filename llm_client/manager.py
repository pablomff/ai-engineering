"""Punto de entrada del cliente LLM para la aplicación.

Define ``AsyncLLMManager``, la única clase que la aplicación necesita usar:
elige el proveedor según la configuración, arma y valida los requests, y
garantiza un contrato de errores simple:

- ``generate`` nunca lanza excepciones: siempre devuelve un ``ModelResponse``,
  con ``ok=False`` y un ``ErrorInfo`` si algo falló.
- ``stream`` propaga los errores, pero siempre como ``LLMError``.

Por qué existe: separa a la aplicación de los detalles de cada proveedor y de
cómo se construyen los requests. Cambiar de proveedor es cambiar la
configuración, sin tocar el código que usa el manager.
"""

import logging
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

from pydantic import ValidationError

from llm_client.base import BaseLLMClient
from llm_client.exceptions import LLMBadRequestError, LLMError, LLMUnknownError
from llm_client.providers.anthropic_client import AnthropicClient
from llm_client.providers.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ChatRequest, ErrorInfo, ModelConfig, ModelResponse
from llm_client.settings import Settings

logger = logging.getLogger(__name__)

# Registro de proveedores disponibles. Agregar un proveedor nuevo es agregar
# una entrada acá, sin tocar el resto del manager.
_PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


class AsyncLLMManager:
    """Fachada asíncrona sobre el cliente del proveedor configurado.

    Se recomienda usarla con ``async with`` para que las conexiones HTTP se
    liberen siempre::

        async with AsyncLLMManager.from_env() as llm:
            respuesta = await llm.generate([ChatMessage(role="user", content="Hola")])
    """

    def __init__(self, settings: Settings) -> None:
        """Crea el cliente del proveedor indicado en ``settings.provider``.

        Raises:
            ValueError: si el proveedor no está registrado, o si el cliente
                del proveedor rechaza la configuración (por ejemplo, falta
                ``OPENAI_MODEL``).
        """
        cliente_cls = _PROVIDERS.get(settings.provider)
        if cliente_cls is None:
            validos = ", ".join(sorted(_PROVIDERS))
            raise ValueError(
                f"Proveedor '{settings.provider}' no soportado. Proveedores válidos: {validos}"
            )
        self._client: BaseLLMClient = cliente_cls(settings)
        logger.info("AsyncLLMManager listo con proveedor %s", self._client.provider_name)

    @classmethod
    def from_env(cls) -> Self:
        """Construye el manager con la configuración leída de ``os.environ``.

        No carga el ``.env``: eso lo hace el punto de entrada de la app.
        """
        return cls(Settings.from_env())

    @property
    def provider(self) -> str:
        """Nombre del proveedor activo."""
        return self._client.provider_name

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> ModelResponse:
        """Genera la respuesta completa. Nunca lanza excepciones.

        Por qué no lanza: así quien llama maneja éxito y error con el mismo
        objeto (``respuesta.ok``), sin ``try/except``. Los errores se
        devuelven como ``ModelResponse(ok=False, error=...)``:

        - request inválido (validación de Pydantic) -> ``bad_request``;
        - ``LLMError`` del cliente -> su ``ErrorInfo``;
        - cualquier otra excepción -> se loguea y se devuelve como ``unknown``.
        """
        model_config = config or ModelConfig()
        # Para las respuestas de error: el modelo que se intentó usar.
        model = model_config.model or self._client.default_model
        inicio = time.perf_counter()
        try:
            request = ChatRequest(messages=messages, config=model_config)
            return await self._client.generate(request)
        except ValidationError as exc:
            error = LLMBadRequestError(_describe_validation(exc), provider=self.provider)
            return self._error_response(error.to_error_info(), model, inicio)
        except LLMError as exc:
            logger.warning("Error en generate (%s): %s", exc.kind, exc.message)
            return self._error_response(exc.to_error_info(), model, inicio)
        except Exception as exc:
            # Se captura Exception (no BaseException) para no tragar una
            # cancelación de asyncio.
            logger.exception("Error inesperado en generate")
            error = LLMUnknownError(
                f"Error inesperado ({type(exc).__name__})", provider=self.provider
            )
            return self._error_response(error.to_error_info(), model, inicio)

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        """Reenvía los fragmentos de texto del cliente a medida que llegan.

        A diferencia de ``generate``, acá los errores SÍ se propagan, porque
        en streaming no hay un "valor de retorno" donde poner el error: lo que
        se devuelve son los fragmentos. Además, si el corte llega a mitad de
        la respuesta, quien consume ya mostró parte del texto y tiene que
        enterarse de que quedó incompleto.

        Los errores siempre se propagan como ``LLMError``, nunca como
        excepciones de un SDK ni excepciones genéricas sin envolver.

        Raises:
            LLMBadRequestError: si el request es inválido.
            LLMError: alguna subclase, ante cualquier error del proveedor o
                inesperado.
        """
        try:
            request = ChatRequest(messages=messages, config=config or ModelConfig())
        except ValidationError as exc:
            raise LLMBadRequestError(_describe_validation(exc), provider=self.provider) from exc

        try:
            async for fragmento in self._client.stream(request):
                yield fragmento
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Error inesperado en stream")
            raise LLMUnknownError(
                f"Error inesperado ({type(exc).__name__})", provider=self.provider
            ) from exc

    async def aclose(self) -> None:
        """Libera las conexiones HTTP del cliente."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Permite usar el manager con ``async with``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cierra el cliente al salir del ``async with``, sin suprimir excepciones."""
        await self.aclose()

    def _error_response(self, error: ErrorInfo, model: str, inicio: float) -> ModelResponse:
        """Arma un ``ModelResponse`` fallido con la latencia hasta el error."""
        return ModelResponse(
            ok=False,
            provider=self.provider,
            model=model,
            latency_ms=(time.perf_counter() - inicio) * 1000,
            error=error,
        )


def _describe_validation(exc: ValidationError) -> str:
    """Resume un ``ValidationError`` sin incluir los valores recibidos.

    El ``str()`` de Pydantic incluye el input, que puede ser el contenido de
    los mensajes: no debe terminar en un ``ErrorInfo`` ni en los logs.
    """
    detalles = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}"
        for e in exc.errors(include_input=False, include_url=False)
    )
    return f"Request inválido: {detalles}"
