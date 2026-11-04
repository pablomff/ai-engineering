"""Contrato común de los clientes LLM.

Define ``BaseLLMClient``, la clase abstracta que implementa cada proveedor
(OpenAI, Anthropic). Por qué existe: el resto de la app programa contra esta
interfaz y nunca contra un SDK concreto, así cambiar de proveedor es cambiar
qué subclase se instancia, sin tocar el código que la usa.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

from llm_client.schemas import ChatRequest, ModelResponse
from llm_client.settings import Settings


class BaseLLMClient(ABC):
    """Clase base abstracta que cumple cualquier proveedor de LLM.

    Contrato de errores: ``generate`` y ``stream`` nunca dejan escapar
    excepciones del SDK. Ante cualquier error del proveedor lanzan una
    subclase de ``LLMError`` (``llm_client.exceptions``), traduciendo la
    excepción original. Así quien llama solo necesita capturar ``LLMError``
    y nunca depende de las excepciones de ``openai`` o ``anthropic``.

    Por qué ``stream`` es un generador asíncrono: permite entregar cada
    fragmento de texto apenas llega, en vez de esperar la respuesta completa.
    Eso reduce el TTFT (time to first token): el usuario empieza a ver texto
    antes de que el modelo termine de generar. Al ser asíncrono, mientras se
    espera el siguiente fragmento el event loop puede atender otras tareas.

    Uso recomendado con ``async with``, que garantiza que ``aclose`` se llame
    siempre, incluso si ocurre una excepción::

        async with MiCliente(settings) as cliente:
            respuesta = await cliente.generate(request)
    """

    def __init__(self, settings: Settings) -> None:
        """Guarda la configuración validada del cliente."""
        self.settings: Settings = settings

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nombre del proveedor (por ejemplo ``"openai"``), usado en respuestas y errores."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Modelo a usar cuando el request no especifica uno."""

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Genera la respuesta completa para el request.

        Ante un error del SDK lanza una subclase de ``LLMError``, nunca la
        excepción original del SDK.
        """

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Genera la respuesta en fragmentos de texto a medida que llegan.

        Las subclases lo implementan como ``async def`` con ``yield``. Mismo
        contrato de errores que ``generate``: ante un error del SDK lanza una
        subclase de ``LLMError``.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Libera las conexiones HTTP del cliente del SDK."""

    async def __aenter__(self) -> Self:
        """Permite usar el cliente con ``async with``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cierra el cliente al salir del ``async with``, haya error o no.

        No suprime la excepción: devuelve ``None``, así cualquier error dentro
        del bloque se sigue propagando.
        """
        await self.aclose()

    def _resolve_model(self, request: ChatRequest) -> str:
        """Devuelve el modelo del request o, si no tiene, el del proveedor.

        Permite elegir el modelo por request sin obligar a hacerlo siempre.
        """
        return request.config.model or self.default_model
