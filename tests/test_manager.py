"""Tests del contrato de errores de AsyncLLMManager.

Usan un cliente falso que cumple ``BaseLLMClient``, registrado en el manager
con ``monkeypatch``: así se prueba el manager sin API keys, sin red y sin
tocar código de producción.
"""

from collections.abc import AsyncIterator, Callable

import pytest

from llm_client import manager as manager_module
from llm_client.base import BaseLLMClient
from llm_client.exceptions import LLMError, LLMRateLimitError
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, ChatRequest, ModelResponse
from llm_client.settings import Settings

MESSAGES = [ChatMessage(role="user", content="¿Qué es la entropía?")]


class FakeClient(BaseLLMClient):
    """Cliente falso cuyo comportamiento se configura con atributos de clase.

    - ``generate_error``: si no es ``None``, ``generate`` lanza esa excepción.
    - ``fragments``: fragmentos que devuelve ``stream``.
    - ``stream_error``: si no es ``None``, ``stream`` la lanza después de los
      fragmentos (simula un corte a mitad de camino).
    """

    generate_error: Exception | None = None
    fragments: list[str] = []
    stream_error: Exception | None = None

    @property
    def provider_name(self) -> str:
        """Nombre del proveedor falso."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Modelo falso."""
        return "fake-model"

    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Devuelve una respuesta fija o lanza ``generate_error``."""
        if self.generate_error is not None:
            raise self.generate_error
        return ModelResponse(
            ok=True,
            provider=self.provider_name,
            model=self._resolve_model(request),
            content="La entropía mide el desorden.",
            latency_ms=1.0,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Devuelve ``fragments`` y, si está configurado, lanza ``stream_error``."""
        for fragment in self.fragments:
            yield fragment
        if self.stream_error is not None:
            raise self.stream_error

    async def aclose(self) -> None:
        """No hay conexiones que cerrar."""


ManagerFactory = Callable[..., AsyncLLMManager]


@pytest.fixture
def make_manager(monkeypatch: pytest.MonkeyPatch) -> ManagerFactory:
    """Devuelve una función que crea un manager con un ``FakeClient`` configurado.

    Se crea una subclase nueva por test para que la configuración de un test
    no afecte a otro.
    """

    def factory(
        generate_error: Exception | None = None,
        fragments: list[str] | None = None,
        stream_error: Exception | None = None,
    ) -> AsyncLLMManager:
        client_cls = type(
            "ConfiguredFakeClient",
            (FakeClient,),
            {
                "generate_error": generate_error,
                "fragments": fragments or [],
                "stream_error": stream_error,
            },
        )
        monkeypatch.setitem(manager_module._PROVIDERS, "anthropic", client_cls)
        settings = Settings(provider="anthropic", anthropic_api_key="fake-key")
        return AsyncLLMManager(settings)

    return factory


async def test_generate_exitoso(make_manager: ManagerFactory) -> None:
    """Si el cliente responde bien, generate devuelve ok=True."""
    async with make_manager() as llm:
        respuesta = await llm.generate(MESSAGES)
    assert respuesta.ok is True
    assert respuesta.error is None
    assert respuesta.content == "La entropía mide el desorden."


async def test_generate_rate_limit_no_lanza(make_manager: ManagerFactory) -> None:
    """Una LLMError se devuelve como ok=False con su kind, sin lanzar."""
    error = LLMRateLimitError("Límite superado", provider="anthropic")
    async with make_manager(generate_error=error) as llm:
        respuesta = await llm.generate(MESSAGES)
    assert respuesta.ok is False
    assert respuesta.error is not None
    assert respuesta.error.kind == "rate_limit"
    assert respuesta.error.retryable is True


async def test_generate_excepcion_inesperada(make_manager: ManagerFactory) -> None:
    """Una excepción que no es LLMError se devuelve como kind "unknown"."""
    async with make_manager(generate_error=RuntimeError("boom")) as llm:
        respuesta = await llm.generate(MESSAGES)
    assert respuesta.ok is False
    assert respuesta.error is not None
    assert respuesta.error.kind == "unknown"


async def test_stream_fragmentos_en_orden(make_manager: ManagerFactory) -> None:
    """stream reenvía los fragmentos en el orden en que llegan."""
    async with make_manager(fragments=["La ", "entropía ", "mide..."]) as llm:
        recibidos = [fragment async for fragment in llm.stream(MESSAGES)]
    assert recibidos == ["La ", "entropía ", "mide..."]


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("Límite superado", provider="anthropic"),
        RuntimeError("Conexión cortada"),
    ],
    ids=["llm_error", "excepcion_generica"],
)
async def test_stream_falla_a_mitad_propaga_llm_error(
    make_manager: ManagerFactory, error: Exception
) -> None:
    """Si el stream se corta a mitad de camino, se propaga siempre una LLMError.

    Quien consume recibe los fragmentos que ya llegaron y después el error.
    """
    recibidos: list[str] = []
    async with make_manager(fragments=["La ", "entropía "], stream_error=error) as llm:
        with pytest.raises(LLMError):
            async for fragment in llm.stream(MESSAGES):
                recibidos.append(fragment)
    assert recibidos == ["La ", "entropía "]
