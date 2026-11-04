"""Tests de la traducción de excepciones de los SDKs a nuestras LLMError.

Construyen las excepciones reales de ``openai`` y ``anthropic`` (con
requests y responses HTTP armados a mano, sin red) y verifican que
``_map_error`` de cada proveedor devuelva la ``LLMError`` correcta.
"""

from collections.abc import AsyncIterator

import anthropic
import httpx2
import openai
import pytest

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
from llm_client.providers.anthropic_client import AnthropicClient
from llm_client.providers.openai_client import OpenAIClient
from llm_client.settings import Settings

REQUEST = httpx2.Request("POST", "https://api.example.com/v1/test")


def response(status: int) -> httpx2.Response:
    """Arma una respuesta HTTP con el status indicado."""
    return httpx2.Response(status, request=REQUEST)


@pytest.fixture
async def openai_client() -> AsyncIterator[OpenAIClient]:
    """Cliente de OpenAI con configuración falsa. No hace llamadas."""
    settings = Settings(provider="openai", openai_api_key="fake-key", openai_model="fake-model")
    async with OpenAIClient(settings) as client:
        yield client


@pytest.fixture
async def anthropic_client() -> AsyncIterator[AnthropicClient]:
    """Cliente de Anthropic con configuración falsa. No hace llamadas."""
    settings = Settings(provider="anthropic", anthropic_api_key="fake-key")
    async with AnthropicClient(settings) as client:
        yield client


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (openai.AuthenticationError("x", response=response(401), body=None), LLMAuthError),
        (openai.PermissionDeniedError("x", response=response(403), body=None), LLMAuthError),
        (openai.RateLimitError("x", response=response(429), body=None), LLMRateLimitError),
        # APITimeoutError es subclase de APIConnectionError: tiene que ganar timeout.
        (openai.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (openai.APIConnectionError(request=REQUEST), LLMConnectionError),
        (openai.InternalServerError("x", response=response(500), body=None), LLMProviderUnavailableError),
        (openai.APIStatusError("x", response=response(503), body=None), LLMProviderUnavailableError),
        (openai.BadRequestError("x", response=response(400), body=None), LLMBadRequestError),
        (openai.NotFoundError("x", response=response(404), body=None), LLMBadRequestError),
        (openai.OpenAIError("x"), LLMUnknownError),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v.__name__,
)
def test_map_error_openai(
    openai_client: OpenAIClient, exc: openai.OpenAIError, expected: type[LLMError]
) -> None:
    """Cada excepción de openai se traduce a la LLMError correspondiente."""
    error = openai_client._map_error(exc)
    assert type(error) is expected
    assert error.provider == "openai"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (anthropic.AuthenticationError("x", response=response(401), body=None), LLMAuthError),
        (anthropic.PermissionDeniedError("x", response=response(403), body=None), LLMAuthError),
        (anthropic.RateLimitError("x", response=response(429), body=None), LLMRateLimitError),
        (anthropic.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (anthropic.APIConnectionError(request=REQUEST), LLMConnectionError),
        # 529: overloaded_error, propio de Anthropic.
        (anthropic.OverloadedError("x", response=response(529), body=None), LLMProviderUnavailableError),
        (anthropic.InternalServerError("x", response=response(500), body=None), LLMProviderUnavailableError),
        (anthropic.BadRequestError("x", response=response(400), body=None), LLMBadRequestError),
        (anthropic.NotFoundError("x", response=response(404), body=None), LLMBadRequestError),
        (anthropic.AnthropicError("x"), LLMUnknownError),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v.__name__,
)
def test_map_error_anthropic(
    anthropic_client: AnthropicClient, exc: anthropic.AnthropicError, expected: type[LLMError]
) -> None:
    """Cada excepción de anthropic se traduce a la LLMError correspondiente."""
    error = anthropic_client._map_error(exc)
    assert type(error) is expected
    assert error.provider == "anthropic"


def test_retryable_coherente_con_el_tipo(anthropic_client: AnthropicClient) -> None:
    """Un 529 es reintentable y un 401 no."""
    sobrecarga = anthropic.OverloadedError("x", response=response(529), body=None)
    auth = anthropic.AuthenticationError("x", response=response(401), body=None)
    assert anthropic_client._map_error(sobrecarga).retryable is True
    assert anthropic_client._map_error(auth).retryable is False
