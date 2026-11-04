"""Tests de validación de los schemas.

Verifican que los datos inválidos se rechacen en el borde, antes de llegar a
un proveedor: así un error de uso se detecta sin gastar una llamada a la API.
"""

import pytest
from pydantic import ValidationError

from llm_client.schemas import ChatMessage, ChatRequest, ErrorInfo, ModelConfig, ModelResponse


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_fuera_de_rango(temperature: float) -> None:
    """La temperatura debe estar entre 0.0 y 2.0."""
    with pytest.raises(ValidationError):
        ModelConfig(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, 64001])
def test_max_tokens_fuera_de_rango(max_tokens: int) -> None:
    """max_tokens debe ser mayor a 0 y menor o igual a 64000."""
    with pytest.raises(ValidationError):
        ModelConfig(max_tokens=max_tokens)


def test_campo_extra_rechazado() -> None:
    """extra="forbid" detecta campos desconocidos (por ejemplo, errores de tipeo)."""
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"temperatura": 0.5})


@pytest.mark.parametrize("content", ["", "   "])
def test_content_vacio_rechazado(content: str) -> None:
    """El content vacío, o solo con espacios, se rechaza."""
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=content)


def test_system_que_no_es_el_primero() -> None:
    """El mensaje system, si existe, tiene que ser el primero."""
    with pytest.raises(ValidationError, match="system"):
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hola"),
                ChatMessage(role="system", content="Sé breve"),
                ChatMessage(role="user", content="¿Qué es la entropía?"),
            ]
        )


def test_ultimo_mensaje_no_es_user() -> None:
    """La conversación tiene que terminar con un mensaje de user."""
    with pytest.raises(ValidationError, match="user"):
        ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hola"),
                ChatMessage(role="assistant", content="¡Hola!"),
            ]
        )


def test_model_response_ok_con_error_es_incoherente() -> None:
    """Una respuesta exitosa no puede traer error."""
    error = ErrorInfo(kind="timeout", message="Timeout", retryable=True, provider="openai")
    with pytest.raises(ValidationError):
        ModelResponse(ok=True, provider="openai", model="m", latency_ms=1.0, error=error)
