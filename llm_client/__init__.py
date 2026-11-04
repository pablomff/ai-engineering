"""Unified Async LLM Client.

Cliente asíncrono unificado para OpenAI y Anthropic. Expone una única API
pública, independiente del proveedor: la aplicación solo necesita lo que se
exporta acá y nunca importa los SDKs directamente.
"""

from llm_client.exceptions import LLMError
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse
from llm_client.settings import Settings

__all__ = [
    "AsyncLLMManager",
    "ChatMessage",
    "ErrorInfo",
    "LLMError",
    "ModelConfig",
    "ModelResponse",
    "Settings",
]
