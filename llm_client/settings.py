"""Configuración del cliente LLM leída desde variables de entorno.

Centraliza en un único objeto validado todo lo que el cliente necesita saber
del entorno: qué proveedor usar, sus API keys, modelos, timeout y reintentos.

Por qué así: validar la configuración una sola vez al arrancar permite fallar
rápido con un mensaje claro (por ejemplo, si falta una API key), en vez de
descubrir el problema recién en la primera llamada a la API.

Este módulo NO carga el archivo ``.env``: solo lee ``os.environ``. Cargar el
``.env`` es responsabilidad del punto de entrada (``main.py``), así el paquete
no tiene efectos secundarios al importarse y es fácil de testear.
"""

import os
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

Provider = Literal["openai", "anthropic"]
"""Proveedores de LLM soportados."""

# Mapeo entre variables de entorno y campos de Settings. Son las mismas
# variables documentadas en .env.example.
_ENV_VARS: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_MODEL": "openai_model",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "ANTHROPIC_MODEL": "anthropic_model",
    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
    "LLM_MAX_RETRIES": "max_retries",
}


class Settings(BaseModel):
    """Configuración validada e inmutable del cliente LLM.

    Las API keys se guardan como ``SecretStr`` para que nunca aparezcan en
    logs, ``repr`` o tracebacks por accidente: al imprimirlas se muestran como
    ``'**********'``. El valor real solo se obtiene con
    ``get_secret_value()``, y únicamente debe hacerse al crear el cliente del
    SDK.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @classmethod
    def from_env(cls, provider: Provider | None = None) -> Self:
        """Construye la configuración a partir de ``os.environ``.

        Si se pasa ``provider``, reemplaza a ``LLM_PROVIDER``. Sirve para
        código que funciona con un único proveedor (como el pipeline de
        extracción, que solo usa Anthropic) y no debe depender de qué
        proveedor esté elegido en el entorno.

        Las variables ausentes o vacías se omiten para que apliquen los
        valores por defecto del modelo. Así, una línea como ``OPENAI_API_KEY=``
        en el ``.env`` cuenta como "no configurada" y no como una key vacía.
        Pydantic convierte los strings a ``float``/``int`` donde corresponde.
        """
        valores: dict[str, str] = {}
        for variable, campo in _ENV_VARS.items():
            valor = os.environ.get(variable, "").strip()
            if valor:
                valores[campo] = valor
        if provider is not None:
            valores["provider"] = provider
        return cls.model_validate(valores)

    @model_validator(mode="after")
    def _validar_api_key(self) -> Self:
        """Exige la API key del proveedor elegido.

        Solo se pide la key del proveedor activo: no tiene sentido obligar a
        configurar la de OpenAI si se va a usar Anthropic, y viceversa.
        """
        if self.provider == "openai" and self.openai_api_key is None:
            raise ValueError("Falta OPENAI_API_KEY para usar el proveedor openai")
        if self.provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("Falta ANTHROPIC_API_KEY para usar el proveedor anthropic")
        return self
