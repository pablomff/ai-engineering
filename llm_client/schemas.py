"""Schemas del cliente LLM unificado.

Define los modelos Pydantic v2 que forman el "idioma común" del proyecto:
mensajes, configuración de generación, requests, respuestas y errores.

Por qué existen: el resto del código nunca trabaja con tipos de los SDKs de
OpenAI o Anthropic, sino con estos modelos. Así cambiar de proveedor no afecta
a quien usa el cliente, y los datos se validan una sola vez, en el borde.

Todos los modelos son inmutables (``frozen=True``) para que un request o una
respuesta no se modifiquen por accidente después de validados, y rechazan
campos desconocidos (``extra="forbid"``) para detectar errores de tipeo.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# Configuración compartida por todos los modelos del módulo.
_CONFIG = ConfigDict(extra="forbid", frozen=True)

Role = Literal["system", "user", "assistant"]
"""Roles de mensaje admitidos por ambos proveedores."""

ErrorKind = Literal[
    "auth",
    "rate_limit",
    "timeout",
    "connection",
    "provider_unavailable",
    "bad_request",
    "unknown",
]
"""Categorías de error independientes del proveedor."""


class ChatMessage(BaseModel):
    """Un mensaje de la conversación.

    El contenido se limpia de espacios sobrantes y debe tener al menos un
    carácter después de limpiarlo: un mensaje vacío no aporta nada y algunos
    proveedores lo rechazan con un error poco claro.
    """

    model_config = _CONFIG

    role: Role
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModelConfig(BaseModel):
    """Parámetros de generación del modelo.

    - ``model``: si es ``None`` se usa el modelo por defecto del proveedor
      configurado (por ejemplo, el definido en las variables de entorno).
    - ``temperature``: el rango 0.0–2.0 es el que acepta OpenAI. Los modelos
      actuales de Anthropic no aceptan ``temperature``, así que el cliente de
      Anthropic la ignora.
    - ``max_tokens``: límite de tokens de salida, entre 1 y 64000.
    """

    model_config = _CONFIG

    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, le=64000)


class ChatRequest(BaseModel):
    """Pedido de generación: la conversación más la configuración.

    Valida la estructura de la conversación antes de llegar al proveedor, para
    fallar rápido y con un mensaje claro en vez de gastar una llamada a la API.
    """

    model_config = _CONFIG

    messages: list[ChatMessage] = Field(min_length=1)
    config: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="after")
    def _validar_conversacion(self) -> Self:
        """Verifica el orden de los roles en la conversación.

        - Como máximo un mensaje ``system``, y si existe debe ser el primero:
          Anthropic recibe el system prompt aparte y OpenAI lo espera al inicio.
        - El último mensaje debe ser de ``user``, porque es lo que el modelo
          tiene que responder.
        """
        indices_system = [i for i, m in enumerate(self.messages) if m.role == "system"]
        if len(indices_system) > 1:
            raise ValueError("Solo se permite un mensaje 'system'.")
        if indices_system and indices_system[0] != 0:
            raise ValueError("El mensaje 'system' debe ser el primero.")
        if self.messages[-1].role != "user":
            raise ValueError("El último mensaje debe ser de 'user'.")
        return self


class ErrorInfo(BaseModel):
    """Error estructurado y serializable, independiente del proveedor.

    Permite devolver el error dentro de la respuesta en vez de propagar
    excepciones de los SDKs. ``message`` debe ser legible y no contener datos
    sensibles (API keys, contenido de los mensajes). ``retryable`` indica si
    tiene sentido volver a intentar más tarde.
    """

    model_config = _CONFIG

    kind: ErrorKind
    message: str
    retryable: bool
    provider: str


class TokenUsage(BaseModel):
    """Tokens consumidos en una generación, para medir costo y uso."""

    model_config = _CONFIG

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Resultado de una generación, exitosa o fallida.

    Usa la misma forma para éxito y error (``ok`` + ``error``) para que quien
    llama siempre reciba un objeto del mismo tipo.

    ``finish_reason`` indica por qué terminó el modelo: fin normal, límite de
    tokens, rechazo, etc. Es ``None`` si el proveedor no lo informa o si hubo
    error.
    """

    model_config = _CONFIG

    ok: bool
    provider: str
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float = Field(ge=0)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _validar_coherencia(self) -> Self:
        """Garantiza que ``ok`` y ``error`` no se contradigan.

        Si ``ok`` es ``True``, ``error`` debe ser ``None``. Si ``ok`` es
        ``False``, ``error`` es obligatorio.
        """
        if self.ok and self.error is not None:
            raise ValueError("Una respuesta exitosa (ok=True) no puede tener error.")
        if not self.ok and self.error is None:
            raise ValueError("Una respuesta fallida (ok=False) debe incluir error.")
        return self
