"""Jerarquía de excepciones propias del cliente LLM.

Por qué existen: cada SDK (``openai``, ``anthropic``) lanza sus propias
excepciones, con nombres y estructuras distintas. Si el resto de la app las
capturara directamente, quedaría atada a un SDK concreto y cambiar de
proveedor obligaría a tocar código en muchos lugares.

Por eso los providers (``llm_client/providers/``) traducen las excepciones del
SDK a estas, y el resto del código solo conoce esta jerarquía. Cada subclase
tiene un ``kind`` fijo y un valor de ``retryable`` por defecto, y todas pueden
convertirse a ``ErrorInfo`` para devolverse dentro de un ``ModelResponse``.
"""

from typing import ClassVar

from llm_client.schemas import ErrorInfo, ErrorKind


class LLMError(Exception):
    """Excepción base de todos los errores del cliente LLM.

    Guarda el proveedor que falló, un mensaje legible (sin datos sensibles) y
    si tiene sentido reintentar. Capturar ``LLMError`` alcanza para manejar
    cualquier error del cliente sin conocer los detalles de cada SDK.
    """

    kind: ClassVar[ErrorKind] = "unknown"
    """Categoría del error; cada subclase define la suya."""

    default_retryable: ClassVar[bool] = False
    """Valor de ``retryable`` que se usa si no se indica uno explícito."""

    def __init__(self, message: str, *, provider: str, retryable: bool | None = None) -> None:
        """Crea el error.

        ``retryable`` es opcional: si no se pasa, se usa el valor por defecto
        de la subclase. Se deja sobrescribir por si un provider tiene más
        información (por ejemplo, un error 400 que sí conviene reintentar).
        """
        super().__init__(message)
        self.message: str = message
        self.provider: str = provider
        self.retryable: bool = self.default_retryable if retryable is None else retryable

    def to_error_info(self) -> ErrorInfo:
        """Convierte la excepción en un ``ErrorInfo`` serializable.

        Permite devolver el error dentro de un ``ModelResponse`` en vez de
        propagar la excepción.
        """
        return ErrorInfo(
            kind=self.kind,
            message=self.message,
            retryable=self.retryable,
            provider=self.provider,
        )


class LLMAuthError(LLMError):
    """API key inválida, ausente o sin permisos. Reintentar no lo arregla."""

    kind = "auth"
    default_retryable = False


class LLMRateLimitError(LLMError):
    """Se superó el límite de requests o tokens. Se resuelve esperando."""

    kind = "rate_limit"
    default_retryable = True


class LLMTimeoutError(LLMError):
    """El proveedor no respondió dentro del timeout configurado."""

    kind = "timeout"
    default_retryable = True


class LLMConnectionError(LLMError):
    """No se pudo establecer la conexión de red con el proveedor."""

    kind = "connection"
    default_retryable = True


class LLMProviderUnavailableError(LLMError):
    """El servicio del proveedor está caído o sobrecargado (errores 5xx)."""

    kind = "provider_unavailable"
    default_retryable = True


class LLMBadRequestError(LLMError):
    """El request es inválido (parámetros, modelo inexistente, etc.).

    No es reintentable: el mismo request va a fallar siempre igual.
    """

    kind = "bad_request"
    default_retryable = False


class LLMUnknownError(LLMError):
    """Error no clasificado. Por prudencia se considera no reintentable."""

    kind = "unknown"
    default_retryable = False
