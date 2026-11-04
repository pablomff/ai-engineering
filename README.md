# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI** y **Anthropic**, en Python 3.12.

## Qué resuelve

Cada proveedor de LLM tiene su propio SDK, con formatos de mensajes, parámetros, respuestas y excepciones distintos. Si la aplicación usa los SDKs directamente, queda atada a uno y cambiar de proveedor obliga a tocar código en muchos lugares.

Este proyecto ofrece una única API asíncrona, `AsyncLLMManager`, que:

- recibe y devuelve siempre los mismos modelos (`ChatMessage`, `ModelConfig`, `ModelResponse`), sea cual sea el proveedor;
- elige el proveedor por configuración (`LLM_PROVIDER`), sin cambiar código;
- valida los datos antes de llamar a la API;
- traduce los errores de cada SDK a errores propios y serializables;
- soporta respuestas completas (`generate`) y streaming (`stream`).

## Estructura

```
.
├── main.py                          # Script de validación: prueba modo normal y streaming
├── pyproject.toml                   # Dependencias y configuración de pytest
├── uv.lock                          # Versiones exactas de las dependencias
├── .env.example                     # Plantilla de variables de entorno
├── .gitignore                       # Archivos que no se suben (.env, .venv, cachés)
├── llm_client/
│   ├── __init__.py                  # API pública del paquete
│   ├── schemas.py                   # Modelos Pydantic: mensajes, config, request, respuesta, errores
│   ├── settings.py                  # Configuración validada leída del entorno
│   ├── exceptions.py                # Jerarquía de excepciones propias (LLMError y subclases)
│   ├── base.py                      # Contrato abstracto BaseLLMClient
│   ├── manager.py                   # AsyncLLMManager: punto de entrada de la aplicación
│   └── providers/
│       ├── __init__.py              # Único paquete donde se importan los SDKs
│       ├── openai_client.py         # Implementación para OpenAI (AsyncOpenAI)
│       └── anthropic_client.py      # Implementación para Anthropic (AsyncAnthropic)
└── tests/
    ├── test_schemas.py              # Validaciones de los schemas
    ├── test_manager.py              # Contrato de errores del manager, con un cliente falso
    └── test_error_mapping.py        # Traducción de excepciones de los SDKs a LLMError
```

## Decisiones de diseño

### Contrato con una clase abstracta (ABC)

`BaseLLMClient` define lo que cumple cualquier proveedor: `provider_name`, `default_model`, `generate`, `stream` y `aclose`, además del soporte para `async with`. El manager programa contra esta interfaz, nunca contra un proveedor concreto. Agregar un proveedor nuevo es escribir una subclase y registrarla en el manager.

### Encapsulamiento de los SDKs

Los paquetes `openai` y `anthropic`, y sus excepciones, solo se importan dentro de `llm_client/providers/`. Cada provider traduce nuestros schemas al formato de su SDK (por ejemplo, Anthropic recibe el mensaje `system` en un parámetro aparte) y traduce la respuesta de vuelta a `ModelResponse`. Fuera de ese paquete, nada depende de un SDK.

### Errores propios y `ErrorInfo`

Cada provider convierte las excepciones de su SDK en una subclase de `LLMError` (`LLMAuthError`, `LLMRateLimitError`, etc.), conservando la original como causa (`raise ... from exc`). Así la aplicación solo necesita conocer una jerarquía de errores.

Cada `LLMError` se puede convertir en un `ErrorInfo`, un modelo serializable con `kind`, `message`, `retryable` y `provider`. `generate` nunca lanza excepciones: si algo falla, devuelve `ModelResponse(ok=False, error=ErrorInfo(...))`, así éxito y error se manejan con el mismo objeto.

Los mensajes de error son propios y no copian los del SDK, para no filtrar datos sensibles como fragmentos de la API key o el contenido de los mensajes.

### Reintentos nativos de los SDKs

No hay reintentos manuales. Ambos SDKs ya reintentan los errores transitorios (conexión, timeout, 429, 5xx) con backoff exponencial y respetan los headers `retry-after` del proveedor. Se configuran con `LLM_MAX_RETRIES` y `LLM_TIMEOUT_SECONDS`.

Reimplementar esa lógica duplicaría código ya probado y podría sumar reintentos encima de los del SDK, multiplicando la cantidad de llamadas.

### Streaming con generadores asíncronos y TTFT

`stream` es un generador asíncrono (`async for`) que entrega cada fragmento de texto apenas llega. Eso reduce el **TTFT** (*time to first token*): el usuario empieza a leer la respuesta antes de que el modelo termine de generarla. Mientras se espera el siguiente fragmento, el event loop puede atender otras tareas.

A diferencia de `generate`, `stream` sí propaga los errores, siempre como `LLMError`. En streaming no hay un valor de retorno donde poner el error, y si el corte llega a mitad de la respuesta, quien consume tiene que enterarse de que el texto quedó incompleto.

### Por qué Anthropic ignora `temperature`

Los modelos actuales de Anthropic no aceptan el parámetro `temperature`: enviarlo produce un error 400. Por eso `AnthropicClient` no lo envía, y si se pidió un valor distinto del por defecto, registra un `warning` avisando que se ignora. El rango 0.0–2.0 de `ModelConfig` es el de OpenAI.

## Requisitos e instalación

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp .env.example .env   # completar las API keys
```

## Variables de entorno

| Variable | Obligatoria | Valor por defecto | Para qué sirve |
|---|---|---|---|
| `LLM_PROVIDER` | Sí | — | Proveedor a usar: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | — | API key de OpenAI |
| `OPENAI_MODEL` | Si el proveedor es `openai` | — | Modelo de OpenAI |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | — | API key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-opus-5-5` | Modelo de Anthropic |
| `LLM_TIMEOUT_SECONDS` | No | `30` | Tiempo máximo de espera por request, en segundos |
| `LLM_MAX_RETRIES` | No | `2` | Reintentos nativos del SDK ante errores transitorios (entre 0 y 10) |

Si `main.py` se corre con `--provider`, ese valor reemplaza a `LLM_PROVIDER`.

> El archivo `.env` contiene las API keys y **nunca se sube al repo**: está en `.gitignore`. Al repo solo se sube `.env.example`, sin valores secretos.

## Cómo correr

```bash
# Script de validación con el proveedor de LLM_PROVIDER
uv run python main.py

# Forzando el proveedor
uv run python main.py --provider anthropic

# Tests (no necesitan API keys ni red)
uv run pytest -v
```

`main.py` prueba el modo normal y el modo streaming, y muestra la respuesta, el modelo, `finish_reason`, los tokens, la latencia y el TTFT. Termina con código 0 si ambos modos funcionaron y con 1 si alguno falló o si la configuración es inválida.

## Ejemplo de uso

```python
import asyncio

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, LLMError, ModelConfig


async def main() -> None:
    load_dotenv()
    messages = [
        ChatMessage(role="system", content="Respondé en español, en forma breve."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]

    async with AsyncLLMManager.from_env() as llm:
        # Respuesta completa: generate nunca lanza, se revisa respuesta.ok.
        respuesta = await llm.generate(messages, ModelConfig(max_tokens=500))
        if respuesta.ok:
            print(respuesta.content)
            print(respuesta.model, respuesta.usage, f"{respuesta.latency_ms:.0f} ms")
        else:
            print(f"Error {respuesta.error.kind}: {respuesta.error.message}")

        # Streaming: los errores llegan como LLMError.
        try:
            async for fragmento in llm.stream(messages):
                print(fragmento, end="", flush=True)
        except LLMError as exc:
            print(f"\nError {exc.kind}: {exc.message}")


asyncio.run(main())
```

## Manejo de errores

| `kind` | Cuándo ocurre | Reintentable |
|---|---|---|
| `auth` | API key inválida o sin permisos (HTTP 401, 403) | No |
| `rate_limit` | Se superó el límite de requests o tokens (HTTP 429) | Sí |
| `timeout` | El proveedor no respondió dentro de `LLM_TIMEOUT_SECONDS` | Sí |
| `connection` | No se pudo establecer la conexión de red | Sí |
| `provider_unavailable` | Servicio caído o sobrecargado (HTTP 5xx, incluido el 529 de Anthropic) | Sí |
| `bad_request` | Request inválido: falló la validación local o el proveedor respondió con otro 4xx (modelo inexistente, parámetros) | No |
| `unknown` | Cualquier otro error no clasificado | No |

"Reintentable" indica si tiene sentido volver a intentar más tarde. Los reintentos inmediatos ya los hace el SDK antes de devolver el error.
