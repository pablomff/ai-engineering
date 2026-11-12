# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI** y **Anthropic**, compatible con Python 3.11 y 3.12.

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
│       ├── anthropic_client.py      # Implementación para Anthropic (AsyncAnthropic)
│       └── langchain_anthropic.py   # ChatAnthropic de LangChain para la Pre-entrega 2
├── pipelines/                       # Pipelines LCEL, uno por subpaquete
│   └── extraccion/                  # Pre-entrega 2: extracción de entidades técnicas validada
│       ├── schemas.py               # Modelo Pydantic ExtraccionTecnica (contrato de salida)
│       ├── chain.py                 # Cadena LCEL, reintentos y process_text()
│       └── demo.py                  # Mini-script de prueba asíncrono
├── modulo-2-ejercicio/
│   └── lcel_chain.py                # Ejercicio del Módulo 2: prompt | modelo | StrOutputParser
└── tests/
    ├── test_schemas.py              # Validaciones de los schemas
    ├── test_manager.py              # Contrato de errores del manager, con un cliente falso
    ├── test_error_mapping.py        # Traducción de excepciones de los SDKs a LLMError
    └── test_extraccion.py           # Esquema, validación de la salida y cadena del pipeline de extracción
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

- Python 3.11 o superior. El proyecto se desarrolla con 3.12 (`.python-version`) y los tests pasan con 3.11 y 3.12.
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync                # si no tenés Python 3.12, uv lo descarga solo
cp .env.example .env   # completar las API keys
```

Para usar el Python 3.11 que ya tengas instalado en lugar de 3.12:

```bash
uv sync --python 3.11
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

# Tests con Python 3.11, en un entorno temporal que no toca el .venv
uv run --isolated --python 3.11 pytest -v
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

## Pre-entrega 2: pipeline de extracción validada

El código está en `pipelines/extraccion/`.

Recibe un texto sin procesar, como una descripción de arquitectura o un log de error, y devuelve un objeto `ExtraccionTecnica` validado con Pydantic.

### Cómo correr

```bash
uv sync                                  # instala Python (3.12 por defecto) y las dependencias
cp .env.example .env                     # completar las API keys
uv run pytest tests/test_extraccion.py -v
```

El mini-script de prueba procesa el texto que le pases, o dos ejemplos de demostración si no le pasás ninguno:

```bash
# Sin texto: corre los ejemplos (una descripción de arquitectura y un texto ambiguo como prueba de estrés)
uv run python -m pipelines.extraccion.demo

# Con un texto como argumento
uv run python -m pipelines.extraccion.demo "El checkout en Next.js tarda 8 segundos porque la API en Django consulta MySQL sin índices"

# Con el contenido de un archivo, por la entrada estándar
cat error.log | uv run python -m pipelines.extraccion.demo
```

El JSON sale por la salida estándar y los logs por la de errores, así que `> resultado.json` guarda solo el resultado. Termina con código 0 si obtuvo una extracción válida y con 1 si no.

Usa Anthropic a través de `ChatAnthropic`, configurado con los mismos `Settings` del Módulo 1 (`llm_client`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `LLM_TIMEOUT_SECONDS` y `LLM_MAX_RETRIES`. Solo hace falta `ANTHROPIC_API_KEY`: el pipeline siempre usa Anthropic, así que ignora `LLM_PROVIDER`. El modelo por defecto, `claude-opus-5-5`, es Claude Opus 5.5, el modelo actual de Anthropic.

### Uso

```python
from pipelines.extraccion.chain import process_text

extraccion = await process_text("Nuestra API en FastAPI usa Redis y PostgreSQL...")
print(extraccion.model_dump_json(indent=2))
```

### Ejemplo de salida

Para el texto de arquitectura de `demo.py` (FastAPI + Redis + PostgreSQL con el pool de conexiones agotado):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Bajo picos de 2000 usuarios concurrentes, el pool de conexiones a PostgreSQL de la API en FastAPI se agota, generando latencias superiores a 10 segundos y errores 503 en producción."
}
```

Logs de una corrida del demo. Los dos textos se procesan en paralelo; el segundo es el texto ambiguo de la prueba de estrés, que no menciona tecnologías:

```
INFO pipelines.extraccion.chain: Procesando texto de 281 caracteres
INFO pipelines.extraccion.chain: Procesando texto de 67 caracteres
INFO pipelines.extraccion.chain: Salida validada: 3 tecnologías, criticidad=alta, stop_reason=end_turn
INFO pipelines.extraccion.chain: Extracción completada en 2872 ms
WARNING pipelines.extraccion.chain: El texto no menciona tecnologías: no se reintenta
```

Ante una salida mal formada, en cambio, cada intento fallido deja un `WARNING` con el motivo (por ejemplo, `Salida inválida: no cumple el esquema (campos: nivel_de_criticidad)`), y si se agotan los 3 intentos se registra `ERROR ... Sin salida válida después de 3 intentos`. Los tests `test_cadena_reintenta_y_se_recupera` y `test_cadena_se_detiene_despues_de_max_intentos` cubren los dos casos.

### Componentes

| Componente | Dónde | Qué hace |
|---|---|---|
| Esquema Pydantic | `schemas.py` | `tecnologias` (lista no vacía, sin duplicados), `nivel_de_criticidad` (enum `baja`/`media`/`alta`) y `resumen_tecnico` (no vacío). No acepta campos extra. |
| Prompt template | `chain.py` → `crear_prompt()` | `ChatPromptTemplate` con roles System/Human y dos variables: `texto` e `instrucciones_formato`. La segunda se fija con `partial`, así que la cadena solo recibe `{"texto": ...}`. |
| Cadena LCEL | `chain.py` → `crear_cadena()` | `prompt \| model.with_structured_output(ExtraccionTecnica) \| validar_salida`, envuelta en `.with_retry()`. |
| Resiliencia | `chain.py` → `validar_salida()` | Rechaza respuestas cortadas por el límite de tokens, JSON mal formado y objetos que no cumplen el esquema. Cada rechazo dispara un nuevo intento, hasta 3. Un texto sin tecnologías o una negativa del modelo cortan sin reintentar. |
| Función asíncrona | `chain.py` → `process_text()` | Ejecuta la cadena con `.ainvoke()` y loguea el proceso de validación. |

### Decisiones de diseño

**Dos niveles de reintento.** Los errores de red y del proveedor (timeout, 429, 5xx) los reintenta el SDK, como en el Módulo 1. `.with_retry()` solo reintenta `ExtraccionInvalidaError`, para no sumar reintentos encima de los del SDK. Con `include_raw=True`, LangChain no lanza los errores de parseo: los devuelve en `parsing_error` y `validar_salida` los convierte en `ExtraccionInvalidaError`.

**Solo se reintenta lo que puede cambiar.** Un JSON mal formado es un error aleatorio del modelo y otro intento puede corregirlo. Una lista de tecnologías vacía o una negativa del modelo (`stop_reason == "refusal"`) se repetirían igual con el mismo texto, así que lanzan `TextoSinTecnologiasError` o `ModeloNegativaError` en el primer intento, sin gastar llamadas.

**Se revisa el `finish_reason` antes de confiar en el objeto.** La cadena usa `with_structured_output(..., include_raw=True)` para tener acceso al mensaje crudo. Si el modelo se detuvo por el límite de tokens (`stop_reason == "max_tokens"`, el `finish_reason` de Anthropic), la respuesta se descarta aunque haya parseado: un resumen cortado a la mitad puede pasar la validación de Pydantic.

**`method="json_schema"`.** Es la salida estructurada nativa de la API. El método por defecto de `ChatAnthropic`, que fuerza una llamada a una herramienta, no está soportado en los modelos Claude actuales.

**La lista de tecnologías no vacía se valida en Pydantic, no en el esquema JSON.** Con `Field(min_length=1)`, el esquema que recibe el modelo incluye `minItems: 1` y la API obliga al modelo a respetarlo. En la prueba de estrés, ante un texto sin tecnologías, el modelo inventaba una ("No identificada", "Servicio no especificado") para cumplir. Con la regla en un `field_validator`, el modelo puede devolver la lista vacía y Pydantic la rechaza con un tipo de error propio (`sin_tecnologias`). Así `process_text` lanza `TextoSinTecnologiasError` en vez de devolver datos inventados.

**Errores propios.** Además de `ValueError` para un texto vacío, `process_text` solo lanza subclases de `ExtraccionError`:

| Excepción | Cuándo |
|---|---|
| `ExtraccionConfigError` | Falta `ANTHROPIC_API_KEY` u otra configuración es inválida |
| `TextoSinTecnologiasError` | El texto no menciona tecnologías |
| `ModeloNegativaError` | El modelo se negó a procesar el texto |
| `ExtraccionInvalidaError` | Después de 3 intentos no hubo una salida válida |
| `ExtraccionProveedorError` | Falló la llamada a Anthropic después de los reintentos del SDK. Guarda la `LLMError` del Módulo 1 en `.error`, con su `kind` y `retryable` |

Cualquier otra excepción es un bug y se propaga sin envolver.

**El SDK queda dentro de `llm_client/providers/`.** `ChatAnthropic` se crea en `providers/langchain_anthropic.py`, que también traduce las excepciones del SDK a `LLMError` reutilizando la traducción de `AnthropicClient`. `pipelines/extraccion` no importa nada de `anthropic`.

**Una sola cadena por proceso.** Si no se le pasa una cadena, `process_text` usa una que crea la primera vez y reutiliza después, así todas las llamadas comparten el cliente HTTP y su pool de conexiones.

**Logs sin datos sensibles.** Se loguean el largo del texto, la cantidad de tecnologías, la criticidad, el `stop_reason` y los nombres de los campos que fallaron, pero nunca el texto ni los valores recibidos.
