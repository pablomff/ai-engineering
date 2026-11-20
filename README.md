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
├── data/                            # Dataset de ejemplo del RAG (.md y .txt de una empresa ficticia)
├── pipelines/                       # Pipelines LCEL, uno por subpaquete
│   ├── extraccion/                  # Pre-entrega 2: extracción de entidades técnicas validada
│   │   ├── schemas.py               # Modelo Pydantic ExtraccionTecnica (contrato de salida)
│   │   ├── chain.py                 # Cadena LCEL, reintentos y process_text()
│   │   └── demo.py                  # Mini-script de prueba asíncrono
│   ├── rag/                         # Pre-entrega 3: RAG local con ChromaDB
│   │   ├── vectorstore.py           # Modelo de embeddings y base ChromaDB (compartidos)
│   │   ├── ingesta.py               # Script de carga: lectura, chunking y persistencia
│   │   ├── schemas.py               # RespuestaRAG: texto + referencias
│   │   ├── chain.py                 # Retriever, cadena LCEL y get_rag_response()
│   │   ├── exceptions.py            # Errores propios del pipeline
│   │   └── demo.py                  # Pregunta con respuesta + pregunta trampa
│   └── modulo-1-ejercicio/          # Independiente: solo librería estándar
│       ├── orquestador.py          # Ejercicio del Módulo 1: gather, timeout y Semaphore
│       └── test_orquestador.py      # Tests: uv run pytest modulo-1-ejercicio
├── modulo-2-ejercicio/
│   └── lcel_chain.py                # Ejercicio del Módulo 2: prompt | modelo | StrOutputParser
└── tests/
    ├── test_schemas.py              # Validaciones de los schemas
    ├── test_manager.py              # Contrato de errores del manager, con un cliente falso
    ├── test_error_mapping.py        # Traducción de excepciones de los SDKs a LLMError
    ├── test_extraccion.py           # Esquema, validación de la salida y cadena del pipeline de extracción
    └── test_rag.py                  # Ingesta, base vectorial, contrato y cadena del RAG
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

Sin uv, con `pip` en un entorno virtual (pip 25.1 o superior, que lee los grupos de dependencias):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . --group dev   # el proyecto, sus dependencias y las de los tests
pytest
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

## Pre-entrega 3: sistema RAG local

El código está en `pipelines/rag/` y el dataset de ejemplo en `data/`.

Responde preguntas usando **solo** los documentos de `data/`: busca los fragmentos más parecidos a la pregunta en una base ChromaDB local y le pide al LLM una respuesta basada en ellos, con las referencias a los fragmentos usados. Si los documentos no tienen la respuesta, responde "No lo sé.".

El dataset son cuatro documentos internos de **Lumen Pagos, una empresa ficticia**: arquitectura, runbook de incidentes, política de despliegues y onboarding. Que sea ficticia es a propósito: el modelo no puede responder con lo que ya sabe, así que cualquier dato que no esté en los documentos sería una alucinación.

### Cómo correr

```bash
uv sync                                   # instala las dependencias (incluye torch para los embeddings)
cp .env.example .env                      # completar ANTHROPIC_API_KEY

uv run python -m pipelines.rag.ingesta    # 1. arma la base en ./vectorstore (una sola vez)
uv run python -m pipelines.rag.demo       # 2. pregunta con respuesta + pregunta trampa

# Una pregunta propia
uv run python -m pipelines.rag.demo "¿Cuándo cambia la rotación de guardias?"

# Volver a indexar (si cambian los documentos o el modelo de embeddings)
uv run python -m pipelines.rag.ingesta --reindexar

# Tests (no usan la red ni descargan modelos)
uv run pytest tests/test_rag.py -v
```

La primera ejecución descarga el modelo de embeddings (unos 470 MB) en la caché de HuggingFace. La ingesta no necesita ninguna API key; la consulta necesita `ANTHROPIC_API_KEY`.

Si la base ya existe, la ingesta no vuelve a indexar:

```
INFO La base ya tiene 8 fragmentos de 4 archivos: no se reindexa (usá --reindexar)
```

### Uso

```python
from pipelines.rag.chain import get_rag_response

respuesta = await get_rag_response("¿Puedo desplegar un hotfix un viernes?")
print(respuesta.respuesta)
print(respuesta.referencias)   # [Referencia(archivo='politica-despliegues.md', fragmento=1), ...]
print(respuesta.encontrada)    # False si la respuesta es "No lo sé."
```

### Las dos pruebas

`demo.py` corre las dos preguntas en paralelo y verifica cada resultado. Salida real:

```
=== 1. Pregunta con respuesta en los documentos ===
Pregunta: ¿Qué condiciones obligan a hacer rollback durante un despliegue canary?
{
  "respuesta": "Hay que hacer rollback inmediato, sin esperar a investigar la causa, si en cualquier etapa se cumple alguna de estas condiciones durante más de 5 minutos: la tasa de errores de la versión nueva supera el 1%, la latencia p99 supera los 800 milisegundos, o la tasa de aprobación de pagos cae más de 2 puntos respecto de la versión anterior. [...]",
  "referencias": [{"archivo": "politica-despliegues.md", "fragmento": 2}],
  "encontrada": true
}

=== 2. Pregunta trampa ===
Pregunta: ¿Qué versión de Kubernetes usa el cluster de producción?
{
  "respuesta": "No lo sé.",
  "referencias": [],
  "encontrada": false
}

=== Verificación ===
PASA: responde con referencias a politica-despliegues.md
PASA: ante la pregunta trampa responde 'No lo sé.'
```

La pregunta trampa es difícil a propósito: `arquitectura.md` dice que los servicios corren en Kubernetes, así que el fragmento recuperado habla del tema, pero en ningún lugar dice qué versión. Un modelo que alucina inventaría una ("1.29"); este responde "No lo sé.".

### Componentes

| Componente | Dónde | Qué hace |
|---|---|---|
| Ingesta | `ingesta.py` → `indexar()` | Lee los `.md` y `.txt` de `data/`, los parte con `RecursiveCharacterTextSplitter` (500 tokens, 50 de solapamiento) y los guarda en ChromaDB con IDs estables (`arquitectura.md#2`). |
| Embeddings | `vectorstore.py` → `crear_embeddings()` | `intfloat/multilingual-e5-small`, local, vía HuggingFace. Es el único lugar donde se elige el modelo: lo usan la ingesta y la consulta. |
| Retriever | `chain.py` → `crear_retriever()` | Convierte la pregunta en un embedding y devuelve los 4 fragmentos más cercanos (distancia coseno). |
| Generación grounded | `chain.py` → `crear_cadena()` | Cadena LCEL: retriever → contexto formateado → prompt → `ChatAnthropic` → `PydanticOutputParser` → validación de referencias. |
| Contrato de salida | `schemas.py` → `RespuestaRAG` | `respuesta` (texto o "No lo sé."), `referencias` (archivo + fragmento) y `encontrada` (calculado). |
| Función asíncrona | `chain.py` → `get_rag_response(query)` | Ejecuta la cadena con `.ainvoke()` y traduce cualquier falla a una excepción propia. |

### Decisiones de diseño

**Embeddings locales.** Anthropic no ofrece embeddings, así que el LLM y los embeddings son de proveedores distintos. Se eligió un modelo local para no depender de otra API key ni pagar por cada fragmento indexado. Lo que importa es que indexar y consultar usen el mismo modelo, no que sea del mismo proveedor que el LLM.

**El modelo de embeddings queda registrado en la base.** La colección guarda en su metadata el nombre del modelo con el que se indexó. Antes de consultar, `abrir_para_consulta` lo compara con el actual y, si no coinciden, falla con `EmbeddingsNoCoincidenError` en lugar de devolver resultados al azar.

**Un modelo de embeddings que acepte 512 tokens.** Muchos modelos populares (`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`) cortan en 128 o 256 tokens: con fragmentos de 500 tokens, el final de cada fragmento quedaría afuera del embedding sin ningún aviso. `multilingual-e5-small` acepta 512.

**Los tokens se cuentan con el tokenizer del modelo de embeddings.** "500 tokens" depende de quién cuente: el mismo texto da cantidades distintas con el tokenizer de OpenAI y con el de este modelo. Se mide con el del modelo de embeddings, que es el que tiene el límite.

**Los 500 tokens se verifican sobre el fragmento ya unido.** El splitter decide el tamaño sumando los tokens de cada pedazo por separado, pero los tokens no se suman exactamente: el texto unido puede quedar unos tokens por encima. Después de partir, `partir_documento` cuenta cada fragmento entero y, si alguno se pasa, vuelve a partir ese documento con un tamaño objetivo menor.

**Indexación atómica.** La ingesta arma la base nueva en una colección temporal y recién al final reemplaza a la anterior. Si algo falla a mitad de camino (sin conexión para bajar el modelo, Ctrl-C), la base anterior queda intacta y nunca queda una base a medio indexar que una corrida posterior tome por completa.

**Rutas ancladas al repositorio.** `data/` y `vectorstore/` se resuelven desde la raíz del repo, no desde la carpeta actual. Para consultar, la base se abre en solo lectura y sin crear la carpeta: si no existe, el error lo dice, en vez de crear una base vacía en otro lugar.

**Chunking por estructura.** El splitter prueba los separadores de Markdown en orden (títulos, párrafos, líneas, palabras) y solo baja de nivel si el pedazo no entra. Con los documentos de ejemplo, cada fragmento termina siendo una o varias secciones enteras. El solapamiento de 50 tokens solo aparece cuando un corte cae dentro de un párrafo: si el corte cae entre secciones, no hay nada que repetir.

**`PydanticOutputParser` con el `stop_reason` revisado antes.** El parser genera las instrucciones de formato del prompt a partir de `RespuestaRAG` y después valida el JSON. Antes de parsear, `verificar_fin` descarta respuestas cortadas por el límite de tokens o negativas del modelo.

**Las referencias se verifican, no se confían.** La cadena conserva los documentos recuperados hasta el final, y `validar_referencias` comprueba que cada fragmento citado esté entre ellos. Una respuesta sin referencias, o con una referencia a un fragmento que no se recuperó, se rechaza con `RespuestaInvalidaError`: es la forma típica de una alucinación.

**El docstring de la clase no viaja al prompt.** Pydantic usa el docstring de cada clase como `description` del esquema, y ese esquema va dentro de las instrucciones de formato. Los docstrings están escritos para quien lee el código, así que se quitan del esquema con `json_schema_extra`. Las `description` de los campos sí están escritas para el modelo y se conservan.

**"No lo sé" no es un error.** Es una respuesta válida con `encontrada=False`. En cambio, una base vacía o mal indexada sí es un error (`BaseNoIndexadaError`): sin documentos, todas las preguntas terminarían en "No lo sé." y el problema real quedaría oculto.

**Nada bloquea el event loop.** Cargar el modelo y abrir la base son operaciones bloqueantes y se hacen con `asyncio.to_thread`. La búsqueda del retriever, en su versión async, LangChain la corre en un hilo aparte.

**Errores propios.** Además de `ValueError` para una pregunta vacía, `get_rag_response` solo lanza subclases de `RAGError`:

| Excepción | Cuándo |
|---|---|
| `RAGConfigError` | Falta `ANTHROPIC_API_KEY` u otra configuración es inválida. Las tres siguientes son subclases |
| `BaseNoIndexadaError` | La base no existe o está vacía: hay que correr la ingesta |
| `EmbeddingsNoCoincidenError` | La base se indexó con otro modelo de embeddings: hay que reindexar |
| `RecursosNoDisponiblesError` | No se pudo cargar el modelo de embeddings (por ejemplo, la primera vez sin conexión) o abrir la base |
| `ModeloNegativaError` | El modelo se negó a responder |
| `RespuestaInvalidaError` | JSON mal formado, respuesta cortada, sin referencias o con referencias inventadas |
| `RAGProveedorError` | Falló la llamada a Anthropic después de los reintentos del SDK. Guarda la `LLMError` en `.error` |
