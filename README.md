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
│       └── langchain_anthropic.py   # ChatAnthropic de LangChain para los pipelines LCEL
├── data/                            # Dataset de ejemplo del RAG (.md y .txt de una empresa ficticia)
│   └── httpx-docs/                  # Dataset del RAG híbrido: documentación de httpx (Markdown) y RFC de HTTP (PDF, en rfc/)
├── pipelines/                       # Pipelines LCEL, uno por subpaquete
│   ├── extraccion/                  # Extracción de entidades técnicas validada
│   │   ├── schemas.py               # Modelo Pydantic ExtraccionTecnica (contrato de salida)
│   │   ├── chain.py                 # Cadena LCEL, reintentos y process_text()
│   │   └── demo.py                  # Mini-script de prueba asíncrono
│   ├── rag/                         # RAG local con ChromaDB
│   │   ├── vectorstore.py           # Modelo de embeddings y base ChromaDB (compartidos)
│   │   ├── ingesta.py               # Script de carga: lectura, chunking y persistencia
│   │   ├── schemas.py               # RespuestaRAG: texto + referencias
│   │   ├── chain.py                 # Retriever, cadena LCEL y get_rag_response()
│   │   ├── exceptions.py            # Errores propios del pipeline
│   │   └── demo.py                  # Pregunta con respuesta + pregunta trampa
│   └── rag_hibrido/                 # RAG híbrido (BM25 + vectores) sobre Pinecone
│       ├── config.py                # Variables de entorno validadas (PINECONE_API_KEY, INDEX_NAME...)
│       ├── embeddings.py            # Embeddings calculados por Pinecone (multilingual-e5-large, 1024 dims)
│       ├── indice.py                # Setup: crea el índice serverless y verifica dimensión y modelo
│       ├── ingesta.py               # Carga, chunking, embeddings y upsert por lotes con metadata
│       ├── retriever.py             # RAGSystem: EnsembleRetriever de BM25 + Pinecone, top-5
│       ├── evaluate.py              # Recall@5 y Precision@5 sobre el golden set
│       ├── golden_set.json          # 5 preguntas con su documento esperado
│       └── exceptions.py            # Errores propios del pipeline
├── ejercicios/                      # Scripts independientes, cada uno con su propio docstring
│   ├── orquestador.py               # asyncio: gather, timeout y Semaphore (solo librería estándar)
│   ├── test_orquestador.py          # Tests: uv run pytest ejercicios
│   ├── lcel_chain.py                # Primera cadena LCEL: prompt | modelo | StrOutputParser
│   ├── pinecone_setup.py            # Pinecone: índice serverless, namespace y query
│   └── ingestion_pipeline.py        # Ingesta por lotes con un índice simulado
└── tests/
    ├── test_schemas.py              # Validaciones de los schemas
    ├── test_manager.py              # Contrato de errores del manager, con un cliente falso
    ├── test_error_mapping.py        # Traducción de excepciones de los SDKs a LLMError
    ├── test_extraccion.py           # Esquema, validación de la salida y cadena del pipeline de extracción
    ├── test_rag.py                  # Ingesta, base vectorial, contrato y cadena del RAG
    └── test_rag_hibrido.py          # Índice, ingesta, recuperador y métricas del RAG híbrido
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
| `PINECONE_API_KEY` | Para el RAG híbrido | — | API key de Pinecone |
| `INDEX_NAME` | Para el RAG híbrido | — | Nombre del índice serverless (minúsculas, números y guiones) |
| `PINECONE_NAMESPACE` | No | `httpx-docs` | Namespace del índice donde vive la documentación |

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

## Pipeline de extracción validada

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

Usa Anthropic a través de `ChatAnthropic`, configurado con los mismos `Settings` del cliente unificado (`llm_client`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `LLM_TIMEOUT_SECONDS` y `LLM_MAX_RETRIES`. Solo hace falta `ANTHROPIC_API_KEY`: el pipeline siempre usa Anthropic, así que ignora `LLM_PROVIDER`. El modelo por defecto, `claude-opus-5-5`, es Claude Opus 5.5, el modelo actual de Anthropic.

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

**Dos niveles de reintento.** Los errores de red y del proveedor (timeout, 429, 5xx) los reintenta el SDK, como en el cliente unificado. `.with_retry()` solo reintenta `ExtraccionInvalidaError`, para no sumar reintentos encima de los del SDK. Con `include_raw=True`, LangChain no lanza los errores de parseo: los devuelve en `parsing_error` y `validar_salida` los convierte en `ExtraccionInvalidaError`.

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
| `ExtraccionProveedorError` | Falló la llamada a Anthropic después de los reintentos del SDK. Guarda la `LLMError` del cliente unificado en `.error`, con su `kind` y `retryable` |

Cualquier otra excepción es un bug y se propaga sin envolver.

**El SDK queda dentro de `llm_client/providers/`.** `ChatAnthropic` se crea en `providers/langchain_anthropic.py`, que también traduce las excepciones del SDK a `LLMError` reutilizando la traducción de `AnthropicClient`. `pipelines/extraccion` no importa nada de `anthropic`.

**Una sola cadena por proceso.** Si no se le pasa una cadena, `process_text` usa una que crea la primera vez y reutiliza después, así todas las llamadas comparten el cliente HTTP y su pool de conexiones.

**Logs sin datos sensibles.** Se loguean el largo del texto, la cantidad de tecnologías, la criticidad, el `stop_reason` y los nombres de los campos que fallaron, pero nunca el texto ni los valores recibidos.

## Sistema RAG local

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

## RAG híbrido en la nube con Pinecone

El código está en `pipelines/rag_hibrido/` y el dataset en `data/httpx-docs/`.

Es un **módulo de recuperación**: recibe una consulta y devuelve los 5 fragmentos más relevantes, combinando búsqueda léxica (BM25) y búsqueda semántica (vectores en Pinecone Serverless). Pinecone hace las dos cosas del lado de la nube: calcula los embeddings con su Inference API (`multilingual-e5-large`) y guarda y busca los vectores, con la misma `PINECONE_API_KEY`. No genera respuestas: lo que se mide es la calidad de lo que recupera, con Recall@5 y Precision@5.

El dataset combina dos formatos, 25 archivos en total (el origen y las licencias están en `data/httpx-docs/FUENTE.txt`):

- **Markdown:** la documentación oficial de [httpx](https://github.com/encode/httpx), 23 archivos con licencia BSD-3, fijada a un commit. Es documentación técnica real y está llena de nombres propios (`AsyncClient`, `MockTransport`, `HTTPStatusError`), justo el caso en que la búsqueda léxica complementa a la semántica.
- **PDF:** las especificaciones de los dos protocolos que implementa httpx, tal como las publica la RFC Editor: el [RFC 9112](https://www.rfc-editor.org/rfc/rfc9112.pdf) (HTTP/1.1, 46 páginas) y el [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113.pdf) (HTTP/2, 78 páginas). En los PDF, cada fragmento guarda la página donde empieza.

### Cómo replicar el índice

1. **Cuenta y keys.** Creá una cuenta en [Pinecone](https://www.pinecone.io/) (el plan gratuito alcanza) y copiá la API key.
2. **Variables de entorno.** En el `.env` (partiendo de `.env.example`):

   ```
   PINECONE_API_KEY=...            # la key de Pinecone
   ANTHROPIC_API_KEY=...           # la key del LLM del proyecto
   INDEX_NAME=httpx-docs-e5-large  # minúsculas, números y guiones
   ```

3. **Dependencias.** `uv sync`. Los embeddings los calcula Pinecone, así que no se descarga ningún modelo: solo el tokenizer (unos pocos MB) para medir los fragmentos.
4. **Crear el índice.** Crea un índice serverless (AWS `us-east-1`, 1024 dimensiones, coseno) si no existe; si existe, verifica que sea compatible:

   ```bash
   uv run python -m pipelines.rag_hibrido.indice
   ```

5. **Cargar los documentos.** Fragmenta, calcula los embeddings y sube 274 fragmentos (113 de Markdown y 161 de los RFC) al namespace `httpx-docs`. También crea el índice si hiciera falta, así que el paso 4 es opcional:

   ```bash
   uv run python -m pipelines.rag_hibrido.ingesta
   uv run python -m pipelines.rag_hibrido.ingesta --reindexar   # vuelve a subir todo (si cambió el texto de un archivo)
   ```

6. **Consultar y evaluar.**

   ```bash
   uv run python -m pipelines.rag_hibrido.retriever "¿Cómo uso MockTransport en los tests?"
   uv run python -m pipelines.rag_hibrido.evaluate
   ```

Los tests no usan la red ni Pinecone: `uv run pytest tests/test_rag_hibrido.py -v`.

### Uso

```python
from pipelines.rag_hibrido.config import ConfigRAG
from pipelines.rag_hibrido.retriever import RAGSystem

rag = await RAGSystem.desde_config(ConfigRAG.from_env())
for doc in await rag.recuperar("¿Cómo activo HTTP/2 en un cliente?"):
    print(doc.metadata["id"], doc.metadata["seccion"])   # http2.md#1 HTTP/2 ...
    print(doc.page_content)                               # el texto del fragmento
```

### Evaluación

`golden_set.json` tiene 5 preguntas en español con el documento que tiene la respuesta. Se escribieron antes de correr la evaluación y no se ajustaron después. Para cada pregunta:

- **Recall@5**: 1 si algún fragmento del documento esperado está entre los 5 recuperados, 0 si no.
- **Precision@5**: proporción de los 5 fragmentos que vienen del documento esperado. "Útil" se aproxima como "viene del documento esperado".

`evaluate.py` evalúa el sistema híbrido y, por separado, cada una de sus dos búsquedas. Resultado real:

```
Forma de búsqueda             Recall@5   Precision@5
Híbrido (BM25 + semántico)        0.80          0.40
Solo BM25                         0.40          0.16
Solo semántico                    1.00          0.56
Máximo posible                    1.00          0.64
```

Cómo leerlo:

- **La búsqueda semántica sola es la mejor** con este golden set: encuentra el documento correcto en las 5 preguntas, y en 4 lo pone primero. Para la pregunta de `MockTransport`, sus 5 resultados son de `transports.md`.
- **El híbrido encuentra el documento correcto en 4 de las 5 preguntas**, y en esas 4 lo pone primero.
- **"Máximo posible" no es 1.00 en precisión.** Tres de los documentos esperados tienen solo 2 fragmentos: aunque el recuperador fuera perfecto, 3 de los 5 lugares serían de otros documentos. El techo promedio es 0.64, no 1.
- **BM25 solo es el más débil** porque las preguntas están en español y la documentación en inglés: solo coinciden los términos técnicos. Acierta las preguntas de `MockTransport` y de `requests`, y falla las otras tres. La de HTTP/2 la acertaba antes de sumar los RFC: ahora hay 106 fragmentos del RFC 9113 que dicen "HTTP/2" en cada párrafo, y `http2.md` queda tapado. La búsqueda semántica, con un modelo multilingüe, cruza el idioma y distingue "cómo activarlo en httpx" de "cómo funciona el protocolo".
- **Con estas preguntas, la fusión perjudica.** En la pregunta de timeouts, `advanced/timeouts.md` estaba 3.º y 4.º en la búsqueda semántica pero no aparecía entre los 10 candidatos de BM25. RRF premia a los fragmentos que aparecen en las dos listas, aunque sea abajo, y `timeouts.md` quedó fuera del top-5. En ninguna de las 5 preguntas el híbrido supera a la búsqueda semántica sola.
- **Qué pasó al cambiar de modelo.** Con el modelo local `multilingual-e5-small` (384 dimensiones) el híbrido sacaba Recall 1.00 y Precision 0.40, y la búsqueda semántica sola, 1.00 y 0.44. Con `multilingual-e5-large` de Pinecone, la búsqueda semántica mejoró mucho (0.44 → 0.56) y BM25 quedó igual. Una búsqueda semántica más precisa deja menos lugar para que BM25 aporte, y más para que meta ruido. Sumar los RFC (de 113 a 274 fragmentos) no movió a la búsqueda semántica, bajó a BM25 solo (Recall 0.60 → 0.40) y dejó al híbrido casi igual (Precision 0.36 → 0.40).
- **Los pesos quedaron en 0.5/0.5 a propósito.** Bajar el peso de BM25 (por ejemplo a 0.3/0.7) tiene un argumento que no depende del benchmark: con preguntas en otro idioma que los documentos, BM25 es débil por diseño. Pero elegir el número mirando estas mismas 5 preguntas sería sobreajustar el sistema al examen. El paso siguiente es ampliar el golden set, idealmente con preguntas reales, y ajustar los pesos sobre ese set más grande.
- **La métrica a veces castiga respuestas buenas.** En la pregunta de timeouts el híbrido devolvió dos fragmentos de `exceptions.md`, que lista `ConnectTimeout`, la respuesta. Cuentan como inútiles porque no son del documento esperado.
- **El golden set sigue siendo de 5 preguntas, todas sobre httpx,** como pide la consigna. Para los RFC, la búsqueda se probó con una consulta aparte: "¿Cuál es el tamaño inicial de la ventana de control de flujo en HTTP/2?" trae 5 fragmentos del RFC 9113, y el 5.º, en la página 42, contiene la respuesta (65.535 octetos).

### Experimento: cortar los RFC por sección

El texto que sale de un PDF no tiene títulos de Markdown, así que el splitter corta los RFC por largo: casi todos los fragmentos quedan entre 480 y 500 tokens y cruzan de una sección a otra. Eso explica por qué a veces la `seccion` de un fragmento es la vecina. Se probó una alternativa antes de decidir si cambiar la ingesta.

**Cómo se probó.** Se comparó la ingesta actual (A) con una variante (B) que antepone `## ` a cada título de sección detectado en los RFC, para que el splitter corte ahí. Todo lo demás era igual: mismo splitter, mismos embeddings y los mismos 23 Markdown. B se subió a otro namespace del mismo índice, con un script aparte que no forma parte del repo. Además del golden set, se usaron 5 preguntas sobre los RFC, escritas antes de correr nada. Para cada una se fijó una frase del RFC que contiene la respuesta y aparece una sola vez (por ejemplo, "window size of 65,535 octets"). Acertar quiere decir que algún fragmento del top-5 contiene esa frase.

**Fragmentos de los RFC**

| | A: por largo (actual) | B: por sección |
|---|---|---|
| Fragmentos | 161 | 210 |
| Tokens promedio | 488 | 360 |
| De menos de 200 tokens | 1 | 41 |
| Empiezan en un título | 4 de 161 | 127 de 210 |

**Resultados**

| | A | B |
|---|---|---|
| Preguntas sobre los RFC, híbrido (Recall@5) | 0.80 | 0.80 |
| Preguntas sobre los RFC, solo semántico (Recall@5) | 0.60 | 1.00 |
| Sección correcta en el fragmento con la respuesta (híbrido) | 1 de 4 | 4 de 4 |
| Golden set, híbrido (Recall@5 / Precision@5) | 0.80 / 0.40 | 0.80 / 0.32 |
| Golden set, solo semántico (Recall@5 / Precision@5) | 1.00 / 0.56 | 1.00 / 0.56 |

**Qué se decidió y por qué.** Cortar por sección arregla la metadata: la sección del fragmento con la respuesta pasa de acertar 1 de 4 veces a acertar siempre. También ayuda a la búsqueda semántica sola, que encuentra las 5 respuestas de los RFC en lugar de 3. Pero el sistema que se evalúa es el híbrido, y ahí no mejora: empata en los RFC y la precisión del golden set baja de 0.40 a 0.32 (2 fragmentos útiles menos sobre 25), probablemente porque los 49 fragmentos extra de los RFC compiten con la documentación de httpx. Además, la página, que es el metadato que pide la consigna, ya era correcta en las dos variantes.

Por eso **la ingesta se quedó con el corte por largo**. Con 5 preguntas por grupo, una diferencia de uno o dos puestos está cerca del ruido, así que la conclusión firme es solo la de la metadata. El cambio conviene retomarlo si el sistema pasa a generar respuestas que citen la sección, o si se usa solo la búsqueda semántica. En ese caso, conviene juntar antes las secciones cortas (las 41 de menos de 200 tokens) y medir con un golden set más grande.

### Componentes

| Componente | Dónde | Qué hace |
|---|---|---|
| Configuración | `config.py` → `ConfigRAG` | Lee y valida `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`, `INDEX_NAME` y `PINECONE_NAMESPACE`. Las keys son `SecretStr`. |
| Setup | `indice.py` → `asegurar_indice()` | Crea el índice serverless si no existe y verifica dimensión, métrica y modelo de embeddings (guardado como tag del índice). |
| Embeddings | `embeddings.py` → `PineconeEmbeddings` | Adaptador entre la Inference API de Pinecone (`multilingual-e5-large`, 1024 dimensiones) y la interfaz `Embeddings` de LangChain. Distingue preguntas (`query`) de fragmentos (`passage`). |
| Ingesta | `ingesta.py` → `indexar()` | Lee Markdown y PDF (`pypdf`), limpia los PDF, los parte con `RecursiveCharacterTextSplitter` (el del RAG local, 500 tokens), descarta fragmentos sin texto, calcula embeddings y hace upsert en lotes de 100. |
| Metadata | `ingesta.py` → `armar_metadata()` | `texto`, `fuente`, `pagina` (en los PDF), `seccion`, `categoria` (`rfc`/`avanzado`/`general`), `fragmento` y `tokens`. |
| Búsqueda semántica | `retriever.py` → `PineconeRetriever` | Retriever de LangChain sobre el SDK nativo de Pinecone. |
| Búsqueda léxica | `retriever.py` → `crear_bm25()` | `BM25Retriever` sobre el corpus leído de Pinecone, con un tokenizador que normaliza mayúsculas y puntuación. |
| Recuperador híbrido | `retriever.py` → `RAGSystem` | `EnsembleRetriever` con pesos 0.5/0.5 y Reciprocal Rank Fusion. 10 candidatos por búsqueda, devuelve el top-5. |
| Evaluación | `evaluate.py` | Recall@5, Precision@5 y el techo de precisión, para el híbrido y cada búsqueda por separado. |

### Decisiones de diseño

**Embeddings de Pinecone, de 1024 dimensiones.** Anthropic no ofrece embeddings y no hay key de OpenAI (la dimensión 1536 de la consigna es la de `text-embedding-3-small`). Se usa la Inference API de Pinecone con `multilingual-e5-large`: no suma otra key ni otra dependencia externa, porque el sistema ya depende de Pinecone, y ninguna máquina tiene que descargar ni cargar un modelo. Es multilingüe (preguntas en español, documentación en inglés) y usa el mismo tokenizer que el modelo del RAG local.

**La dimensión se mide, no se escribe.** Se pide el embedding de un texto de prueba y se cuenta su largo. El setup frena si el índice existente tiene otra dimensión: el error típico de subir vectores a un índice creado para otro modelo. Por eso el índice actual se llama `httpx-docs-e5-large`: el anterior, de 384 dimensiones, no puede cambiar.

**El texto nunca se corta en silencio.** El modelo acepta hasta 507 tokens por texto. Los fragmentos tienen como máximo 500, medidos con su tokenizer, y los embeddings se piden con `truncate="NONE"`: si alguno se pasara, Pinecone devuelve un error en vez de cortar el final.

**El modelo queda registrado en el índice.** Dos modelos distintos pueden tener la misma dimensión. Por eso el índice se crea con el tag `modelo_embeddings` y el setup y el recuperador lo comparan con el modelo actual.

**El texto va en la metadata.** Un vector solo no se puede leer. Guardando el texto junto al vector, una consulta a Pinecone devuelve todo lo necesario, sin una segunda consulta a otra base. Pinecone permite hasta 40 KB de metadata por vector; un fragmento de 500 tokens ocupa unos 2 KB.

**Página y sección.** Cada fragmento guarda el título de la sección donde empieza; los de PDF guardan además la página. En Markdown no hay páginas, y Pinecone no acepta valores nulos en la metadata, así que ahí el campo `pagina` no se incluye. Las secciones de Markdown salen de los títulos `#`, ignorando los comentarios de los bloques de código; las de los RFC, de líneas como "5.2. Flow Control".

**Un PDF es un solo texto, no un documento por página.** Así un párrafo que cruza de una página a la siguiente no queda partido en dos fragmentos. Para no perder la página, se guarda dónde empieza cada una y cada fragmento toma la página donde empieza.

**Los PDF se limpian antes de fragmentar.** Se quitan el encabezado y el pie que el RFC repite en cada página ("RFC 9113 HTTP/2 June 2022", "Standards Track Page 20") y el índice de contenidos, que aparecía en casi cualquier búsqueda porque nombra todos los temas. Y se normaliza el texto (NFKC): el PDF usa ligaduras como "ﬁ", y sin normalizar BM25 no encontraría "specified".

**La posición de cada fragmento se recalcula.** El splitter de LangChain calcula dónde empieza cada fragmento restando el solapamiento como si fueran caracteres, pero el nuestro se mide en tokens. Cuando dos fragmentos se solapan, deja la posición en -1, y con -1 la página y la sección salían mal (le pasaba a la mitad de los fragmentos de los PDF y a 7 de Markdown). `recalcular_inicios()` busca cada fragmento a partir del anterior.

**Límite conocido: el orden de extracción del PDF.** `pypdf` no siempre extrae el texto en el orden visual de la página. En la página 42 del RFC 9113 entrega el final de la sección 6.9.3 antes que la 6.9.2, así que un fragmento puede quedar etiquetado con la sección vecina. La página siempre es correcta.

**Chunks de 500 tokens.** La consigna sugiere entre 500 y 800, y el modelo de embeddings acepta hasta 507 (contando el prefijo y los tokens especiales que agrega Pinecone). Se usa el máximo que entra.

**Namespace propio.** La documentación vive en el namespace `httpx-docs`. El mismo índice puede guardar otros conjuntos de documentos sin que se mezclen en las búsquedas.

**SDK nativo en vez de `PineconeVectorStore`.** `langchain-pinecone` exige `pinecone<8` (el proyecto usa la 10) e instala `langchain-openai`. `PineconeRetriever` hereda de `BaseRetriever`, lo que alcanza para meterlo en el `EnsembleRetriever`.

**El corpus de BM25 sale de Pinecone.** `RAGSystem` lista los ids del namespace y trae su metadata. Así las dos búsquedas trabajan sobre exactamente los mismos fragmentos e ids, y el `EnsembleRetriever` puede detectar cuándo el mismo fragmento vino de las dos.

**`BM25Retriever` viene de `langchain-community`,** que se está discontinuando. Es el único paquete que lo tiene y la consigna lo pide; el aviso de deprecación se silencia solo en ese import.

**Reintentos del SDK.** El cliente de Pinecone se crea con `RetryConfig` (espera exponencial ante 429 y 5xx) y timeout. Un lote de upsert que falla igual no lanza excepción, así que se revisa la respuesta y se lanza `IngestaIncompletaError`.

**La ingesta no repite trabajo, y se repara sola.** Antes de subir, fragmenta los documentos para saber cuántos vectores debería haber. Si el namespace tiene exactamente esa cantidad, no hace nada. Si tiene otra (por ejemplo, una ingesta anterior falló y dejó solo algunos lotes), vuelve a subir todo para completarlo: alcanza con correrla de nuevo. `--reindexar` fuerza la subida aunque la cantidad coincida, para cuando cambió el texto de un archivo.

**Reemplazar sin dejar el namespace vacío.** No se usa `delete_all`. Primero se suben todos los vectores (los ids son estables, así que los existentes se reemplazan) y después se borran solo los ids viejos que ya no corresponden a ningún fragmento. En Pinecone serverless las escrituras se aplican de forma asíncrona: un borrado total aplicado tarde podría llevarse vectores recién subidos. Como los ids que se borran nunca son los que se subieron, el orden deja de importar.

**Consistencia eventual.** Las estadísticas y el listado de ids de Pinecone tardan unos segundos en reflejar una escritura. `RAGSystem` verifica que el corpus de BM25 no venga vacío y avisa si su tamaño no coincide con las estadísticas.

**`ANTHROPIC_API_KEY` se valida pero no se usa.** La consigna la pide en el `.env`, y esta entrega solo recupera. Queda lista para el paso de generación.

**Errores propios.** Todos heredan de `RAGHibridoError`:

| Excepción | Cuándo |
|---|---|
| `IndiceIncompatibleError` | El índice existe con otra dimensión, métrica o modelo de embeddings |
| `IndiceInexistenteError` | No hay ningún índice con ese `INDEX_NAME` |
| `IndiceNoListoError` | El índice existe pero Pinecone no lo marca como listo |
| `NamespaceVacioError` | No se corrió la ingesta, o Pinecone todavía no muestra los vectores recién subidos |
| `IngestaIncompletaError` | Algún lote del upsert no se cargó ni con los reintentos. Volver a correr la ingesta completa el namespace |
| `PineconeNoDisponibleError` | Falló Pinecone después de los reintentos. Guarda la excepción original como causa |
