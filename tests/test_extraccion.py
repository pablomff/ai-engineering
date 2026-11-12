"""Tests del pipeline de extracción: esquema, validación y cadena LCEL.

No llaman a la API. La salida del modelo se simula con ``ModeloFalso``, que
devuelve diccionarios como los de ``with_structured_output(...,
include_raw=True)``; así se prueba la cadena real, con sus reintentos.
"""

from collections.abc import Iterator

import anthropic
import httpx2
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from llm_client import Settings
from pipelines.extraccion import chain
from pipelines.extraccion.chain import (
    MAX_INTENTOS,
    ExtraccionConfigError,
    ExtraccionInvalidaError,
    ExtraccionProveedorError,
    ModeloNegativaError,
    TextoSinTecnologiasError,
    crear_cadena,
    crear_modelo,
    crear_prompt,
    process_text,
    validar_salida,
)
from pipelines.extraccion.schemas import ERROR_SIN_TECNOLOGIAS, ExtraccionTecnica, NivelCriticidad

VALIDO: dict[str, object] = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "Cuello de botella en conexiones concurrentes.",
}

Salida = dict[str, object]

# Los tests ponen la espera entre reintentos en 0. LangChain la pasa a tenacity
# como "initial", que con un valor distinto de 1 emite un DeprecationWarning
# (con el valor de producción, 1, no aparece).
pytestmark = pytest.mark.filterwarnings(
    "ignore:The 'initial' parameter is deprecated:DeprecationWarning"
)


def salida_modelo(
    parsed: ExtraccionTecnica | None = None,
    parsing_error: BaseException | None = None,
    stop_reason: str = "end_turn",
) -> Salida:
    """Arma una salida como la de ``with_structured_output(include_raw=True)``."""
    raw = AIMessage(content="", response_metadata={"stop_reason": stop_reason})
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}


def error_de_esquema(datos: dict[str, object]) -> OutputParserException:
    """Reproduce cómo el parser de LangChain envuelve una ValidationError."""
    try:
        ExtraccionTecnica.model_validate(datos)
    except ValidationError as exc:
        try:
            raise OutputParserException("Failed to parse") from exc
        except OutputParserException as envuelta:
            return envuelta
    raise AssertionError("los datos deberían ser inválidos")


SALIDA_VALIDA: Salida = salida_modelo(parsed=ExtraccionTecnica.model_validate(VALIDO))
SALIDA_ESQUEMA_INVALIDO: Salida = salida_modelo(
    parsing_error=error_de_esquema({**VALIDO, "nivel_de_criticidad": "critica"})
)
SALIDA_SIN_TECNOLOGIAS: Salida = salida_modelo(
    parsing_error=error_de_esquema({**VALIDO, "tecnologias": []})
)


class ModeloFalso(BaseChatModel):
    """Chat model que devuelve salidas estructuradas predefinidas, en orden.

    Reemplaza a ``ChatAnthropic`` en la cadena real: ``with_structured_output``
    devuelve un runnable que entrega la siguiente salida de ``salidas`` (o la
    última, si se terminaron) y cuenta las llamadas.
    """

    salidas: list[Salida]
    llamadas: int = 0

    @property
    def _llm_type(self) -> str:
        """Identificador del modelo, requerido por ``BaseChatModel``."""
        return "falso"

    def _generate(self, messages: list[BaseMessage], *args: object, **kwargs: object) -> ChatResult:
        """No se usa: la cadena solo llama a ``with_structured_output``."""
        raise NotImplementedError

    def with_structured_output(  # type: ignore[override]
        self, schema: object, *, include_raw: bool = False, **kwargs: object
    ) -> Runnable[object, Salida]:
        """Devuelve un runnable que entrega las salidas predefinidas."""

        def responder(_: object) -> Salida:
            indice = min(self.llamadas, len(self.salidas) - 1)
            self.llamadas += 1
            return self.salidas[indice]

        return RunnableLambda(responder)


@pytest.fixture(autouse=True)
def sin_espera_entre_reintentos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elimina la espera entre reintentos para que los tests sean rápidos."""
    monkeypatch.setattr(chain, "ESPERA_INICIAL_S", 0.0)


@pytest.fixture
def entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Borra las variables del LLM y la cadena cacheada antes y después del test."""
    for variable in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(variable, raising=False)
    chain._cadena_por_defecto.cache_clear()
    yield monkeypatch
    chain._cadena_por_defecto.cache_clear()


# --- Esquema ---------------------------------------------------------------


def test_esquema_valido() -> None:
    extraccion = ExtraccionTecnica.model_validate(VALIDO)
    assert extraccion.nivel_de_criticidad is NivelCriticidad.ALTA
    assert extraccion.model_dump(mode="json") == VALIDO


def test_tecnologias_se_limpian_y_deduplican() -> None:
    extraccion = ExtraccionTecnica.model_validate(
        {**VALIDO, "tecnologias": [" Redis ", "redis", "", "PostgreSQL"]}
    )
    assert extraccion.tecnologias == ["Redis", "PostgreSQL"]


@pytest.mark.parametrize("tecnologias", [[], ["", "  "]])
def test_tecnologias_vacias_tienen_su_propio_tipo_de_error(tecnologias: list[str]) -> None:
    with pytest.raises(ValidationError) as info:
        ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": tecnologias})
    assert [e["type"] for e in info.value.errors()] == [ERROR_SIN_TECNOLOGIAS]


@pytest.mark.parametrize(
    "cambio",
    [
        {"nivel_de_criticidad": "critica"},
        {"resumen_tecnico": "   "},
        {"campo_extra": "x"},
    ],
)
def test_esquema_rechaza_datos_invalidos(cambio: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, **cambio})


def test_esquema_enviado_al_modelo_no_exige_minimo_de_tecnologias() -> None:
    # Si el esquema exigiera minItems, la API forzaría al modelo a inventar
    # una tecnología ante un texto que no menciona ninguna.
    esquema = ExtraccionTecnica.model_json_schema()
    assert "minItems" not in esquema["properties"]["tecnologias"]


# --- validar_salida ----------------------------------------------------------


def test_validar_salida_devuelve_el_objeto() -> None:
    assert validar_salida(SALIDA_VALIDA) is SALIDA_VALIDA["parsed"]


def test_validar_salida_rechaza_respuesta_truncada() -> None:
    # Aunque el objeto haya parseado, una respuesta cortada no es confiable.
    extraccion = ExtraccionTecnica.model_validate(VALIDO)
    with pytest.raises(ExtraccionInvalidaError, match="incompleta"):
        validar_salida(salida_modelo(parsed=extraccion, stop_reason="max_tokens"))


def test_validar_salida_detecta_negativa_del_modelo() -> None:
    error = OutputParserException("Invalid json output")
    with pytest.raises(ModeloNegativaError):
        validar_salida(salida_modelo(parsing_error=error, stop_reason="refusal"))


def test_validar_salida_detecta_texto_sin_tecnologias() -> None:
    with pytest.raises(TextoSinTecnologiasError):
        validar_salida(SALIDA_SIN_TECNOLOGIAS)


def test_validar_salida_lista_vacia_con_otros_errores_es_invalida() -> None:
    # Si además hay otros errores, la salida está mal formada y vale reintentar.
    error = error_de_esquema({**VALIDO, "tecnologias": [], "nivel_de_criticidad": "critica"})
    with pytest.raises(ExtraccionInvalidaError, match="esquema"):
        validar_salida(salida_modelo(parsing_error=error))


def test_validar_salida_rechaza_esquema_invalido() -> None:
    with pytest.raises(ExtraccionInvalidaError, match="esquema"):
        validar_salida(SALIDA_ESQUEMA_INVALIDO)


def test_validar_salida_rechaza_json_mal_formado() -> None:
    error = OutputParserException("Invalid json output")
    with pytest.raises(ExtraccionInvalidaError, match="JSON"):
        validar_salida(salida_modelo(parsing_error=error))


def test_validar_salida_rechaza_salida_sin_objeto() -> None:
    with pytest.raises(ExtraccionInvalidaError):
        validar_salida(salida_modelo())


# --- Prompt y cadena ---------------------------------------------------------


def test_prompt_tiene_roles_y_solo_pide_el_texto() -> None:
    prompt = crear_prompt()
    # instrucciones_formato ya viene fijada con partial.
    assert prompt.input_variables == ["texto"]
    mensajes = prompt.format_messages(texto="Usamos Kafka.")
    assert [m.type for m in mensajes] == ["system", "human"]
    assert "Usamos Kafka." in mensajes[1].content


async def test_cadena_reintenta_y_se_recupera() -> None:
    modelo = ModeloFalso(salidas=[SALIDA_ESQUEMA_INVALIDO, SALIDA_VALIDA])
    resultado = await crear_cadena(modelo).ainvoke({"texto": "Usamos Redis."})
    assert resultado == SALIDA_VALIDA["parsed"]
    assert modelo.llamadas == 2


async def test_cadena_se_detiene_despues_de_max_intentos() -> None:
    modelo = ModeloFalso(salidas=[SALIDA_ESQUEMA_INVALIDO])
    with pytest.raises(ExtraccionInvalidaError):
        await crear_cadena(modelo).ainvoke({"texto": "Usamos Redis."})
    assert modelo.llamadas == MAX_INTENTOS


@pytest.mark.parametrize(
    ("salida", "excepcion"),
    [
        (SALIDA_SIN_TECNOLOGIAS, TextoSinTecnologiasError),
        (salida_modelo(stop_reason="refusal"), ModeloNegativaError),
    ],
)
async def test_cadena_no_reintenta_rechazos_que_se_repetirian(
    salida: Salida, excepcion: type[Exception]
) -> None:
    modelo = ModeloFalso(salidas=[salida, SALIDA_VALIDA])
    with pytest.raises(excepcion):
        await crear_cadena(modelo).ainvoke({"texto": "Ayer algo andaba lento."})
    assert modelo.llamadas == 1


# --- Configuración -----------------------------------------------------------


def test_crear_modelo_sin_api_key_es_error_de_configuracion(
    entorno_limpio: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ExtraccionConfigError, match="ANTHROPIC_API_KEY"):
        crear_modelo()


def test_crear_modelo_no_depende_de_llm_provider(entorno_limpio: pytest.MonkeyPatch) -> None:
    # Con LLM_PROVIDER=openai y sin key de OpenAI, el pipeline igual arranca.
    entorno_limpio.setenv("LLM_PROVIDER", "openai")
    entorno_limpio.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(crear_modelo(), BaseChatModel)


def test_settings_from_env_permite_forzar_el_proveedor(
    entorno_limpio: pytest.MonkeyPatch,
) -> None:
    entorno_limpio.setenv("LLM_PROVIDER", "openai")
    entorno_limpio.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings.from_env(provider="anthropic").provider == "anthropic"


# --- process_text ------------------------------------------------------------


async def test_process_text_devuelve_la_extraccion() -> None:
    modelo = ModeloFalso(salidas=[SALIDA_VALIDA])
    resultado = await process_text("Usamos Redis.", crear_cadena(modelo))
    assert resultado == SALIDA_VALIDA["parsed"]


async def test_process_text_rechaza_texto_vacio() -> None:
    with pytest.raises(ValueError):
        await process_text("   ", RunnableLambda(lambda _: None))


async def test_process_text_envuelve_error_de_configuracion(
    entorno_limpio: pytest.MonkeyPatch,
) -> None:
    # Sin cadena explícita, una configuración inválida no escapa como
    # ValidationError o ValueError.
    with pytest.raises(ExtraccionConfigError):
        await process_text("Usamos Redis.")


def test_cadena_por_defecto_se_crea_una_sola_vez(entorno_limpio: pytest.MonkeyPatch) -> None:
    entorno_limpio.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert chain._cadena_por_defecto() is chain._cadena_por_defecto()


async def test_process_text_traduce_errores_de_anthropic() -> None:
    def falsa(_: dict[str, str]) -> ExtraccionTecnica:
        respuesta = httpx2.Response(
            429, request=httpx2.Request("POST", "https://api.example.com")
        )
        raise anthropic.RateLimitError("x", response=respuesta, body=None)

    with pytest.raises(ExtraccionProveedorError) as info:
        await process_text("texto", RunnableLambda(falsa))
    assert info.value.error.kind == "rate_limit"
    assert info.value.error.retryable is True


async def test_process_text_no_oculta_bugs() -> None:
    def falsa(_: dict[str, str]) -> ExtraccionTecnica:
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        await process_text("texto", RunnableLambda(falsa))
