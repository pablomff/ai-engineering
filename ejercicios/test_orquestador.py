"""Tests del orquestador concurrente.

Ejecutar desde la raíz: ``uv run pytest ejercicios -v``

En los tests no se esperan segundos reales: ``monkeypatch`` achica las
latencias 10 veces, manteniendo las proporciones entre modelos.
"""

import ast
import logging
import pathlib
import time

import pytest

import orquestador as orq


@pytest.fixture
def latencias_rapidas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reemplaza LATENCIAS solo durante el test; monkeypatch lo restaura al terminar."""
    monkeypatch.setattr(
        orq, "LATENCIAS", {"gpt-4": 0.10, "claude-3": 0.15, "local-llama": 0.30}
    )


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Captura los logs INFO del orquestador para inspeccionarlos en el test."""
    caplog.set_level(logging.INFO, logger=orq.__name__)
    return caplog


# --- Concurrencia -----------------------------------------------------------


@pytest.mark.usefixtures("latencias_rapidas")
async def test_las_llamadas_corren_en_paralelo() -> None:
    """En paralelo tarda lo que el más lento (~0.30s), no la suma (0.55s)."""
    t0 = time.perf_counter()
    resultados = await orq.orquestar("hola", timeout=1.0)
    duracion = time.perf_counter() - t0

    assert duracion < 0.45, f"parece secuencial: tardó {duracion:.2f}s"
    assert None not in resultados.values()


@pytest.mark.usefixtures("latencias_rapidas")
async def test_las_tareas_inician_casi_al_mismo_tiempo(
    logs: pytest.LogCaptureFixture,
) -> None:
    """Los tres logs de inicio deben estar a menos de 50 ms entre sí."""
    await orq.orquestar("hola", timeout=1.0)

    inicios = [r.created for r in logs.records if r.getMessage().endswith("inicia")]
    assert len(inicios) == 3
    assert max(inicios) - min(inicios) < 0.05


# --- Timeout ----------------------------------------------------------------


@pytest.mark.usefixtures("latencias_rapidas")
async def test_timeout_conserva_las_respuestas_que_llegaron() -> None:
    """Con timeout de 0.2s llegan gpt-4 (0.10) y claude-3 (0.15), pero no local-llama."""
    resultados = await orq.orquestar("hola", timeout=0.2)

    assert resultados["gpt-4"] is not None
    assert resultados["claude-3"] is not None
    assert resultados["local-llama"] is None


@pytest.mark.usefixtures("latencias_rapidas")
async def test_timeout_se_loguea_como_error(logs: pytest.LogCaptureFixture) -> None:
    """El TimeoutError se registra con nivel ERROR y nombra al modelo que faltó."""
    await orq.orquestar("hola", timeout=0.2)

    errores = [r for r in logs.records if r.levelno == logging.ERROR]
    assert len(errores) == 1
    assert "TimeoutError" in errores[0].getMessage()
    assert "local-llama" in errores[0].getMessage()


@pytest.mark.usefixtures("latencias_rapidas")
async def test_sin_timeout_no_hay_errores(logs: pytest.LogCaptureFixture) -> None:
    """Si el tiempo alcanza, responden todos y no se loguea ningún error."""
    resultados = await orq.orquestar("hola", timeout=1.0)

    assert None not in resultados.values()
    assert not [r for r in logs.records if r.levelno == logging.ERROR]


# --- Semáforo ---------------------------------------------------------------


@pytest.mark.parametrize("limite", [1, 2, 3])
async def test_semaforo_limita_la_concurrencia(limite: int) -> None:
    """El pico de llamadas activas nunca supera el límite del semáforo."""
    pico = await orq.demo_semaforo(n=10, limite=limite, latencia=0.02)
    assert pico == limite


async def test_semaforo_ejecuta_por_tandas() -> None:
    """10 llamadas de 0.05s de a 2 son 5 tandas: ~0.25s en total."""
    t0 = time.perf_counter()
    await orq.demo_semaforo(n=10, limite=2, latencia=0.05)
    duracion = time.perf_counter() - t0

    assert 0.24 <= duracion < 0.40


# --- Criterio de aceptación: nada de time.sleep -----------------------------


def test_no_usa_time_sleep() -> None:
    """Revisa el código fuente (AST) y falla si encuentra una llamada a time.sleep."""
    arbol = ast.parse(pathlib.Path(orq.__file__).read_text(encoding="utf-8"))
    usos = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Attribute)
        and nodo.attr == "sleep"
        and isinstance(nodo.value, ast.Name)
        and nodo.value.id == "time"
    ]
    assert not usos, "se encontró time.sleep: bloquea el event loop"
