"""Tests del orquestador. Ejecutar desde la raíz: uv run pytest -v

Idea clave: en los tests NO queremos esperar segundos reales. Por eso achicamos
las latencias con `monkeypatch` (los tiempos quedan proporcionales, pero 10x más rápidos).
"""

import ast
import pathlib
import time

import pytest

import ejercicio_orquestador as orq


@pytest.fixture
def latencias_rapidas(monkeypatch):
    """Reemplaza LATENCIAS solo durante el test; monkeypatch lo restaura al terminar."""
    monkeypatch.setattr(
        orq, "LATENCIAS", {"gpt-4": 0.10, "claude-3": 0.15, "local-llama": 0.30}
    )


# --- Concurrencia -----------------------------------------------------------

async def test_las_llamadas_corren_en_paralelo(latencias_rapidas):
    # Si fueran secuenciales tardaría 0.10 + 0.15 + 0.30 = 0.55s.
    # En paralelo tarda lo que el más lento: ~0.30s.
    t0 = time.perf_counter()
    resultados = await orq.orquestar("hola", timeout=1.0)
    duracion = time.perf_counter() - t0

    assert duracion < 0.45, f"parece secuencial: tardó {duracion:.2f}s"
    assert all(r is not None for r in resultados.values())


async def test_las_tareas_inician_casi_al_mismo_tiempo(latencias_rapidas, capsys):
    # capsys captura lo que se imprimió, así validamos los logs.
    await orq.orquestar("hola", timeout=1.0)
    salida = capsys.readouterr().out

    inicios = [
        float(linea[1:6])  # "[ 0.12s] → gpt-4 inicia" -> 0.12
        for linea in salida.splitlines()
        if "inicia" in linea
    ]
    assert len(inicios) == 3
    assert max(inicios) - min(inicios) < 0.05


# --- Timeout ----------------------------------------------------------------

async def test_timeout_no_rompe_y_conserva_las_respuestas_que_llegaron(latencias_rapidas):
    # Timeout de 0.2s: gpt-4 (0.10) y claude-3 (0.15) llegan, local-llama (0.30) no.
    resultados = await orq.orquestar("hola", timeout=0.2)

    assert resultados["gpt-4"] is not None
    assert resultados["claude-3"] is not None
    assert resultados["local-llama"] is None


async def test_timeout_loguea_el_error(latencias_rapidas, capsys):
    await orq.orquestar("hola", timeout=0.2)
    salida = capsys.readouterr().out
    assert "TimeoutError" in salida
    assert "local-llama" in salida


async def test_todos_responden_si_el_timeout_alcanza(latencias_rapidas, capsys):
    resultados = await orq.orquestar("hola", timeout=1.0)
    assert None not in resultados.values()
    assert "TimeoutError" not in capsys.readouterr().out


# --- Semáforo ---------------------------------------------------------------

@pytest.mark.parametrize("limite", [1, 2, 3])
async def test_semaforo_limita_la_concurrencia(limite):
    pico = await orq.demo_semaforo(n=10, limite=limite, latencia=0.02)
    assert pico == limite


async def test_semaforo_tarda_lo_esperado():
    # 10 llamadas de 0.05s de a 2 => 5 "tandas" => ~0.25s.
    t0 = time.perf_counter()
    await orq.demo_semaforo(n=10, limite=2, latencia=0.05)
    duracion = time.perf_counter() - t0
    assert 0.24 <= duracion < 0.40


# --- Criterio: nada de time.sleep ------------------------------------------

def test_no_usa_time_sleep():
    # Analizamos el código fuente (AST) buscando llamadas a time.sleep.
    codigo = pathlib.Path(orq.__file__).read_text()
    arbol = ast.parse(codigo)
    llamadas = [
        n for n in ast.walk(arbol)
        if isinstance(n, ast.Attribute)
        and n.attr == "sleep"
        and isinstance(n.value, ast.Name)
        and n.value.id == "time"
    ]
    assert not llamadas, "se encontró time.sleep: bloquea el event loop"
