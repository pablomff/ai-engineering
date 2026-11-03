"""Ejercicio del módulo 1: orquestador concurrente de modelos.

Ejecutar desde la raíz del curso:
    uv run python modulo-01-asyncio/ejercicio_orquestador.py

Tests:
    uv run pytest -v

Criterios de aceptación:
  - No usar time.sleep (bloqueante).
  - En los logs se ve que las tareas inician casi al mismo tiempo.
  - Se maneja TimeoutError sin que el programa se detenga.
"""

import asyncio
import time

INICIO = time.perf_counter()

# Latencia simulada de cada "modelo", en segundos. Está en un dict a nivel de módulo
# para que los tests puedan cambiarla (monkeypatch) sin tocar el código.
LATENCIAS: dict[str, float] = {
    "gpt-4": 1.0,
    "claude-3": 1.5,
    "local-llama": 3.0,  # más lento que el timeout de 2s: va a fallar a propósito
}

TIMEOUT_TOTAL = 2.0


def log(msg: str) -> None:
    """Imprime con los segundos transcurridos desde el arranque, para ver la concurrencia."""
    print(f"[{time.perf_counter() - INICIO:5.2f}s] {msg}")


async def _simular_modelo(nombre: str, prompt: str) -> str:
    """Lógica común: loguear, 'esperar la red' sin bloquear, y responder."""
    log(f"→ {nombre} inicia")
    # asyncio.sleep cede el control al event loop: mientras este modelo "espera",
    # los demás siguen avanzando. time.sleep congelaría todo el programa.
    await asyncio.sleep(LATENCIAS[nombre])
    log(f"← {nombre} termina")
    return f"[{nombre}] respuesta a: {prompt!r}"


# Paso 2: tres corrutinas que simulan modelos con distinta latencia.
async def gpt_4_call(prompt: str) -> str:
    return await _simular_modelo("gpt-4", prompt)


async def claude_3_call(prompt: str) -> str:
    return await _simular_modelo("claude-3", prompt)


async def local_llama_call(prompt: str) -> str:
    return await _simular_modelo("local-llama", prompt)


# Pasos 3 y 4: disparar las tres a la vez con asyncio.gather,
# con un asyncio.timeout(2.0) sobre la ejecución total y captura de TimeoutError.
async def orquestar(prompt: str, timeout: float = TIMEOUT_TOTAL) -> dict[str, str | None]:
    """Consulta los tres modelos en paralelo.

    Devuelve {modelo: respuesta}; los que no llegaron a tiempo quedan en None.
    """
    log(f"Orquestando 3 modelos (timeout total {timeout}s)")

    # create_task agenda cada corrutina en el event loop YA (empiezan casi a la vez).
    # Guardamos las tareas para poder revisar cuáles terminaron si hay timeout.
    tareas = {
        "gpt-4": asyncio.create_task(gpt_4_call(prompt)),
        "claude-3": asyncio.create_task(claude_3_call(prompt)),
        "local-llama": asyncio.create_task(local_llama_call(prompt)),
    }

    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*tareas.values())
    except TimeoutError:
        # Al vencer el timeout, gather cancela las tareas que seguían corriendo.
        # Las que ya habían terminado conservan su resultado.
        pendientes = [n for n, t in tareas.items() if t.cancelled()]
        log(f"⚠ TimeoutError: superados {timeout}s. Sin respuesta de: {', '.join(pendientes)}")

    resultados = {
        nombre: (t.result() if t.done() and not t.cancelled() else None)
        for nombre, t in tareas.items()
    }
    for nombre, resp in resultados.items():
        log(f"  {nombre}: {resp or 'sin respuesta (timeout)'}")
    return resultados


# Paso 5: 10 llamadas simuladas, pero un asyncio.Semaphore(2) deja correr solo 2 a la vez.
async def demo_semaforo(n: int = 10, limite: int = 2, latencia: float = 0.5) -> int:
    """Lanza `n` llamadas a la vez, pero solo `limite` pueden estar activas juntas.

    Devuelve el pico de concurrencia observado (debería ser == limite).
    """
    log(f"Lanzando {n} llamadas con Semaphore({limite})")
    semaforo = asyncio.Semaphore(limite)
    activas = 0
    pico = 0

    async def llamada(i: int) -> None:
        nonlocal activas, pico
        log(f"  llamada {i:02d} en cola")
        # El semáforo es un contador de "permisos". Si no hay permisos libres,
        # la tarea espera aquí (sin bloquear el loop) hasta que otra lo libere.
        async with semaforo:
            activas += 1
            pico = max(pico, activas)
            log(f"  llamada {i:02d} EJECUTANDO (activas={activas})")
            await asyncio.sleep(latencia)
            activas -= 1
        # Al salir del `async with` el permiso se libera automáticamente.

    await asyncio.gather(*(llamada(i) for i in range(1, n + 1)))
    log(f"Semáforo OK: pico de concurrencia = {pico}")
    return pico


async def main() -> None:
    await orquestar("Explicá qué es asyncio en una frase")
    print()
    await demo_semaforo()


if __name__ == "__main__":
    asyncio.run(main())
