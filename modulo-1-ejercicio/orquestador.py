"""Orquestador concurrente de modelos (ejercicio del Módulo 1).

Simula un backend que consulta varias fuentes de IA a la vez. No llama a
ninguna API real: cada "modelo" es una corrutina que espera con
``asyncio.sleep`` para imitar la latencia de red. Es independiente del resto
del proyecto (solo usa la librería estándar).

Qué demuestra:

- **Concurrencia**: ``asyncio.gather`` dispara las tres llamadas a la vez, así
  que el total tarda lo que la más lenta y no la suma de todas.
- **Resiliencia**: ``asyncio.timeout`` corta la ejecución a los 2 segundos; el
  ``TimeoutError`` se captura y el programa sigue, conservando las respuestas
  que sí llegaron.
- **Control de flujo**: un ``asyncio.Semaphore(2)`` deja que de 10 llamadas
  solo 2 corran al mismo tiempo (como un rate limit).

Uso::

    uv run python modulo-1-ejercicio/orquestador.py

Tests::

    uv run pytest modulo-1-ejercicio
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Latencia simulada de cada modelo, en segundos. Vive en un dict de módulo para
# que los tests puedan achicarla con monkeypatch y no esperar segundos reales.
LATENCIAS: dict[str, float] = {
    "gpt-4": 1.0,
    "claude-3": 1.5,
    "local-llama": 3.0,  # supera el timeout de 2s a propósito
}

TIMEOUT_TOTAL: float = 2.0
"""Tiempo máximo, en segundos, para la consulta conjunta a los tres modelos."""


async def _simular_modelo(nombre: str, prompt: str) -> str:
    """Simula una llamada de red al modelo ``nombre`` y devuelve su respuesta.

    Centraliza la lógica común de los tres modelos: loguear el inicio, esperar
    sin bloquear el event loop y loguear el fin.
    """
    logger.info("→ %s inicia", nombre)
    # asyncio.sleep cede el control al event loop: mientras este modelo espera,
    # los demás avanzan. time.sleep congelaría todo el programa.
    await asyncio.sleep(LATENCIAS[nombre])
    logger.info("← %s termina", nombre)
    return f"[{nombre}] respuesta a: {prompt!r}"


async def gpt_4_call(prompt: str) -> str:
    """Simula una consulta a GPT-4 (latencia media)."""
    return await _simular_modelo("gpt-4", prompt)


async def claude_3_call(prompt: str) -> str:
    """Simula una consulta a Claude 3 (latencia media-alta)."""
    return await _simular_modelo("claude-3", prompt)


async def local_llama_call(prompt: str) -> str:
    """Simula una consulta a un Llama local (la más lenta)."""
    return await _simular_modelo("local-llama", prompt)


async def orquestar(prompt: str, timeout: float = TIMEOUT_TOTAL) -> dict[str, str | None]:
    """Consulta los tres modelos en paralelo con un límite de tiempo total.

    Si se supera ``timeout``, registra el error y sigue: devuelve las
    respuestas que llegaron a tiempo y ``None`` para las que no. Así un modelo
    lento no hace perder el trabajo de los rápidos.
    """
    logger.info("Orquestando 3 modelos (timeout total %.1fs)", timeout)

    # Se crean tareas explícitas (no corrutinas sueltas) para poder revisar,
    # tras un timeout, cuáles terminaron y cuáles fueron canceladas.
    tareas: dict[str, asyncio.Task[str]] = {
        "gpt-4": asyncio.create_task(gpt_4_call(prompt)),
        "claude-3": asyncio.create_task(claude_3_call(prompt)),
        "local-llama": asyncio.create_task(local_llama_call(prompt)),
    }

    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*tareas.values())
    except TimeoutError:
        # Al vencer el timeout, gather cancela las tareas que seguían corriendo;
        # las que ya habían terminado conservan su resultado.
        pendientes = [nombre for nombre, tarea in tareas.items() if tarea.cancelled()]
        logger.error(
            "TimeoutError: se superaron %.1fs. Sin respuesta de: %s",
            timeout,
            ", ".join(pendientes),
        )

    resultados: dict[str, str | None] = {
        nombre: tarea.result() if tarea.done() and not tarea.cancelled() else None
        for nombre, tarea in tareas.items()
    }
    for nombre, respuesta in resultados.items():
        logger.info("  %s: %s", nombre, respuesta or "sin respuesta (timeout)")
    return resultados


async def demo_semaforo(n: int = 10, limite: int = 2, latencia: float = 0.5) -> int:
    """Lanza ``n`` llamadas a la vez, pero solo ``limite`` corren en simultáneo.

    Devuelve el pico de concurrencia observado, que debería ser igual a
    ``limite``. Sirve para respetar el rate limit de una API sin dejar de
    disparar todo el trabajo de una vez.
    """
    logger.info("Lanzando %d llamadas con Semaphore(%d)", n, limite)
    semaforo = asyncio.Semaphore(limite)
    activas = 0
    pico = 0

    async def llamada(i: int) -> None:
        """Una llamada simulada que espera un permiso del semáforo antes de correr."""
        nonlocal activas, pico
        logger.info("  llamada %02d en cola", i)
        # Si no hay permisos libres, la tarea espera acá sin bloquear el loop.
        # Al salir del bloque, el permiso se libera aunque haya una excepción.
        async with semaforo:
            activas += 1
            pico = max(pico, activas)
            logger.info("  llamada %02d EJECUTANDO (activas=%d)", i, activas)
            await asyncio.sleep(latencia)
            activas -= 1

    await asyncio.gather(*(llamada(i) for i in range(1, n + 1)))
    logger.info("Pico de concurrencia observado: %d", pico)
    return pico


async def main() -> None:
    """Corre la demo completa: orquestación con timeout y luego el semáforo."""
    await orquestar("Explicá qué es asyncio en una frase")
    await demo_semaforo()


if __name__ == "__main__":
    # relativeCreated: milisegundos desde que arrancó el programa. Permite ver
    # en los logs que las tareas inician casi al mismo tiempo.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(relativeCreated)6.0f ms] %(levelname)-5s %(message)s",
    )
    asyncio.run(main())
