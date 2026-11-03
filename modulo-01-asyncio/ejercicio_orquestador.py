"""Ejercicio del módulo 1: orquestador concurrente de modelos.

Ejecutar desde la raíz del curso:
    uv run python modulo-01-asyncio/ejercicio_orquestador.py

Criterios de aceptación:
  - No usar time.sleep (bloqueante).
  - En los logs se ve que las tareas inician casi al mismo tiempo.
  - Se maneja TimeoutError sin que el programa se detenga.
"""

import asyncio
import time

INICIO = time.perf_counter()


def log(msg: str) -> None:
    """Imprime con los segundos transcurridos desde el arranque, para ver la concurrencia."""
    print(f"[{time.perf_counter() - INICIO:5.2f}s] {msg}")


# Paso 2: tres corrutinas que simulan modelos con distinta latencia.
async def gpt_4_call(prompt: str) -> str:
    # TODO: log de inicio, await asyncio.sleep(...), log de fin y return
    ...


async def claude_3_call(prompt: str) -> str:
    # TODO
    ...


async def local_llama_call(prompt: str) -> str:
    # TODO
    ...


# Pasos 3 y 4: disparar las tres a la vez con asyncio.gather,
# con un asyncio.timeout(2.0) sobre la ejecución total y captura de TimeoutError.
async def orquestar(prompt: str) -> None:
    # TODO
    ...


# Paso 5: 10 llamadas simuladas, pero un asyncio.Semaphore(2) deja correr solo 2 a la vez.
async def demo_semaforo() -> None:
    # TODO
    ...


async def main() -> None:
    await orquestar("Explicá qué es asyncio en una frase")
    await demo_semaforo()


if __name__ == "__main__":
    asyncio.run(main())
