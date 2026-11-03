"""Ejemplo resuelto del módulo 1: batch de embeddings con control de concurrencia.

Ejecutar desde la raíz del curso:
    uv run python modulo-01-asyncio/ejemplo_embeddings.py
"""

import asyncio
import random
import time


# --- 1. Simulamos una llamada a la API de embeddings (tarea I/O-bound) ---
async def get_embedding(text: str, sem: asyncio.Semaphore) -> dict:
    async with sem:  # respeta el límite de concurrencia
        latency = random.uniform(0.5, 2.5)  # latencia de red simulada
        try:
            async with asyncio.timeout(2.0):  # corta si tarda más de 2s
                await asyncio.sleep(latency)  # await, NO time.sleep (bloquearía el loop)
                return {"text": text, "dim": 1536, "ok": True}
        except TimeoutError:
            return {"text": text, "ok": False, "error": "timeout"}


# --- 2. Orquestador: dispara todas las tareas concurrentemente ---
async def embed_batch(texts: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(3)  # máximo 3 llamadas en paralelo
    tasks = [get_embedding(t, sem) for t in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)


# --- 3. Punto de entrada ---
async def main():
    textos = [f"documento_{i}" for i in range(8)]
    inicio = time.perf_counter()
    resultados = await embed_batch(textos)
    elapsed = time.perf_counter() - inicio

    ok = [r for r in resultados if isinstance(r, dict) and r.get("ok")]
    fallidos = [r for r in resultados if isinstance(r, dict) and not r.get("ok")]

    print(f"Procesados OK: {len(ok)}/{len(textos)} en {elapsed:.2f}s")
    for f in fallidos:
        print(f"  Falló: {f['text']} -> {f['error']}")


if __name__ == "__main__":
    asyncio.run(main())
