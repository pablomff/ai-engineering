"""Mini-script de prueba del RAG: una pregunta con respuesta y una pregunta trampa.

Uso::

    uv run python -m pipelines.rag.ingesta      # una sola vez, para armar la base
    uv run python -m pipelines.rag.demo         # las dos pruebas
    uv run python -m pipelines.rag.demo "¿Cuándo cambia la rotación de guardias?"

Las dos pruebas se corren en paralelo y cada una se verifica:

1. **Pregunta con respuesta**: los criterios de rollback están en
   ``politica-despliegues.md``. Pasa si la respuesta salió de los documentos
   y cita ese archivo.
2. **Pregunta trampa**: los documentos dicen que la plataforma corre en
   Kubernetes, pero nunca qué versión. Es una trampa a propósito: el contexto
   recuperado habla del tema, así que un modelo que alucina inventaría una
   versión. Pasa si responde "No lo sé." sin referencias.

El JSON sale por la salida estándar y los logs por la de errores.

Código de salida: 0 si las dos pruebas pasan (o, con una pregunta propia, si
hubo respuesta), 1 si no.
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv
from langchain_core.runnables import Runnable

from pipelines.rag.chain import cadena_por_defecto, get_rag_response
from pipelines.rag.exceptions import RAGConfigError, RAGError
from pipelines.rag.schemas import RespuestaRAG

PREGUNTA_CON_RESPUESTA = "¿Qué condiciones obligan a hacer rollback durante un despliegue canary?"
ARCHIVO_ESPERADO = "politica-despliegues.md"
PREGUNTA_TRAMPA = "¿Qué versión de Kubernetes usa el cluster de producción?"

Cadena = Runnable[str, RespuestaRAG]


def mostrar(titulo: str, pregunta: str, resultado: RespuestaRAG | BaseException) -> None:
    """Imprime la pregunta y la respuesta en JSON, o el error."""
    print(f"\n=== {titulo} ===\nPregunta: {pregunta}")
    if isinstance(resultado, RespuestaRAG):
        print(resultado.model_dump_json(indent=2))
    else:
        print(f"ERROR ({type(resultado).__name__}): {resultado}")


def verificar_con_respuesta(resultado: RespuestaRAG | BaseException) -> bool:
    """La respuesta tiene que salir de los documentos y citar el archivo esperado."""
    return (
        isinstance(resultado, RespuestaRAG)
        and resultado.encontrada
        and any(r.archivo == ARCHIVO_ESPERADO for r in resultado.referencias)
    )


def verificar_trampa(resultado: RespuestaRAG | BaseException) -> bool:
    """El modelo tiene que decir "No lo sé" en vez de inventar una versión."""
    return isinstance(resultado, RespuestaRAG) and not resultado.encontrada


async def correr_pruebas(cadena: Cadena) -> int:
    """Corre las dos pruebas en paralelo, muestra el veredicto y devuelve el código de salida."""
    con_respuesta, trampa = await asyncio.gather(
        get_rag_response(PREGUNTA_CON_RESPUESTA, cadena),
        get_rag_response(PREGUNTA_TRAMPA, cadena),
        return_exceptions=True,
    )
    mostrar("1. Pregunta con respuesta en los documentos", PREGUNTA_CON_RESPUESTA, con_respuesta)
    mostrar("2. Pregunta trampa", PREGUNTA_TRAMPA, trampa)

    for resultado in (con_respuesta, trampa):
        if isinstance(resultado, BaseException) and not isinstance(resultado, RAGError):
            # Un error que no es del pipeline es un bug: se relanza con su traceback.
            raise resultado

    ok_respuesta = verificar_con_respuesta(con_respuesta)
    ok_trampa = verificar_trampa(trampa)
    print("\n=== Verificación ===")
    print(f"{'PASA' if ok_respuesta else 'FALLA'}: responde con referencias a {ARCHIVO_ESPERADO}")
    print(f"{'PASA' if ok_trampa else 'FALLA'}: ante la pregunta trampa responde 'No lo sé.'")
    return 0 if ok_respuesta and ok_trampa else 1


async def correr_pregunta(pregunta: str, cadena: Cadena) -> int:
    """Responde una pregunta propia y devuelve el código de salida."""
    try:
        resultado = await get_rag_response(pregunta, cadena)
    except (RAGError, ValueError) as exc:
        mostrar("Resultado", pregunta, exc)
        return 1
    mostrar("Resultado", pregunta, resultado)
    return 0


async def main() -> int:
    """Corre la pregunta recibida por argumento, o las dos pruebas."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Logs HTTP de los SDKs y de HuggingFace: una línea por request, tapan los del pipeline.
    for ruidoso in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    try:
        # Carga el modelo de embeddings y abre la base: es bloqueante, va a un hilo.
        cadena = await asyncio.to_thread(cadena_por_defecto)
    except RAGConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(sys.argv) > 1:
        return await correr_pregunta(" ".join(sys.argv[1:]), cadena)
    return await correr_pruebas(cadena)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
