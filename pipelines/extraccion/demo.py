"""Mini-script de prueba asíncrono del pipeline de extracción.

Procesa con ``process_text`` el texto que se le pase, o los ejemplos de
demostración si no recibe ninguno.

Uso::

    # Un texto como argumento
    uv run python -m pipelines.extraccion.demo "La API en Django consulta MySQL sin índices"

    # El contenido de un archivo, por la entrada estándar
    cat error.log | uv run python -m pipelines.extraccion.demo

    # Sin texto: corre los dos ejemplos de demostración
    uv run python -m pipelines.extraccion.demo

Los ejemplos de demostración son dos y se procesan en paralelo:

1. Una descripción de arquitectura con un problema claro: debe devolver un
   objeto válido.
2. Un texto ambiguo, sin tecnologías identificables (prueba de estrés): el
   modelo devuelve una lista vacía, la validación la rechaza y el pipeline
   termina en ``TextoSinTecnologiasError`` sin gastar reintentos, en vez de
   devolver tecnologías inventadas.

El JSON resultante se imprime por la salida estándar y los logs por la salida
de errores, así que el resultado se puede redirigir a un archivo o a ``jq``
sin mezclarse con los logs.

Código de salida:

- Con un texto propio: 0 si se obtuvo una extracción válida, 1 si no.
- Con los ejemplos: 0 si el de arquitectura se procesó bien, 1 si falló. El
  resultado del texto ambiguo no cuenta, porque fallar ahí es lo esperado.
- 1 si la configuración es inválida.
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv
from langchain_core.runnables import Runnable

from pipelines.extraccion.chain import (
    ExtraccionConfigError,
    ExtraccionError,
    crear_cadena,
    crear_modelo,
    process_text,
)
from pipelines.extraccion.schemas import ExtraccionTecnica

# Textos que se usan solo cuando no se pasa ninguno por argumento ni por la
# entrada estándar.
EJEMPLO_ARQUITECTURA = (
    "Nuestra API está hecha con FastAPI y usa Redis como caché de sesiones. Los datos "
    "persisten en PostgreSQL. Desde el lanzamiento de la campaña, con picos de 2000 "
    "usuarios concurrentes, el pool de conexiones a la base se agota y los requests "
    "tardan más de 10 segundos o devuelven 503."
)
EJEMPLO_AMBIGUO = "Ayer algo andaba medio lento, pero después pareció arreglarse solo."

Cadena = Runnable[dict[str, str], ExtraccionTecnica]


def leer_texto_de_entrada() -> str | None:
    """Devuelve el texto a procesar, o ``None`` si no se pasó ninguno.

    Primero mira los argumentos (todos juntos, por si el texto se pasó sin
    comillas) y después la entrada estándar. La entrada estándar solo se lee
    si no es la terminal: si lo fuera, ``read()`` se quedaría esperando que el
    usuario escriba. Una entrada estándar vacía cuenta como "sin texto", para
    que el script corra los ejemplos en entornos que la conectan a la nada.
    """
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    if not sys.stdin.isatty():
        contenido = sys.stdin.read()
        if contenido.strip():
            return contenido
    return None


def mostrar_resultado(titulo: str, resultado: ExtraccionTecnica | BaseException) -> None:
    """Imprime el JSON validado o el error de un texto."""
    print(f"\n=== {titulo} ===")
    if isinstance(resultado, ExtraccionTecnica):
        print(resultado.model_dump_json(indent=2))
    else:
        print(f"ERROR ({type(resultado).__name__}): {resultado}")


async def procesar_texto_propio(texto: str, cadena: Cadena) -> int:
    """Procesa el texto que pasó el usuario y devuelve el código de salida."""
    try:
        resultado = await process_text(texto, cadena)
    except (ExtraccionError, ValueError) as exc:
        # ValueError: el texto estaba en blanco.
        mostrar_resultado("Resultado", exc)
        return 1
    mostrar_resultado("Resultado", resultado)
    return 0


async def procesar_ejemplos(cadena: Cadena) -> int:
    """Corre los dos ejemplos en paralelo y devuelve el código de salida."""
    # La misma cadena se reutiliza para ambos textos.
    arquitectura, ambiguo = await asyncio.gather(
        process_text(EJEMPLO_ARQUITECTURA, cadena),
        process_text(EJEMPLO_AMBIGUO, cadena),
        return_exceptions=True,
    )

    mostrar_resultado("1. Descripción de arquitectura", arquitectura)
    mostrar_resultado("2. Texto ambiguo (prueba de estrés)", ambiguo)

    if isinstance(arquitectura, BaseException) and not isinstance(arquitectura, ExtraccionError):
        # Un error que no es del pipeline es un bug: se relanza con su traceback.
        raise arquitectura
    return 0 if isinstance(arquitectura, ExtraccionTecnica) else 1


async def main() -> int:
    """Procesa el texto recibido, o los ejemplos, y devuelve el código de salida."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Los logs HTTP del SDK de Anthropic (una línea por request) tapan los
    # del pipeline.
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    texto = leer_texto_de_entrada()

    try:
        cadena = crear_cadena(crear_modelo())
    except ExtraccionConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    if texto is None:
        print("Sin texto de entrada: se procesan los ejemplos de demostración.", file=sys.stderr)
        return await procesar_ejemplos(cadena)
    return await procesar_texto_propio(texto, cadena)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
