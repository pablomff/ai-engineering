"""Script de validación del cliente LLM unificado.

Prueba el cliente LLM unificado contra el proveedor configurado, en dos modos:

1. Normal: pide la respuesta completa con ``generate``.
2. Streaming: imprime los fragmentos a medida que llegan y mide el TTFT.

Uso::

    uv run python main.py                      # usa LLM_PROVIDER del .env
    uv run python main.py --provider openai    # fuerza el proveedor

Código de salida: 0 si ambos modos funcionaron, 1 si alguno falló o si la
configuración es inválida.

Este es el único lugar donde se carga el ``.env`` y se configura el logging:
el paquete ``llm_client`` no tiene efectos secundarios al importarse.
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, ChatMessage, ErrorInfo, LLMError, Settings

PREGUNTA = "¿Qué es la entropía?"
SYSTEM = "Respondé en español en no más de 3 oraciones."


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Validación del cliente LLM unificado.")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="Proveedor a usar. Si se pasa, tiene prioridad sobre LLM_PROVIDER.",
    )
    return parser.parse_args()


def describir_error_config(exc: ValidationError) -> str:
    """Convierte un error de validación de ``Settings`` en un mensaje legible.

    No incluye los valores recibidos, para no mostrar API keys.
    """
    mensajes: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        # Pydantic antepone "Value error, " a los errores de los validadores.
        mensaje = error["msg"].removeprefix("Value error, ")
        campo = ".".join(str(p) for p in error["loc"])
        mensajes.append(f"{campo}: {mensaje}" if campo else mensaje)
    return "; ".join(mensajes)


def mostrar_error(error: ErrorInfo) -> None:
    """Muestra un ``ErrorInfo`` de forma legible."""
    reintentable = "sí" if error.retryable else "no"
    print(f"  ERROR [{error.kind}] ({error.provider}): {error.message}")
    print(f"  Reintentable: {reintentable}")


def crear_manager(provider: str | None) -> AsyncLLMManager | None:
    """Crea el manager desde el entorno, o devuelve ``None`` si la config es inválida.

    El ``--provider`` se aplica sobre ``LLM_PROVIDER`` antes de leer la
    configuración, así la validación de la API key usa el proveedor correcto.
    """
    if provider is not None:
        os.environ["LLM_PROVIDER"] = provider
    try:
        return AsyncLLMManager(Settings.from_env())
    except ValidationError as exc:
        print(f"Configuración inválida: {describir_error_config(exc)}", file=sys.stderr)
    except ValueError as exc:
        # Errores de configuración de los clientes (por ejemplo, falta OPENAI_MODEL).
        print(f"Configuración inválida: {exc}", file=sys.stderr)
    return None


async def modo_normal(llm: AsyncLLMManager, messages: list[ChatMessage]) -> bool:
    """Prueba ``generate``. Devuelve ``True`` si la respuesta fue exitosa."""
    print("\n=== 1. Modo normal ===")
    respuesta = await llm.generate(messages)
    if not respuesta.ok:
        # generate garantiza que error no es None cuando ok es False.
        assert respuesta.error is not None
        mostrar_error(respuesta.error)
        return False

    print(respuesta.content)
    print(f"\n  Modelo:        {respuesta.model}")
    print(f"  Finish reason: {respuesta.finish_reason}")
    if respuesta.usage is not None:
        print(
            f"  Tokens:        {respuesta.usage.input_tokens} entrada / "
            f"{respuesta.usage.output_tokens} salida"
        )
    else:
        print("  Tokens:        no informados")
    print(f"  Latencia:      {respuesta.latency_ms:.0f} ms")
    return True


async def modo_streaming(llm: AsyncLLMManager, messages: list[ChatMessage]) -> bool:
    """Prueba ``stream`` midiendo TTFT y tiempo total. Devuelve ``True`` si terminó bien."""
    print("\n=== 2. Modo streaming ===")
    inicio = time.perf_counter()
    ttft_ms: float | None = None
    try:
        async for fragmento in llm.stream(messages):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - inicio) * 1000
            print(fragmento, end="", flush=True)
    except LLMError as exc:
        # Si el corte llegó a mitad de la respuesta, se separa del texto parcial.
        if ttft_ms is not None:
            print()
        mostrar_error(exc.to_error_info())
        return False
    total_ms = (time.perf_counter() - inicio) * 1000

    print()
    ttft = f"{ttft_ms:.0f} ms" if ttft_ms is not None else "sin fragmentos"
    print(f"\n  TTFT:          {ttft}")
    print(f"  Tiempo total:  {total_ms:.0f} ms")
    return True


async def main() -> int:
    """Corre ambos modos y devuelve el código de salida."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    manager = crear_manager(args.provider)
    if manager is None:
        return 1

    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content=PREGUNTA),
    ]
    async with manager as llm:
        print(f"Proveedor: {llm.provider}")
        ok_normal = await modo_normal(llm, messages)
        ok_stream = await modo_streaming(llm, messages)

    return 0 if ok_normal and ok_stream else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
