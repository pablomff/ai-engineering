"""Ejercicio Módulo 2: refactorización a LCEL asíncrono.

Reemplaza la llamada imperativa al SDK del Módulo 1 por una cadena declarativa
de LangChain: ``ChatPromptTemplate | ChatAnthropic | StrOutputParser``.

Por qué LCEL: cada pieza es un ``Runnable`` intercambiable y el operador ``|``
las compone en un único pipeline que ya trae ``ainvoke``, ``astream`` y
``abatch`` sin escribir lógica de orquestación a mano.

Uso::

    uv run python modulo-2-ejercicio/lcel_chain.py
    uv run python modulo-2-ejercicio/lcel_chain.py "¿Qué es un agujero negro?"

"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

PREGUNTA_POR_DEFECTO = "¿Qué es la entropía?"


def crear_cadena() -> Runnable[dict[str, str], str]:
    """Arma la cadena LCEL ``prompt | modelo | parser``.

    Recibe un diccionario con la clave ``pregunta`` y devuelve la respuesta
    como ``str``. El ``StrOutputParser`` es el que extrae el texto del
    ``AIMessage`` que devuelve el modelo.
    """
    # Roles explícitos System/Human. La variable {pregunta} debe coincidir
    # con la clave del diccionario que se pasa a ainvoke.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Sos un asistente experto. Respondé en español en no más de 3 oraciones."),
            ("human", "{pregunta}"),
        ]
    )

    # La familia Claude 5 rechaza `temperature` (error 400), así que solo se
    # envía si LLM_TEMPERATURE está definida; con None el parámetro se omite.
    temperatura = os.environ.get("LLM_TEMPERATURE")

    # Reintentos y timeout nativos del SDK, como en el Módulo 1.
    modelo = ChatAnthropic(
        model_name=os.environ.get("ANTHROPIC_MODEL") or "claude-opus-5-5",
        temperature=float(temperatura) if temperatura else None,
        timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS") or 30),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 2),
    )

    return prompt | modelo | StrOutputParser()


async def main() -> None:
    """Ejecuta la cadena de forma asíncrona e imprime la respuesta en texto plano."""
    load_dotenv()
    pregunta = sys.argv[1] if len(sys.argv) > 1 else PREGUNTA_POR_DEFECTO

    cadena = crear_cadena()
    respuesta = await cadena.ainvoke({"pregunta": pregunta})

    print(f"Pregunta: {pregunta}\n")
    print(respuesta)


if __name__ == "__main__":
    asyncio.run(main())
