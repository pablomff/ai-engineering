"""Evaluación del recuperador con un golden set: Recall@5 y Precision@5.

Uso::

    uv run python -m pipelines.rag_hibrido.evaluate
    uv run python -m pipelines.rag_hibrido.evaluate --golden otro_golden_set.json

El golden set (``golden_set.json``) es una lista de preguntas para las que
sabemos de antemano qué documento tiene la respuesta::

    {"pregunta": "¿Cómo activo HTTP/2 en un cliente?", "documento_id_esperado": "http2.md"}

Para cada pregunta se recuperan 5 fragmentos y se mide:

- **Recall@5**: ¿hay al menos un fragmento del documento esperado entre los
  5? Vale 1 o 0 por pregunta; el promedio es el porcentaje de preguntas
  donde el recuperador encontró el documento correcto.
- **Precision@5**: de los 5 fragmentos, ¿qué proporción es del documento
  esperado? Mide cuánto ruido acompaña a lo útil.

"Útil" se aproxima como "viene del documento esperado". Es una
simplificación: un fragmento de otro archivo podría ayudar igual, y uno del
archivo correcto podría hablar de otra cosa. Para un benchmark chico y
reproducible es un criterio objetivo, que no depende de juzgar a mano.

Además del sistema híbrido, se evalúan por separado sus dos componentes
(solo BM25 y solo semántico) para ver qué aporta la combinación.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pinecone import PineconeError
from pydantic import BaseModel, TypeAdapter, ValidationError

from pipelines.rag_hibrido.config import ConfigRAG
from pipelines.rag_hibrido.exceptions import PineconeNoDisponibleError, RAGHibridoError
from pipelines.rag_hibrido.indice import configurar_logs
from pipelines.rag_hibrido.retriever import TOP_K, BM25Retriever, RAGSystem

GOLDEN_SET_POR_DEFECTO = Path(__file__).with_name("golden_set.json")


class CasoEvaluacion(BaseModel):
    """Una pregunta del golden set con el documento que debería encontrarse."""

    pregunta: str
    documento_id_esperado: str


class ResultadoCaso(BaseModel):
    """Métricas de una pregunta para una forma de búsqueda."""

    pregunta: str
    esperado: str
    fuentes: list[str]
    """Fuente de cada uno de los fragmentos recuperados, en orden."""
    recall: float
    precision: float
    techo_precision: float
    """La Precision@5 máxima posible: si el documento tiene 2 fragmentos, 2/5."""

    @property
    def posicion(self) -> int | None:
        """Posición (desde 1) del primer fragmento del documento esperado, o ``None``."""
        return next((i for i, f in enumerate(self.fuentes, start=1) if f == self.esperado), None)


class Resumen(BaseModel):
    """Promedios de una forma de búsqueda sobre todo el golden set."""

    nombre: str
    recall: float
    precision: float
    casos: list[ResultadoCaso]


def recall_at_k(fuentes: Sequence[str], esperado: str, k: int = TOP_K) -> float:
    """1.0 si algún fragmento entre los ``k`` primeros viene de ``esperado``; si no, 0.0."""
    return 1.0 if esperado in fuentes[:k] else 0.0


def precision_at_k(fuentes: Sequence[str], esperado: str, k: int = TOP_K) -> float:
    """Proporción de los ``k`` primeros fragmentos que vienen de ``esperado``.

    Se divide siempre por ``k``, aunque se hayan recuperado menos: un hueco
    cuenta como un resultado no útil.
    """
    return sum(1 for fuente in fuentes[:k] if fuente == esperado) / k


def techo_precision(fragmentos_del_documento: int, k: int = TOP_K) -> float:
    """Precision@k máxima alcanzable para un documento con esa cantidad de fragmentos.

    Si el documento esperado tiene 2 fragmentos, ni el recuperador perfecto
    puede llenar los 5 lugares con fragmentos útiles: el máximo es 2/5.
    Sin este dato, una Precision@5 de 0,4 parece mala cuando es perfecta.
    """
    return min(fragmentos_del_documento, k) / k


def cargar_golden_set(ruta: Path) -> list[CasoEvaluacion]:
    """Lee y valida el golden set.

    Raises:
        OSError: si no se puede leer el archivo.
        pydantic.ValidationError: si el JSON no tiene la forma esperada.
        ValueError: si el golden set está vacío. Sin preguntas no hay
            promedio que calcular, y el reporte dividiría por cero.
    """
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    golden_set = TypeAdapter(list[CasoEvaluacion]).validate_python(contenido)
    if not golden_set:
        raise ValueError(f"El golden set {ruta} no tiene preguntas")
    return golden_set


def fragmentos_por_fuente(corpus: Sequence[Document]) -> Counter[str]:
    """Cuenta cuántos fragmentos tiene cada documento del corpus."""
    return Counter(str(doc.metadata["fuente"]) for doc in corpus)


def evaluar_caso(
    caso: CasoEvaluacion, recuperados: Sequence[Document], conteo: Counter[str]
) -> ResultadoCaso:
    """Calcula las métricas de una pregunta a partir de los fragmentos recuperados."""
    fuentes = [str(doc.metadata["fuente"]) for doc in recuperados[:TOP_K]]
    return ResultadoCaso(
        pregunta=caso.pregunta,
        esperado=caso.documento_id_esperado,
        fuentes=fuentes,
        recall=recall_at_k(fuentes, caso.documento_id_esperado),
        precision=precision_at_k(fuentes, caso.documento_id_esperado),
        techo_precision=techo_precision(conteo[caso.documento_id_esperado]),
    )


def resumir(nombre: str, casos: Sequence[ResultadoCaso]) -> Resumen:
    """Promedia las métricas de todas las preguntas.

    Raises:
        ValueError: si no hay casos (no hay promedio posible).
    """
    if not casos:
        raise ValueError("No hay casos para resumir")
    return Resumen(
        nombre=nombre,
        recall=sum(c.recall for c in casos) / len(casos),
        precision=sum(c.precision for c in casos) / len(casos),
        casos=list(casos),
    )


async def _top_k(retriever: BaseRetriever, pregunta: str) -> list[Document]:
    """Los ``TOP_K`` primeros resultados de un retriever suelto (BM25 o semántico).

    Traduce los errores de Pinecone igual que ``RAGSystem.recuperar``: el
    retriever semántico suelto no pasa por ese método.

    Raises:
        PineconeNoDisponibleError: si falla Pinecone tras los reintentos.
    """
    try:
        return (await retriever.ainvoke(pregunta))[:TOP_K]
    except PineconeError as exc:
        raise PineconeNoDisponibleError(
            f"Falló la búsqueda en Pinecone ({type(exc).__name__})"
        ) from exc


async def evaluar(rag: RAGSystem, golden_set: Sequence[CasoEvaluacion]) -> list[Resumen]:
    """Evalúa el sistema híbrido y cada uno de sus dos componentes.

    Las preguntas van una por una, no en paralelo: son pocas, y en orden
    los logs quedan legibles.
    """
    if not isinstance(rag.lexico, BM25Retriever):
        raise TypeError("La evaluación necesita el corpus de BM25 para calcular el techo")
    conteo = fragmentos_por_fuente(rag.lexico.docs)
    formas = {
        "Híbrido (BM25 + semántico)": rag.recuperar,
        "Solo BM25": lambda p: _top_k(rag.lexico, p),
        "Solo semántico": lambda p: _top_k(rag.semantico, p),
    }
    resumenes: list[Resumen] = []
    for nombre, buscar in formas.items():
        casos = [evaluar_caso(c, await buscar(c.pregunta), conteo) for c in golden_set]
        resumenes.append(resumir(nombre, casos))
    return resumenes


def imprimir_reporte(resumenes: Sequence[Resumen]) -> None:
    """Imprime el detalle del sistema híbrido y la comparación de las tres formas."""
    hibrido = resumenes[0]
    print(f"\n=== Evaluación: {hibrido.nombre}, top-{TOP_K} ===\n")
    for numero, caso in enumerate(hibrido.casos, start=1):
        posicion = f"puesto {caso.posicion}" if caso.posicion else "no aparece"
        print(f"{numero}. {caso.pregunta}")
        print(f"   esperado: {caso.esperado} ({posicion})")
        print(
            f"   Recall@{TOP_K}={caso.recall:.0f}  Precision@{TOP_K}={caso.precision:.2f}"
            f"  (máximo posible {caso.techo_precision:.2f})"
        )
        print(f"   recuperados: {', '.join(caso.fuentes)}\n")

    techo = sum(c.techo_precision for c in hibrido.casos) / len(hibrido.casos)
    print(f"=== Comparación ({len(hibrido.casos)} preguntas) ===\n")
    print(f"{'Forma de búsqueda':<28}{'Recall@' + str(TOP_K):>10}{'Precision@' + str(TOP_K):>14}")
    for resumen in resumenes:
        print(f"{resumen.nombre:<28}{resumen.recall:>10.2f}{resumen.precision:>14.2f}")
    print(f"{'Máximo posible':<28}{1:>10.2f}{techo:>14.2f}")


async def _main_async(ruta_golden: Path) -> int:
    """Carga todo, evalúa e imprime el reporte."""
    try:
        golden_set = cargar_golden_set(ruta_golden)
        rag = await RAGSystem.desde_config(ConfigRAG.from_env())
        resumenes = await evaluar(rag, golden_set)
    except ValidationError as exc:
        print(f"Configuración o golden set inválido:\n{exc}", file=sys.stderr)
        return 1
    except (RAGHibridoError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    imprimir_reporte(resumenes)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evalúa el recuperador con Recall@5 y Precision@5.")
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_POR_DEFECTO)
    args = parser.parse_args(argv)
    load_dotenv()
    configurar_logs()
    return asyncio.run(_main_async(args.golden))


if __name__ == "__main__":
    sys.exit(main())
