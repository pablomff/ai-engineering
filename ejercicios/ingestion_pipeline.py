"""Pipeline de ingesta por lotes con metadatos y búsqueda filtrada por categoría.

Trabaja contra ``MockPineconeIndex``, un índice simulado con la misma forma
que el de Pinecone (``upsert`` y ``query`` asíncronos), así el ejercicio corre
sin cuenta ni red. Los embeddings también son simulados.

Uso::

    uv run python ejercicios/ingestion_pipeline.py
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import TypedDict

# Tamaño del embedding simulado. Uno real (p. ej. OpenAI) tendría 1536.
DIMENSION_MOCK = 8


class Documento(TypedDict):
    """Documento de entrada: texto más los datos con los que se arma la metadata."""

    text: str
    category: str
    author: str


class Metadata(TypedDict):
    """Metadatos guardados junto a cada vector, usados después para filtrar."""

    category: str
    author: str
    ingested_at: str
    char_count: int


class Vector(TypedDict):
    """Registro que se envía al índice, con el formato de diccionario de Pinecone."""

    id: str
    values: list[float]
    metadata: Metadata


class UpsertResult(TypedDict):
    """Respuesta del upsert: cuántos vectores se cargaron."""

    upserted_count: int


class QueryResult(TypedDict):
    """Respuesta de la consulta: los vectores más similares."""

    matches: list[Vector]


# Filtro de metadatos de Pinecone, p. ej. {"category": {"$eq": "legal"}}.
Filtro = dict[str, dict[str, str]]


class MockPineconeIndex:
    """Índice de Pinecone simulado para probar el pipeline sin red.

    Imprime lo que recibe para poder verificar el tamaño de cada lote y el
    filtro que llega a la consulta.
    """

    async def upsert(self, vectors: list[Vector]) -> UpsertResult:
        """Simula la inserción de un lote y muestra su tamaño."""
        print(len(vectors))
        return {"upserted_count": len(vectors)}

    async def query(
        self, vector: list[float], top_k: int, filter: Filtro | None = None
    ) -> QueryResult:
        """Simula una consulta con filtro; siempre devuelve ``matches`` vacío."""
        print(f"query(top_k={top_k}, filter={filter})")
        return {"matches": []}


class IngestionPipeline:
    """Carga documentos en el índice por lotes y permite buscarlos por categoría."""

    def __init__(self, index: MockPineconeIndex) -> None:
        """Recibe el índice por parámetro para poder cambiar el mock por uno real."""
        self.index: MockPineconeIndex = index

    def create_metadata(self, doc_text: str, category: str, author: str) -> Metadata:
        """Arma los metadatos de un documento.

        ``category`` y ``author`` permiten filtrar en las búsquedas;
        ``ingested_at`` deja registro de cuándo se cargó y ``char_count`` el
        tamaño del texto, sin guardar el texto completo en la metadata.
        """
        return {
            "category": category,
            "author": author,
            "ingested_at": datetime.now().isoformat(),
            "char_count": len(doc_text),
        }

    async def process_and_upsert_batches(
        self, documents: list[Documento], batch_size: int = 100
    ) -> int:
        """Carga los documentos en lotes de ``batch_size`` y devuelve el total cargado.

        Por qué en lotes: Pinecone limita cada request en cantidad de vectores y
        en tamaño, así que un volumen grande no se puede mandar en una sola llamada.
        """
        total = 0
        for inicio in range(0, len(documents), batch_size):
            lote = documents[inicio : inicio + batch_size]
            vectors: list[Vector] = [
                {
                    "id": str(uuid.uuid4()),
                    "values": [0.1] * DIMENSION_MOCK,  # Embedding simulado.
                    "metadata": self.create_metadata(
                        doc["text"], doc["category"], doc["author"]
                    ),
                }
                for doc in lote
            ]
            # Se suma lo que confirma el índice, no lo que se envió.
            respuesta = await self.index.upsert(vectors)
            total += respuesta["upserted_count"]
        return total

    async def search_by_category(
        self, query_vector: list[float], category: str
    ) -> QueryResult:
        """Busca los vectores más similares restringidos a una categoría.

        El filtro ``$eq`` se aplica en el índice, antes de rankear por
        similitud, así que solo compiten documentos de esa categoría.
        """
        return await self.index.query(
            vector=query_vector, top_k=5, filter={"category": {"$eq": category}}
        )


async def main() -> None:
    """Ingesta 5 documentos en lotes de 2 y busca la categoría "legal"."""
    index = MockPineconeIndex()
    pipeline = IngestionPipeline(index)

    # Simulación de documentos: 2 de legal, 2 de hr y 1 de finanzas.
    docs: list[Documento] = [
        {"text": "Contrato de confidencialidad con proveedores", "category": "legal", "author": "User1"},
        {"text": "Política de protección de datos personales", "category": "legal", "author": "User2"},
        {"text": "Procedimiento de alta de nuevos empleados", "category": "hr", "author": "User3"},
        {"text": "Régimen de licencias y vacaciones", "category": "hr", "author": "User1"},
        {"text": "Presupuesto anual por área", "category": "finanzas", "author": "User2"},
    ]

    # 1. Ejecutar ingesta
    total = await pipeline.process_and_upsert_batches(docs, batch_size=2)
    print(f"Total procesado: {total}")

    # 2. Ejecutar búsqueda filtrada
    resultado = await pipeline.search_by_category([0.1] * DIMENSION_MOCK, "legal")
    print(f"Matches: {resultado['matches']}")


if __name__ == "__main__":
    asyncio.run(main())
