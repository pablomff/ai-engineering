"""Embeddings calculados por Pinecone (Inference API) en vez de un modelo local.

Por qué Pinecone: el sistema ya depende de Pinecone para guardar y buscar, y
la misma ``PINECONE_API_KEY`` sirve para calcular los vectores. Así nada
pesado corre en la máquina: no hay que descargar ni cargar un modelo de
casi 500 MB para indexar o para consultar, y cualquier máquina con la key
puede usar el sistema.

El modelo es ``multilingual-e5-large``: la versión grande de la familia e5
que usa el RAG local (multilingüe, con el mismo tokenizer), con vectores de
1024 dimensiones.

Lo único que todavía se descarga es el tokenizer (unos pocos MB, no el
modelo), para medir los fragmentos en tokens antes de mandarlos.
"""

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from langchain_core.embeddings import Embeddings
from pinecone import Pinecone
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODELO_EMBEDDINGS = "multilingual-e5-large"
"""Modelo de embeddings de Pinecone.

Multilingüe (las preguntas están en español y la documentación en inglés)
y asimétrico: embebe distinto una pregunta (``query``) y un texto donde se
busca (``passage``). Pinecone agrega internamente los prefijos que el
modelo espera, así que acá no se escriben.
"""

TOKENIZER = "intfloat/multilingual-e5-large"
"""Tokenizer del mismo modelo en HuggingFace, para contar tokens localmente.

Es el mismo tokenizer de ``multilingual-e5-small`` (XLM-RoBERTa), así que
los fragmentos de 500 tokens del RAG local miden lo mismo acá.
"""

MAX_TEXTOS_POR_PEDIDO = 96
"""Máximo de textos que Pinecone acepta en un pedido de embeddings."""

TRUNCADO = "NONE"
"""Qué hace Pinecone con un texto de más de 507 tokens: ``NONE`` es fallar.

La alternativa (``END``) cortaría el final sin avisar, y esa parte del
fragmento quedaría fuera de la búsqueda. Los fragmentos tienen como máximo
500 tokens, y con el prefijo y los tokens especiales que agrega Pinecone
quedan por debajo de 507: si alguno se pasara, es mejor enterarse.
"""

TipoEntrada = Literal["query", "passage"]


class InferenciaPinecone(Protocol):
    """La parte de ``pc.inference`` que se usa: el método ``embed``.

    Es un ``Protocol`` para que los tests pasen una versión falsa, sin red.
    """

    def embed(
        self, model: str, inputs: Sequence[str], parameters: dict[str, str] | None = None
    ) -> Sequence["EmbeddingDenso"]:
        """Devuelve un embedding por texto, en el mismo orden."""
        ...


class EmbeddingDenso(Protocol):
    """Un embedding de la respuesta: lo único que se usa son sus valores."""

    values: list[float]


class PineconeEmbeddings(Embeddings):
    """Adaptador entre la Inference API de Pinecone y la interfaz ``Embeddings`` de LangChain.

    Heredar de ``Embeddings`` permite usarlo en cualquier lugar que espere
    un modelo de LangChain, igual que el modelo local del RAG local.
    ``langchain-pinecone`` trae uno parecido, pero exige ``pinecone<8``.
    """

    def __init__(self, inferencia: InferenciaPinecone, modelo: str = MODELO_EMBEDDINGS) -> None:
        """Recibe ``pc.inference`` (o un falso en los tests) y el nombre del modelo."""
        self.inferencia: InferenciaPinecone = inferencia
        self.modelo: str = modelo

    def _embeber(self, textos: Sequence[str], tipo: TipoEntrada) -> list[list[float]]:
        """Pide los embeddings de ``textos`` en lotes de ``MAX_TEXTOS_POR_PEDIDO``."""
        vectores: list[list[float]] = []
        for inicio in range(0, len(textos), MAX_TEXTOS_POR_PEDIDO):
            lote = textos[inicio : inicio + MAX_TEXTOS_POR_PEDIDO]
            respuesta = self.inferencia.embed(
                model=self.modelo,
                inputs=list(lote),
                parameters={"input_type": tipo, "truncate": TRUNCADO},
            )
            vectores.extend(list(embedding.values) for embedding in respuesta)
        return vectores

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddings de los fragmentos que se guardan (``input_type="passage"``)."""
        return self._embeber(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        """Embedding de una pregunta (``input_type="query"``)."""
        return self._embeber([text], "query")[0]


def crear_embeddings(pc: Pinecone) -> PineconeEmbeddings:
    """Crea el adaptador de embeddings sobre el cliente de Pinecone ya configurado.

    Usa el mismo cliente (y por lo tanto la misma key, timeout y política
    de reintentos) que el resto del pipeline.
    """
    return PineconeEmbeddings(pc.inference)


@functools.cache
def crear_contador_tokens() -> Callable[[str], int]:
    """Devuelve una función que cuenta tokens con el tokenizer del modelo.

    Solo descarga el tokenizer (unos pocos MB), no el modelo. Se cachea
    porque cargarlo tarda un par de segundos.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def contar_tokens(texto: str) -> int:
        """Cantidad de tokens de ``texto``, sin los tokens especiales de inicio y fin."""
        return len(tokenizer.encode(texto, add_special_tokens=False))

    return contar_tokens
