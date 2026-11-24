"""Recuperador híbrido: combina búsqueda léxica (BM25) y semántica (Pinecone).

Uso::

    uv run python -m pipelines.rag_hibrido.retriever "¿Cómo configuro un timeout?"

Por qué dos búsquedas:

- La **semántica** (embeddings en Pinecone) entiende el significado: encuentra
  "tiempo máximo de espera" aunque el documento diga "timeout", y hasta
  cruza idiomas (preguntas en español, documentación en inglés). Pero un
  nombre propio como ``MockTransport`` es para ella una palabra más, y
  puede devolver fragmentos que hablan de algo parecido en general.
- La **léxica** (BM25) busca las palabras exactas, pesando más las raras.
  ``MockTransport`` aparece en muy pocos fragmentos, así que los encuentra
  enseguida. Pero no sabe que "tiempo de espera" y "timeout" son lo mismo.

``EnsembleRetriever`` las fusiona con *Reciprocal Rank Fusion* (RRF): cada
fragmento suma ``peso / (60 + posición)`` por cada lista donde aparece. Un
fragmento que está arriba en las dos listas gana; uno que solo aparece en
una puede entrar igual si está muy arriba. RRF usa posiciones y no
puntajes, así que no importa que BM25 y coseno midan en escalas distintas.

El corpus de BM25 se arma leyendo los fragmentos desde Pinecone (el texto
está en la metadata), no desde los archivos: así las dos búsquedas trabajan
sobre exactamente los mismos fragmentos, con los mismos ids.
"""

import asyncio
import logging
import re
import sys
import warnings
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import NotFoundError, PineconeError
from pydantic import ConfigDict, ValidationError

from pipelines.rag_hibrido.config import ConfigRAG
from pipelines.rag_hibrido.embeddings import crear_embeddings
from pipelines.rag_hibrido.exceptions import (
    IndiceInexistenteError,
    IndiceNoListoError,
    NamespaceVacioError,
    PineconeNoDisponibleError,
    RAGHibridoError,
)
from pipelines.rag_hibrido.indice import (
    IndiceVectorial,
    MetadataFragmento,
    configurar_logs,
    conectar,
    crear_cliente,
    dimension_embeddings,
    verificar_compatible,
)
from pipelines.rag_hibrido.ingesta import vectores_en

# langchain-community se está discontinuando y avisa con un DeprecationWarning
# al importarlo. BM25Retriever no tiene todavía otro paquete, y la consigna
# lo pide, así que se importa igual y se silencia solo ese aviso.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

TOP_K = 5
"""Cantidad de fragmentos que devuelve ``RAGSystem.recuperar``."""

K_CANDIDATOS = 10
"""Fragmentos que aporta cada búsqueda antes de fusionar.

Se piden más que ``TOP_K`` para que la fusión tenga de dónde elegir: un
fragmento que está sexto en BM25 y séptimo en la búsqueda semántica puede
terminar entre los 5 primeros del ranking combinado. Con solo 5 por lado,
nunca lo vería.
"""

PESOS = (0.5, 0.5)
"""Peso de BM25 y de la búsqueda semántica en la fusión, en ese orden.

Mitad y mitad es el punto de partida neutral. Si la evaluación mostrara que
una búsqueda acierta mucho más que la otra, se ajustaría acá.
"""

CLAVE_ID = "id"
"""Clave de la metadata con el id del fragmento.

``EnsembleRetriever`` la usa para detectar que el mismo fragmento vino de
las dos búsquedas y sumar sus puntajes en vez de mostrarlo dos veces.
"""


def tokenizar(texto: str) -> list[str]:
    """Parte un texto en palabras para BM25: minúsculas y solo letras, números y ``_``.

    El tokenizador por defecto de ``BM25Retriever`` es ``str.split()``, que
    separa solo por espacios: ``AsyncClient`` y ``AsyncClient()`` serían
    palabras distintas, y ``Timeout`` no coincidiría con ``timeout``. Con
    esta versión, ``httpx.AsyncClient()`` da ``["httpx", "asyncclient"]``.
    """
    return re.findall(r"\w+", texto.lower())


def documento_desde(id_fragmento: str, metadata: MetadataFragmento) -> Document:
    """Convierte un vector de Pinecone (id + metadata) en un ``Document`` de LangChain.

    El texto sale de la metadata y pasa a ``page_content``; el resto
    (fuente, sección, categoría...) queda como metadata, más el id.
    """
    datos: dict[str, str | int] = {k: v for k, v in metadata.items() if k != "texto"}
    datos[CLAVE_ID] = id_fragmento
    return Document(id=id_fragmento, page_content=str(metadata["texto"]), metadata=datos)


class PineconeRetriever(BaseRetriever):
    """Búsqueda semántica en un namespace de Pinecone, como retriever de LangChain.

    Se escribió con el SDK nativo porque ``langchain-pinecone`` exige
    ``pinecone<8`` (el proyecto usa la 10) e instala ``langchain-openai``.
    Heredar de ``BaseRetriever`` es lo que permite meterlo en un
    ``EnsembleRetriever`` junto a BM25.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: IndiceVectorial
    embeddings: Embeddings
    namespace: str
    k: int = K_CANDIDATOS

    def _buscar(self, consulta: str) -> list[Document]:
        """Calcula el embedding de la consulta y pide los ``k`` fragmentos más cercanos."""
        vector = self.embeddings.embed_query(consulta)
        respuesta = self.index.query(
            vector=vector, top_k=self.k, namespace=self.namespace, include_metadata=True
        )
        documentos: list[Document] = []
        for match in respuesta.matches:
            documento = documento_desde(match.id, match.metadata or {})
            documento.metadata["score"] = round(match.score, 4)
            documentos.append(documento)
        return documentos

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Versión síncrona, la que llama ``invoke``."""
        return self._buscar(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Versión async, la que llama ``ainvoke``.

        El modelo de embeddings y el cliente de Pinecone son síncronos: se
        corren en un hilo aparte para no bloquear el event loop.
        """
        return await asyncio.to_thread(self._buscar, query)


def cargar_corpus(index: IndiceVectorial, namespace: str) -> list[Document]:
    """Trae del namespace todos los fragmentos (texto + metadata), ordenados por id.

    Primero lista los ids (de a páginas de 100) y después trae cada página
    con ``fetch``. Es la base de BM25, que necesita el corpus completo en
    memoria para calcular qué tan rara es cada palabra.

    Escala bien hasta decenas de miles de fragmentos. Con millones, BM25 en
    memoria deja de ser viable y conviene la búsqueda híbrida nativa de
    Pinecone (vectores *sparse*).
    """
    corpus: list[Document] = []
    for pagina in index.list(namespace=namespace):
        ids = [item.id for item in pagina.vectors]
        respuesta = index.fetch(ids=ids, namespace=namespace)
        for id_fragmento, vector in respuesta.vectors.items():
            corpus.append(documento_desde(id_fragmento, vector.metadata or {}))
    return sorted(corpus, key=lambda d: str(d.metadata[CLAVE_ID]))


def crear_bm25(corpus: Sequence[Document], k: int = K_CANDIDATOS) -> BM25Retriever:
    """Crea el retriever léxico sobre ``corpus`` con el tokenizador propio."""
    return BM25Retriever.from_documents(corpus, k=k, preprocess_func=tokenizar)


class RAGSystem:
    """Sistema de recuperación híbrido: recibe una consulta y devuelve el top-5.

    Encapsula un ``EnsembleRetriever`` que fusiona BM25 y Pinecone. Se crea
    con ``await RAGSystem.desde_config(config)``, que conecta, verifica el
    índice y arma el corpus de BM25; o directamente con dos retrievers, que
    es como lo usan los tests.
    """

    def __init__(
        self,
        lexico: BaseRetriever,
        semantico: BaseRetriever,
        *,
        top_k: int = TOP_K,
        pesos: tuple[float, float] = PESOS,
    ) -> None:
        """Arma el ensemble con los dos retrievers, en el orden de ``pesos``."""
        self.lexico: BaseRetriever = lexico
        self.semantico: BaseRetriever = semantico
        self.top_k: int = top_k
        self._ensemble: EnsembleRetriever = EnsembleRetriever(
            retrievers=[lexico, semantico], weights=list(pesos), id_key=CLAVE_ID
        )

    @classmethod
    async def desde_config(cls, config: ConfigRAG) -> "RAGSystem":
        """Conecta con Pinecone, verifica el índice y arma los dos retrievers.

        Todo el trabajo es bloqueante (llamadas HTTP síncronas), así que corre
        en un hilo aparte.

        Raises:
            IndiceInexistenteError: si no hay un índice con ``INDEX_NAME``.
            IndiceIncompatibleError: si el índice no coincide con el modelo.
            IndiceNoListoError: si el índice todavía no está listo.
            NamespaceVacioError: si no se corrió la ingesta.
            PineconeNoDisponibleError: si falla Pinecone tras los reintentos.
        """
        try:
            return await asyncio.to_thread(cls._construir, config)
        except PineconeError as exc:
            raise PineconeNoDisponibleError(
                f"No se pudo leer el índice de Pinecone ({type(exc).__name__})"
            ) from exc

    @classmethod
    def _construir(cls, config: ConfigRAG) -> "RAGSystem":
        """Parte síncrona de ``desde_config``: verifica el índice y se conecta.

        Raises:
            IndiceInexistenteError: si no hay un índice con ese nombre.
            IndiceIncompatibleError: si no coincide con el modelo.
            IndiceNoListoError: si Pinecone todavía no lo marca como listo.
        """
        pc = crear_cliente(config)
        embeddings = crear_embeddings(pc)
        try:
            descripcion = pc.describe_index(config.index_name)
        except NotFoundError as exc:
            raise IndiceInexistenteError(
                f"No existe el índice {config.index_name}: revisá INDEX_NAME o corré "
                "'uv run python -m pipelines.rag_hibrido.ingesta', que lo crea"
            ) from exc
        verificar_compatible(
            config.index_name,
            descripcion.dimension,
            descripcion.metric,
            descripcion.tags,
            dimension_embeddings(embeddings),
        )
        if not descripcion.status.ready:
            raise IndiceNoListoError(
                f"El índice {config.index_name} no está listo (estado: "
                f"{descripcion.status.state}): esperá unos segundos y volvé a intentar"
            )
        index = conectar(pc, descripcion.host)
        return cls.desde_indice(index, config.namespace, embeddings)

    @classmethod
    def desde_indice(
        cls, index: IndiceVectorial, namespace: str, embeddings: Embeddings
    ) -> "RAGSystem":
        """Arma los dos retrievers sobre un índice ya conectado.

        Está separada de ``_construir`` para poder probarla con un índice
        falso, sin Pinecone.

        Las estadísticas y el listado de ids de Pinecone son *eventualmente
        consistentes*: justo después de una ingesta pueden mostrar menos
        vectores de los que hay. Por eso se verifica que el corpus no esté
        vacío y se avisa si su tamaño no coincide con las estadísticas: en
        ese caso BM25 vería menos fragmentos que la búsqueda semántica.

        Raises:
            NamespaceVacioError: si el namespace no tiene vectores, o si
                todavía no se pueden listar.
        """
        esperados = vectores_en(index, namespace)
        if esperados == 0:
            raise NamespaceVacioError(
                f"El namespace {namespace} está vacío: corré primero "
                "'uv run python -m pipelines.rag_hibrido.ingesta' (si recién la corriste, "
                "esperá unos segundos: las estadísticas de Pinecone tardan en actualizarse)"
            )
        corpus = cargar_corpus(index, namespace)
        if not corpus:
            raise NamespaceVacioError(
                f"Pinecone todavía no lista los fragmentos de {namespace}: esperá unos "
                "segundos y volvé a intentar"
            )
        if len(corpus) != esperados:
            logger.warning(
                "BM25 leyó %d fragmentos pero el namespace tiene %d: si recién corriste la "
                "ingesta, esperá unos segundos para que BM25 vea el corpus completo",
                len(corpus), esperados,
            )
        logger.info("Corpus de BM25: %d fragmentos leídos de Pinecone", len(corpus))
        semantico = PineconeRetriever(index=index, embeddings=embeddings, namespace=namespace)
        return cls(crear_bm25(corpus), semantico)

    async def recuperar(self, consulta: str) -> list[Document]:
        """Devuelve los ``top_k`` fragmentos más relevantes para ``consulta``.

        Cada ``Document`` trae el texto en ``page_content`` y en la metadata
        ``id``, ``fuente``, ``seccion``, ``categoria`` y ``fragmento``.

        Raises:
            ValueError: si la consulta está vacía.
            PineconeNoDisponibleError: si falla Pinecone tras los reintentos.
        """
        if not consulta.strip():
            raise ValueError("La consulta está vacía")
        try:
            fusionados = await self._ensemble.ainvoke(consulta)
        except PineconeError as exc:
            raise PineconeNoDisponibleError(
                f"Falló la búsqueda en Pinecone ({type(exc).__name__})"
            ) from exc
        return fusionados[: self.top_k]


async def _demo(consulta: str) -> int:
    """Corre una consulta y muestra el top-5 en consola."""
    try:
        rag = await RAGSystem.desde_config(ConfigRAG.from_env())
        documentos = await rag.recuperar(consulta)
    except ValidationError as exc:
        print(f"Configuración inválida en el .env:\n{exc}", file=sys.stderr)
        return 1
    except (RAGHibridoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"\nConsulta: {consulta}\n")
    for posicion, documento in enumerate(documentos, start=1):
        meta = documento.metadata
        # Solo los fragmentos de PDF tienen página.
        pagina = f"  página {meta['pagina']}" if "pagina" in meta else ""
        print(f"{posicion}. {meta[CLAVE_ID]}  [{meta['categoria']}]  sección: {meta['seccion']}{pagina}")
        print(f"   {documento.page_content[:160].replace(chr(10), ' ')}...\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos: la consulta es el primer argumento."""
    argumentos = list(sys.argv[1:] if argv is None else argv)
    consulta = " ".join(argumentos) or "¿Cómo configuro un timeout en httpx?"
    load_dotenv()
    configurar_logs()
    return asyncio.run(_demo(consulta))


if __name__ == "__main__":
    sys.exit(main())
