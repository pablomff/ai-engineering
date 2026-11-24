"""Tests del RAG híbrido sobre Pinecone.

No usan la red ni una cuenta de Pinecone:

- ``IndiceFalso`` imita las operaciones del índice que usa el pipeline
  (upsert, query, list, fetch, stats, delete) guardando todo en memoria.
- Los embeddings son ``DeterministicFakeEmbedding`` y los tokens se cuentan
  por palabras, así no se descarga ningún modelo.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever
from pinecone import PineconeError
from pydantic import ValidationError

from pipelines.rag_hibrido import evaluate
from pipelines.rag_hibrido.config import NAMESPACE_POR_DEFECTO, ConfigRAG
from pipelines.rag_hibrido.evaluate import (
    GOLDEN_SET_POR_DEFECTO,
    CasoEvaluacion,
    cargar_golden_set,
    evaluar_caso,
    fragmentos_por_fuente,
    precision_at_k,
    recall_at_k,
    resumir,
    techo_precision,
)
from pipelines.rag_hibrido.exceptions import (
    IndiceIncompatibleError,
    IngestaIncompletaError,
    NamespaceVacioError,
    PineconeNoDisponibleError,
)
from pipelines.rag_hibrido.indice import (
    ETIQUETA_MODELO,
    METRICA,
    IndiceVectorial,
    MetadataFragmento,
    VectorPinecone,
    valor_etiqueta,
    verificar_compatible,
)
from pipelines.rag_hibrido.ingesta import (
    CARPETA_POR_DEFECTO,
    armar_metadata,
    cargar_documentos,
    categoria_de,
    leer_pdf,
    limpiar_pagina,
    pagina_de,
    recalcular_inicios,
    descartar_vacios,
    indexar,
    seccion_de,
    subir,
    titulos,
)
from pipelines.rag_hibrido.retriever import (
    CLAVE_ID,
    BM25Retriever,
    PineconeRetriever,
    RAGSystem,
    cargar_corpus,
    crear_bm25,
    documento_desde,
    tokenizar,
)
from pipelines.rag_hibrido.embeddings import (
    MAX_TEXTOS_POR_PEDIDO,
    MODELO_EMBEDDINGS,
    PineconeEmbeddings,
)

DIMENSION = 16
NAMESPACE = "pruebas"


def contar_palabras(texto: str) -> int:
    """Contador de tokens falso: una palabra = un token."""
    return len(texto.split())


class IndiceFalso:
    """Índice de Pinecone en memoria, con las operaciones de ``IndiceVectorial``."""

    def __init__(self, fallar_lotes: bool = False, tamano_pagina: int = 2) -> None:
        """``fallar_lotes`` simula un upsert con lotes fallidos."""
        self.datos: dict[str, dict[str, tuple[list[float], MetadataFragmento]]] = {}
        self.fallar_lotes: bool = fallar_lotes
        self.tamano_pagina: int = tamano_pagina
        self.borrados: list[str] = []
        self.operaciones: list[str] = []
        # Simula estadísticas atrasadas (consistencia eventual): si no es None,
        # describe_index_stats informa este número en vez del real.
        self.cantidad_informada: int | None = None

    def upsert(
        self,
        *,
        vectors: Sequence[VectorPinecone],
        namespace: str = "",
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> SimpleNamespace:
        """Guarda los vectores, o informa lotes fallidos sin lanzar (como el SDK)."""
        if self.fallar_lotes:
            return SimpleNamespace(
                has_errors=True, failed_item_count=len(vectors),
                total_item_count=len(vectors), failed_batch_count=1, upserted_count=0,
            )
        self.operaciones.append("upsert")
        espacio = self.datos.setdefault(namespace, {})
        for id_vector, valores, metadata in vectors:
            espacio[id_vector] = (valores, metadata)
        return SimpleNamespace(has_errors=False, upserted_count=len(vectors))

    def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float] | None = None,
        namespace: str = "",
        include_metadata: bool = False,
    ) -> SimpleNamespace:
        """Devuelve los ``top_k`` más parecidos por coseno."""
        assert vector is not None

        def coseno(otro: Sequence[float]) -> float:
            producto = sum(a * b for a, b in zip(vector, otro, strict=True))
            normas = math.sqrt(sum(a * a for a in vector)) * math.sqrt(sum(b * b for b in otro))
            return producto / normas

        candidatos = [
            SimpleNamespace(id=i, score=coseno(v), metadata=m)
            for i, (v, m) in self.datos.get(namespace, {}).items()
        ]
        candidatos.sort(key=lambda c: c.score, reverse=True)
        return SimpleNamespace(matches=candidatos[:top_k])

    def list(self, *, namespace: str = "") -> Iterator[SimpleNamespace]:
        """Pagina los ids de a ``tamano_pagina``."""
        ids = sorted(self.datos.get(namespace, {}))
        for inicio in range(0, len(ids), self.tamano_pagina):
            pagina = ids[inicio : inicio + self.tamano_pagina]
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in pagina])

    def fetch(self, *, ids: Sequence[str], namespace: str = "") -> SimpleNamespace:
        """Trae la metadata de los ids pedidos."""
        espacio = self.datos.get(namespace, {})
        return SimpleNamespace(
            vectors={i: SimpleNamespace(metadata=espacio[i][1]) for i in ids if i in espacio}
        )

    def describe_index_stats(self) -> SimpleNamespace:
        """Cantidad de vectores por namespace (o ``cantidad_informada``, si se fijó)."""
        if self.cantidad_informada is not None:
            return SimpleNamespace(
                namespaces={NAMESPACE: SimpleNamespace(vector_count=self.cantidad_informada)}
            )
        return SimpleNamespace(
            namespaces={ns: SimpleNamespace(vector_count=len(v)) for ns, v in self.datos.items()}
        )

    def delete(self, *, ids: Sequence[str] | None = None, namespace: str = "") -> None:
        """Borra los ids pedidos y los registra en ``borrados``."""
        self.operaciones.append("delete")
        espacio = self.datos.get(namespace, {})
        for id_vector in ids or []:
            espacio.pop(id_vector, None)
            self.borrados.append(id_vector)


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    """Embeddings falsos: el mismo texto da siempre el mismo vector."""
    return DeterministicFakeEmbedding(size=DIMENSION)


@pytest.fixture
def carpeta_docs(tmp_path: Path) -> Path:
    """Una documentación mínima con subcarpeta, licencia y un archivo vacío."""
    (tmp_path / "advanced").mkdir()
    (tmp_path / "quickstart.md").write_text(
        "# QuickStart\n\nIntro.\n\n## Timeouts\n\nUsá timeout=5.\n", encoding="utf-8"
    )
    (tmp_path / "advanced" / "transports.md").write_text(
        "# Transports\n\n## Mock transports\n\nMockTransport devuelve respuestas fijas.\n",
        encoding="utf-8",
    )
    (tmp_path / "LICENSE.md").write_text("Copyright", encoding="utf-8")
    (tmp_path / "vacio.md").write_text("   ", encoding="utf-8")
    return tmp_path


# --- Configuración -----------------------------------------------------------


@pytest.fixture
def entorno(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Un entorno válido; cada test lo modifica a su gusto."""
    monkeypatch.setenv("PINECONE_API_KEY", "pc-falsa")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setenv("INDEX_NAME", "httpx-docs")
    monkeypatch.delenv("PINECONE_NAMESPACE", raising=False)
    return monkeypatch


def test_config_lee_el_entorno_y_oculta_las_keys(entorno: pytest.MonkeyPatch) -> None:
    config = ConfigRAG.from_env()
    assert config.index_name == "httpx-docs"
    assert config.namespace == NAMESPACE_POR_DEFECTO
    assert "pc-falsa" not in repr(config)
    assert config.pinecone_api_key.get_secret_value() == "pc-falsa"


@pytest.mark.parametrize("variable", ["PINECONE_API_KEY", "ANTHROPIC_API_KEY", "INDEX_NAME"])
def test_config_exige_las_variables_obligatorias(
    entorno: pytest.MonkeyPatch, variable: str
) -> None:
    entorno.setenv(variable, "")  # Vacía cuenta como ausente.
    with pytest.raises(ValidationError):
        ConfigRAG.from_env()


@pytest.mark.parametrize("nombre", ["Mayusculas", "con_guion_bajo", "-empieza-con-guion", "a" * 46])
def test_config_rechaza_nombres_de_indice_invalidos(
    entorno: pytest.MonkeyPatch, nombre: str
) -> None:
    entorno.setenv("INDEX_NAME", nombre)
    with pytest.raises(ValidationError):
        ConfigRAG.from_env()


# --- Índice ------------------------------------------------------------------


def test_valor_etiqueta_saca_la_barra() -> None:
    assert valor_etiqueta("intfloat/multilingual-e5-large") == "intfloat_multilingual-e5-large"
    assert valor_etiqueta(MODELO_EMBEDDINGS) == "multilingual-e5-large"


def test_verificar_compatible_acepta_un_indice_correcto() -> None:
    tags = {ETIQUETA_MODELO: valor_etiqueta(MODELO_EMBEDDINGS)}
    verificar_compatible("i", 1024, METRICA, tags, 1024)


def test_verificar_compatible_rechaza_otra_dimension() -> None:
    with pytest.raises(IndiceIncompatibleError, match="1536"):
        verificar_compatible("i", 1536, METRICA, None, 1024)


def test_verificar_compatible_rechaza_otra_metrica() -> None:
    with pytest.raises(IndiceIncompatibleError, match="euclidean"):
        verificar_compatible("i", 1024, "euclidean", None, 1024)


def test_verificar_compatible_rechaza_otro_modelo() -> None:
    with pytest.raises(IndiceIncompatibleError, match="otro-modelo"):
        verificar_compatible("i", 1024, METRICA, {ETIQUETA_MODELO: "otro-modelo"}, 1024)


def test_verificar_compatible_sin_etiqueta_solo_avisa(caplog: pytest.LogCaptureFixture) -> None:
    verificar_compatible("i", 1024, METRICA, None, 1024)
    assert "no tiene la etiqueta" in caplog.text


def test_el_indice_falso_cumple_el_protocolo() -> None:
    assert isinstance(IndiceFalso(), IndiceVectorial)


# --- Ingesta -----------------------------------------------------------------


def test_categoria_segun_la_carpeta() -> None:
    assert categoria_de("advanced/timeouts.md") == "avanzado"
    assert categoria_de("quickstart.md") == "general"
    assert categoria_de("rfc/rfc9113.pdf") == "rfc"


def test_cargar_documentos_recorre_subcarpetas_y_excluye_licencia(carpeta_docs: Path) -> None:
    documentos = cargar_documentos(carpeta_docs)
    assert [d.metadata["archivo"] for d in documentos] == [
        "advanced/transports.md",
        "quickstart.md",
    ]


def test_cargar_documentos_falla_si_no_existe_la_carpeta(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cargar_documentos(tmp_path / "no-existe")


def test_titulos_ignora_comentarios_dentro_de_bloques_de_codigo() -> None:
    texto = "# Título\n\n```python\n# un comentario\n```\n\n## Sección\n"
    assert [t for _, t in titulos(texto)] == ["Título", "Sección"]


def test_seccion_de_toma_el_ultimo_titulo_anterior() -> None:
    texto = "# Uno\n\nTexto.\n\n## Dos\n\nMás texto."
    lista = titulos(texto)
    assert seccion_de(lista, 0, "a.md") == "Uno"
    assert seccion_de(lista, texto.index("## Dos"), "a.md") == "Dos"
    assert seccion_de(lista, len(texto) - 1, "a.md") == "Dos"


def test_seccion_de_sin_titulos_usa_el_nombre_del_archivo() -> None:
    assert seccion_de([], 10, "advanced/event-hooks.md") == "event-hooks"


def test_descartar_vacios_quita_los_fragmentos_sin_texto() -> None:
    fragmentos = [
        Document(page_content="```", metadata={"archivo": "a.md", "fragmento": 1}),
        Document(page_content="Texto útil", metadata={"archivo": "a.md", "fragmento": 2}),
    ]
    assert [f.page_content for f in descartar_vacios(fragmentos)] == ["Texto útil"]


def test_armar_metadata_incluye_texto_fuente_seccion_y_categoria() -> None:
    documento = Document(
        page_content="# T\n\n## Mock transports\n\nMockTransport.",
        metadata={"archivo": "advanced/transports.md"},
    )
    fragmento = Document(
        page_content="## Mock transports\n\nMockTransport.",
        metadata={"archivo": "advanced/transports.md", "start_index": 5, "fragmento": 1, "tokens": 4},
    )
    [metadata] = armar_metadata([documento], [fragmento])
    assert metadata == {
        "texto": "## Mock transports\n\nMockTransport.",
        "fuente": "advanced/transports.md",
        "seccion": "Mock transports",
        "categoria": "avanzado",
        "fragmento": 1,
        "tokens": 4,
    }


def test_indexar_sube_los_fragmentos_con_ids_estables(
    carpeta_docs: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    indice = IndiceFalso()
    resultado = indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras)
    assert not resultado.omitida
    assert resultado.archivos == 2
    assert set(indice.datos[NAMESPACE]) == {"advanced/transports.md#1", "quickstart.md#1"}
    valores, metadata = indice.datos[NAMESPACE]["quickstart.md#1"]
    assert len(valores) == DIMENSION
    assert metadata["fuente"] == "quickstart.md"
    assert "timeout=5" in str(metadata["texto"])


def test_indexar_no_repite_si_el_namespace_esta_completo(
    carpeta_docs: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    indice = IndiceFalso()
    indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras)
    resultado = indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras)
    assert resultado.omitida
    assert indice.operaciones == ["upsert"]  # Solo el de la primera corrida.


def test_volver_a_correr_completa_un_namespace_a_medias(
    carpeta_docs: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    # Una ingesta anterior que falló a mitad de camino dejó solo uno de los dos fragmentos.
    indice = IndiceFalso()
    indice.datos[NAMESPACE] = {"quickstart.md#1": ([0.0] * DIMENSION, {"texto": "viejo"})}
    resultado = indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras)
    assert not resultado.omitida
    assert set(indice.datos[NAMESPACE]) == {"advanced/transports.md#1", "quickstart.md#1"}
    assert indice.borrados == []


def test_reindexar_sube_antes_de_borrar_y_solo_borra_los_obsoletos(
    carpeta_docs: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    indice = IndiceFalso()
    indice.datos[NAMESPACE] = {
        "viejo.md#1": ([0.0] * DIMENSION, {"texto": "de un archivo que ya no existe"}),
        "quickstart.md#1": ([0.0] * DIMENSION, {"texto": "versión vieja"}),
    }
    resultado = indexar(
        indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras, reindexar=True
    )
    # Primero se sube todo, después se borra: el namespace nunca queda vacío.
    assert indice.operaciones == ["upsert", "delete"]
    assert indice.borrados == ["viejo.md#1"]
    assert resultado.borrados == 1
    assert set(indice.datos[NAMESPACE]) == {"advanced/transports.md#1", "quickstart.md#1"}
    assert "timeout=5" in str(indice.datos[NAMESPACE]["quickstart.md#1"][1]["texto"])


def test_reindexar_sin_obsoletos_no_borra_nada(
    carpeta_docs: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    indice = IndiceFalso()
    indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras)
    indexar(indice, NAMESPACE, carpeta_docs, embeddings, contar_palabras, reindexar=True)
    assert indice.operaciones == ["upsert", "upsert"]


def test_indexar_falla_si_no_hay_documentos(
    tmp_path: Path, embeddings: DeterministicFakeEmbedding
) -> None:
    with pytest.raises(ValueError, match="No hay archivos"):
        indexar(IndiceFalso(), NAMESPACE, tmp_path, embeddings, contar_palabras)


def test_subir_avisa_si_fallan_lotes() -> None:
    vector: VectorPinecone = ("a.md#1", [0.1] * DIMENSION, {"texto": "a"})
    with pytest.raises(IngestaIncompletaError, match="Fallaron 1 de 1.*Volvé a correr"):
        subir(IndiceFalso(fallar_lotes=True), [vector], NAMESPACE)


def test_el_dataset_real_esta_en_el_repo() -> None:
    documentos = cargar_documentos(CARPETA_POR_DEFECTO)
    fuentes = {d.metadata["archivo"] for d in documentos}
    assert len(documentos) == 25  # 23 Markdown de httpx + 2 RFC en PDF
    assert {"advanced/timeouts.md", "rfc/rfc9112.pdf", "rfc/rfc9113.pdf"} <= fuentes


# --- Recuperador -------------------------------------------------------------


def test_tokenizar_normaliza_mayusculas_y_puntuacion() -> None:
    assert tokenizar("Usá httpx.AsyncClient()!") == ["usá", "httpx", "asyncclient"]


def test_documento_desde_saca_el_texto_de_la_metadata() -> None:
    documento = documento_desde("a.md#1", {"texto": "Hola", "fuente": "a.md"})
    assert documento.page_content == "Hola"
    assert documento.metadata == {"fuente": "a.md", CLAVE_ID: "a.md#1"}


def _cargar(indice: IndiceFalso, embeddings: DeterministicFakeEmbedding, textos: dict[str, str]) -> None:
    """Carga fragmentos de prueba con su embedding falso."""
    indice.upsert(
        vectors=[
            (id_, embeddings.embed_query(texto), {"texto": texto, "fuente": id_.split("#")[0]})
            for id_, texto in textos.items()
        ],
        namespace=NAMESPACE,
    )


async def test_pinecone_retriever_devuelve_documentos_con_score(
    embeddings: DeterministicFakeEmbedding,
) -> None:
    indice = IndiceFalso()
    _cargar(indice, embeddings, {"a.md#1": "uno", "b.md#1": "dos", "c.md#1": "tres"})
    retriever = PineconeRetriever(index=indice, embeddings=embeddings, namespace=NAMESPACE, k=2)
    documentos = await retriever.ainvoke("dos")
    assert len(documentos) == 2
    assert documentos[0].metadata[CLAVE_ID] == "b.md#1"  # Mismo texto: coseno 1.
    assert documentos[0].metadata["score"] == pytest.approx(1.0)


def test_cargar_corpus_recorre_todas_las_paginas(embeddings: DeterministicFakeEmbedding) -> None:
    indice = IndiceFalso(tamano_pagina=2)
    _cargar(indice, embeddings, {f"d{i}.md#1": f"texto {i}" for i in range(5)})
    corpus = cargar_corpus(indice, NAMESPACE)
    assert [d.metadata[CLAVE_ID] for d in corpus] == [f"d{i}.md#1" for i in range(5)]


class RetrieverFijo(BaseRetriever):
    """Retriever falso que devuelve siempre la misma lista."""

    documentos: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Ignora la consulta."""
        return self.documentos


class RetrieverQueFalla(BaseRetriever):
    """Retriever falso que simula una caída de Pinecone."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Siempre lanza un error de Pinecone."""
        raise PineconeError("sin conexión")


def _docs(*ids: str) -> list[Document]:
    """Documentos de prueba con id en la metadata."""
    return [Document(page_content=f"texto {i}", metadata={CLAVE_ID: i, "fuente": i}) for i in ids]


async def test_rag_system_fusiona_y_premia_lo_que_aparece_en_las_dos_listas() -> None:
    lexico = RetrieverFijo(documentos=_docs("a", "b", "c"))
    semantico = RetrieverFijo(documentos=_docs("d", "c", "e"))
    rag = RAGSystem(lexico, semantico, top_k=3)
    resultado = [d.metadata[CLAVE_ID] for d in await rag.recuperar("pregunta")]
    # "c" está en las dos listas: suma puntaje de ambas y queda primero.
    assert resultado[0] == "c"
    assert len(resultado) == 3


async def test_rag_system_no_repite_fragmentos() -> None:
    rag = RAGSystem(RetrieverFijo(documentos=_docs("a", "b")), RetrieverFijo(documentos=_docs("a", "b")))
    ids = [d.metadata[CLAVE_ID] for d in await rag.recuperar("pregunta")]
    assert ids == ["a", "b"]


async def test_rag_system_rechaza_consultas_vacias() -> None:
    rag = RAGSystem(RetrieverFijo(documentos=[]), RetrieverFijo(documentos=[]))
    with pytest.raises(ValueError):
        await rag.recuperar("   ")


async def test_rag_system_traduce_los_errores_de_pinecone() -> None:
    rag = RAGSystem(RetrieverFijo(documentos=[]), RetrieverQueFalla())
    with pytest.raises(PineconeNoDisponibleError):
        await rag.recuperar("pregunta")


def test_bm25_encuentra_nombres_tecnicos_exactos() -> None:
    corpus = [
        Document(page_content="Use MockTransport in tests", metadata={CLAVE_ID: "t#1"}),
        Document(page_content="Configure a timeout", metadata={CLAVE_ID: "x#1"}),
        Document(page_content="Proxies and SSL", metadata={CLAVE_ID: "p#1"}),
    ]
    resultado = crear_bm25(corpus, k=1).invoke("¿Cómo uso mocktransport?")
    assert resultado[0].metadata[CLAVE_ID] == "t#1"


# --- Evaluación --------------------------------------------------------------


def test_recall_vale_uno_si_el_documento_aparece() -> None:
    assert recall_at_k(["x", "a", "y"], "a") == 1.0
    assert recall_at_k(["x", "y"], "a") == 0.0


def test_recall_solo_mira_los_primeros_k() -> None:
    assert recall_at_k(["x", "y", "a"], "a", k=2) == 0.0


def test_precision_divide_siempre_por_k() -> None:
    assert precision_at_k(["a", "a", "x", "y", "z"], "a") == pytest.approx(0.4)
    assert precision_at_k(["a"], "a") == pytest.approx(0.2)  # Los huecos no suman.


def test_techo_de_precision() -> None:
    assert techo_precision(2) == pytest.approx(0.4)
    assert techo_precision(9) == pytest.approx(1.0)


def test_evaluar_caso_y_resumir() -> None:
    caso = CasoEvaluacion(pregunta="p", documento_id_esperado="a.md")
    recuperados = _docs("x.md", "a.md", "a.md", "y.md", "z.md")
    conteo = fragmentos_por_fuente(_docs("a.md", "a.md", "a.md", "x.md"))
    resultado = evaluar_caso(caso, recuperados, conteo)
    assert resultado.posicion == 2
    assert resultado.recall == 1.0
    assert resultado.precision == pytest.approx(0.4)
    assert resultado.techo_precision == pytest.approx(0.6)
    fallido = evaluar_caso(caso, _docs("x.md"), conteo)
    assert fallido.posicion is None
    resumen = resumir("prueba", [resultado, fallido])
    assert resumen.recall == pytest.approx(0.5)


def test_el_golden_set_tiene_cinco_preguntas_con_documentos_existentes() -> None:
    golden_set = cargar_golden_set(GOLDEN_SET_POR_DEFECTO)
    fuentes = {d.metadata["archivo"] for d in cargar_documentos(CARPETA_POR_DEFECTO)}
    assert len(golden_set) == 5
    assert all(caso.documento_id_esperado in fuentes for caso in golden_set)


def test_golden_set_invalido(tmp_path: Path) -> None:
    ruta = tmp_path / "golden.json"
    ruta.write_text('[{"pregunta": "sin documento esperado"}]', encoding="utf-8")
    with pytest.raises(ValidationError):
        evaluate.cargar_golden_set(ruta)


def test_resumir_sin_casos_falla() -> None:
    with pytest.raises(ValueError):
        resumir("prueba", [])


def test_golden_set_vacio(tmp_path: Path) -> None:
    ruta = tmp_path / "golden.json"
    ruta.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="no tiene preguntas"):
        cargar_golden_set(ruta)


async def test_top_k_traduce_los_errores_de_pinecone() -> None:
    with pytest.raises(PineconeNoDisponibleError):
        await evaluate._top_k(RetrieverQueFalla(), "pregunta")


# --- RAGSystem.desde_indice: consistencia eventual ---------------------------


def test_desde_indice_arma_el_sistema(embeddings: DeterministicFakeEmbedding) -> None:
    indice = IndiceFalso()
    _cargar(indice, embeddings, {"a.md#1": "uno", "b.md#1": "dos"})
    rag = RAGSystem.desde_indice(indice, NAMESPACE, embeddings)
    assert isinstance(rag.lexico, BM25Retriever)
    assert len(rag.lexico.docs) == 2


def test_desde_indice_falla_si_el_namespace_esta_vacio(
    embeddings: DeterministicFakeEmbedding,
) -> None:
    with pytest.raises(NamespaceVacioError, match="está vacío"):
        RAGSystem.desde_indice(IndiceFalso(), NAMESPACE, embeddings)


def test_desde_indice_falla_si_todavia_no_se_pueden_listar(
    embeddings: DeterministicFakeEmbedding,
) -> None:
    # Las estadísticas ya muestran vectores pero el listado todavía no los devuelve.
    indice = IndiceFalso()
    indice.cantidad_informada = 5
    with pytest.raises(NamespaceVacioError, match="todavía no lista"):
        RAGSystem.desde_indice(indice, NAMESPACE, embeddings)


def test_desde_indice_avisa_si_el_corpus_esta_incompleto(
    embeddings: DeterministicFakeEmbedding, caplog: pytest.LogCaptureFixture
) -> None:
    indice = IndiceFalso()
    _cargar(indice, embeddings, {"a.md#1": "uno", "b.md#1": "dos"})
    indice.cantidad_informada = 3
    RAGSystem.desde_indice(indice, NAMESPACE, embeddings)
    assert "BM25 leyó 2 fragmentos pero el namespace tiene 3" in caplog.text


# --- Embeddings de Pinecone --------------------------------------------------


class InferenciaFalsa:
    """Imita ``pc.inference``: devuelve un vector por texto y guarda cada pedido."""

    def __init__(self) -> None:
        """Arranca sin pedidos registrados."""
        self.pedidos: list[tuple[str, list[str], dict[str, str] | None]] = []

    def embed(
        self, model: str, inputs: Sequence[str], parameters: dict[str, str] | None = None
    ) -> list[SimpleNamespace]:
        """Devuelve como vector el largo de cada texto, para poder verificar el orden."""
        self.pedidos.append((model, list(inputs), parameters))
        return [SimpleNamespace(values=[float(len(texto))]) for texto in inputs]


def test_pinecone_embeddings_distingue_pasajes_y_preguntas() -> None:
    inferencia = InferenciaFalsa()
    embeddings = PineconeEmbeddings(inferencia)
    assert embeddings.embed_documents(["uno", "cuatro"]) == [[3.0], [6.0]]
    assert embeddings.embed_query("hola") == [4.0]
    tipos = [parametros["input_type"] for _, _, parametros in inferencia.pedidos if parametros]
    assert tipos == ["passage", "query"]
    assert all(modelo == MODELO_EMBEDDINGS for modelo, _, _ in inferencia.pedidos)


def test_pinecone_embeddings_no_trunca_en_silencio() -> None:
    inferencia = InferenciaFalsa()
    PineconeEmbeddings(inferencia).embed_query("hola")
    _, _, parametros = inferencia.pedidos[0]
    assert parametros is not None and parametros["truncate"] == "NONE"


def test_pinecone_embeddings_parte_en_lotes_y_conserva_el_orden() -> None:
    inferencia = InferenciaFalsa()
    textos = ["x" * (i + 1) for i in range(MAX_TEXTOS_POR_PEDIDO + 5)]
    vectores = PineconeEmbeddings(inferencia).embed_documents(textos)
    assert [len(lote) for _, lote, _ in inferencia.pedidos] == [MAX_TEXTOS_POR_PEDIDO, 5]
    assert vectores == [[float(i + 1)] for i in range(len(textos))]


# --- PDF ---------------------------------------------------------------------


def test_limpiar_pagina_saca_encabezado_pie_y_ligaduras() -> None:
    texto = (
        "Data is speci\ufb01ed in Section 5.\n"
        "RFC 9113 HTTP/2 June 2022\n"
        "Thomson & Ben\ufb01eld Standards Track Page 20"
    )
    lineas, titulos_linea = limpiar_pagina(texto)
    assert lineas == ["Data is specified in Section 5."]
    assert titulos_linea == [""]


def test_limpiar_pagina_descarta_el_indice_y_detecta_titulos() -> None:
    texto = (
        "5.2.\xa0\xa0Flow Control\n"       # índice de contenidos: espacio no separable
        "5.2. Flow Control \n"               # título real, en el cuerpo
        "1. The receiver sends a frame.\n"   # ítem de lista: termina en punto
        "Appendix A. Prohibited Cipher Suites"
    )
    lineas, titulos_linea = limpiar_pagina(texto)
    # La entrada del índice se descarta; el título real queda.
    assert lineas[0] == "5.2. Flow Control"
    assert titulos_linea == ["5.2. Flow Control", "", "Appendix A. Prohibited Cipher Suites"]


def test_pagina_de_toma_la_pagina_donde_empieza() -> None:
    paginas = [(0, 1), (100, 2), (250, 3)]
    assert pagina_de(paginas, 0) == 1
    assert pagina_de(paginas, 99) == 1
    assert pagina_de(paginas, 100) == 2
    assert pagina_de(paginas, 999) == 3


def test_leer_pdf_real_arma_el_mapa_de_paginas_y_secciones() -> None:
    documento = leer_pdf(CARPETA_POR_DEFECTO / "rfc" / "rfc9113.pdf", "rfc/rfc9113.pdf")
    paginas = documento.metadata["paginas"]
    titulos_pdf = [titulo for _, titulo in documento.metadata["titulos"]]
    assert paginas[0] == (0, 1)
    # 78 páginas, menos la 3 y la 4, que eran solo índice de contenidos y se descartan.
    assert len(paginas) == 76
    assert 3 not in {numero for _, numero in paginas}
    assert "5.2. Flow Control" in titulos_pdf
    assert "\ufb01" not in documento.page_content  # sin ligaduras
    assert "Standards Track Page" not in documento.page_content  # sin pies de página
    inicio = documento.page_content.index("5.2. Flow Control")
    assert pagina_de(paginas, inicio) == 20  # la misma página que muestra el PDF


def test_armar_metadata_agrega_la_pagina_solo_en_los_pdf() -> None:
    pdf = Document(
        page_content="1. Intro\nTexto.\n2. Frames\nMás texto.",
        metadata={"archivo": "rfc/x.pdf", "paginas": [(0, 1), (16, 2)], "titulos": [(0, "1. Intro"), (16, "2. Frames")]},
    )
    md = Document(page_content="# Hola\n\nTexto.", metadata={"archivo": "a.md"})
    fragmentos = [
        Document(page_content="2. Frames\nMás texto.", metadata={"archivo": "rfc/x.pdf", "start_index": 16, "fragmento": 2, "tokens": 4}),
        Document(page_content="# Hola", metadata={"archivo": "a.md", "start_index": 0, "fragmento": 1, "tokens": 2}),
    ]
    meta_pdf, meta_md = armar_metadata([pdf, md], fragmentos)
    assert meta_pdf["pagina"] == 2
    assert meta_pdf["seccion"] == "2. Frames"
    assert meta_pdf["categoria"] == "rfc"
    assert "pagina" not in meta_md


def test_recalcular_inicios_corrige_fragmentos_solapados() -> None:
    # El splitter deja -1 cuando hay solapamiento medido en tokens (ver docstring).
    texto = "uno dos tres cuatro cinco seis siete"
    documento = Document(page_content=texto, metadata={"archivo": "a.md"})
    fragmentos = [
        Document(page_content="uno dos tres cuatro", metadata={"archivo": "a.md", "start_index": 0, "fragmento": 1}),
        Document(page_content="tres cuatro cinco", metadata={"archivo": "a.md", "start_index": -1, "fragmento": 2}),
        Document(page_content="cinco seis siete", metadata={"archivo": "a.md", "start_index": -1, "fragmento": 3}),
    ]
    recalcular_inicios([documento], fragmentos)
    assert [f.metadata["start_index"] for f in fragmentos] == [0, 8, 20]


def test_recalcular_inicios_falla_si_el_fragmento_no_esta() -> None:
    documento = Document(page_content="abc", metadata={"archivo": "a.md"})
    fragmento = Document(page_content="xyz", metadata={"archivo": "a.md", "fragmento": 1})
    with pytest.raises(ValueError, match="a.md#1"):
        recalcular_inicios([documento], [fragmento])
