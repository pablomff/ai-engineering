"""Tests del pipeline RAG: ingesta, base vectorial, contrato de salida y cadena.

No usan la red ni descargan modelos:

- Los embeddings son ``DeterministicFakeEmbedding`` (el mismo texto da
  siempre el mismo vector), y los tokens se cuentan por palabras.
- ChromaDB es real, persistido en una carpeta temporal de cada test.
- El LLM es ``ModeloFalso``, que devuelve respuestas predefinidas y guarda
  los prompts que recibió.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import anthropic
import httpx2
import pytest
from chromadb.api import ClientAPI
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from pipelines.rag import chain
from pipelines.rag.chain import (
    TOP_K,
    crear_cadena,
    crear_modelo,
    crear_retriever,
    formatear_contexto,
    get_rag_response,
    validar_referencias,
    verificar_fin,
)
from pipelines.rag.exceptions import (
    BaseNoIndexadaError,
    EmbeddingsNoCoincidenError,
    ModeloNegativaError,
    RAGConfigError,
    RAGProveedorError,
    RecursosNoDisponiblesError,
    RespuestaInvalidaError,
)
from pipelines.rag.ingesta import (
    CARPETA_POR_DEFECTO,
    TOKENS_DE_SOLAPAMIENTO,
    TOKENS_POR_FRAGMENTO,
    cargar_documentos,
    crear_splitter,
    fragmentar,
    id_de,
    indexar,
    partir_documento,
)
from pipelines.rag.schemas import RESPUESTA_NO_SE, Referencia, RespuestaRAG
from pipelines.rag.vectorstore import (
    DIRECTORIO_POR_DEFECTO,
    NOMBRE_COLECCION_TEMPORAL,
    RAIZ_PROYECTO,
    abrir_cliente,
    abrir_cliente_existente,
    abrir_para_consulta,
    leer_coleccion,
)

MODELO = "modelo-falso"


def contar_palabras(texto: str) -> int:
    """Contador de "tokens" para tests: una palabra, un token."""
    return len(texto.split())


def doc(archivo: str, fragmento: int, texto: str = "texto") -> Document:
    """Arma un fragmento como los que devuelve la base."""
    return Document(page_content=texto, metadata={"archivo": archivo, "fragmento": fragmento})


def respuesta_json(respuesta: str, referencias: list[tuple[str, int]]) -> str:
    """JSON como el que devolvería el modelo siguiendo las instrucciones de formato."""
    return json.dumps(
        {
            "respuesta": respuesta,
            "referencias": [{"archivo": a, "fragmento": f} for a, f in referencias],
        }
    )


class ModeloFalso(BaseChatModel):
    """Chat model que devuelve textos predefinidos y guarda los prompts recibidos."""

    respuestas: list[str]
    stop_reason: str = "end_turn"
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        """Identificador del modelo, requerido por ``BaseChatModel``."""
        return "falso"

    def _generate(self, messages: list[BaseMessage], *args: object, **kwargs: object) -> ChatResult:
        """Devuelve la siguiente respuesta (o la última) con el ``stop_reason`` configurado."""
        indice = min(len(self.prompts), len(self.respuestas) - 1)
        self.prompts.append(messages)
        mensaje = AIMessage(
            content=self.respuestas[indice], response_metadata={"stop_reason": self.stop_reason}
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def retriever_fijo(documentos: list[Document]) -> Runnable[str, list[Document]]:
    """Retriever falso que siempre devuelve ``documentos``."""
    return RunnableLambda(lambda _pregunta: documentos)


@pytest.fixture
def embeddings() -> Embeddings:
    """Embeddings deterministas y sin red."""
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def cliente(tmp_path: Path) -> ClientAPI:
    """Base ChromaDB persistida en una carpeta temporal."""
    return abrir_cliente(tmp_path / "vectorstore")


@pytest.fixture
def carpeta(tmp_path: Path) -> Path:
    """Carpeta de documentos con un .md largo, un .txt, uno vacío y uno ignorado."""
    data = tmp_path / "data"
    data.mkdir()
    secciones = "\n\n".join(f"## Sección {i}\n\n" + " ".join(["palabra"] * 300) for i in range(3))
    (data / "largo.md").write_text(f"# Largo\n\n{secciones}", encoding="utf-8")
    (data / "corto.txt").write_text("Las guardias cambian los lunes.", encoding="utf-8")
    (data / "vacio.md").write_text("   \n", encoding="utf-8")
    (data / "imagen.png").write_bytes(b"\x89PNG")
    return data


@pytest.fixture
def sin_cadena_cacheada(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Borra las variables del LLM y la cadena cacheada antes y después del test."""
    for variable in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(chain, "_cadena", None)
    yield monkeypatch


# --- Ingesta ---------------------------------------------------------------


def test_cargar_documentos_lee_md_y_txt_en_orden(carpeta: Path) -> None:
    documentos = cargar_documentos(carpeta)
    assert [d.metadata["archivo"] for d in documentos] == ["corto.txt", "largo.md"]


def test_cargar_documentos_falla_si_no_existe_la_carpeta(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cargar_documentos(tmp_path / "no-existe")


def test_splitter_usa_500_tokens_con_50_de_solapamiento() -> None:
    splitter = crear_splitter(contar_palabras)
    assert TOKENS_POR_FRAGMENTO >= 500
    assert splitter._chunk_size == TOKENS_POR_FRAGMENTO
    assert splitter._chunk_overlap == TOKENS_DE_SOLAPAMIENTO == 50


def test_fragmentar_respeta_el_tamano_y_numera_desde_1(carpeta: Path) -> None:
    fragmentos = fragmentar(cargar_documentos(carpeta), contar_palabras)
    largos = [f for f in fragmentos if f.metadata["archivo"] == "largo.md"]
    assert len(largos) > 1
    assert [f.metadata["fragmento"] for f in largos] == list(range(1, len(largos) + 1))
    assert all(f.metadata["tokens"] <= TOKENS_POR_FRAGMENTO for f in fragmentos)
    # Corta por títulos: cada fragmento después del primero arranca en una sección.
    assert all(f.page_content.startswith("## Sección") for f in largos[1:])
    assert id_de(largos[1]) == "largo.md#2"


def test_fragmentar_solapa_cuando_corta_dentro_de_un_parrafo() -> None:
    # Un único párrafo de 1200 palabras numeradas: el splitter tiene que cortar
    # entre palabras, y ahí el solapamiento repite las del final del fragmento anterior.
    texto = " ".join(f"p{i}" for i in range(1200))
    fragmentos = fragmentar(
        [Document(page_content=texto, metadata={"archivo": "a.md"})], contar_palabras
    )
    primero, segundo = fragmentos[0].page_content.split(), fragmentos[1].page_content.split()
    repetidas = [p for p in segundo if p in set(primero)]
    assert 0 < len(repetidas) <= TOKENS_DE_SOLAPAMIENTO


def test_indexar_guarda_los_fragmentos_con_el_modelo(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    resultado = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    assert resultado.omitida is False
    assert resultado.archivos == 2
    coleccion = leer_coleccion(cliente)
    assert coleccion is not None
    assert coleccion.count() == resultado.fragmentos
    assert coleccion.metadata["modelo_embeddings"] == MODELO
    assert "corto.txt#1" in coleccion.get()["ids"]


def test_indexar_no_reindexa_si_la_base_ya_existe(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    primera = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    (carpeta / "nuevo.txt").write_text("Un documento nuevo.", encoding="utf-8")

    segunda = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    assert segunda.omitida is True
    assert segunda.fragmentos == primera.fragmentos

    tercera = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras, reindexar=True)
    assert tercera.omitida is False
    assert tercera.archivos == 3


def test_indexar_rechaza_base_de_otro_modelo_sin_reindexar(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    indexar(carpeta, cliente, embeddings, "otro-modelo", contar_palabras)
    with pytest.raises(EmbeddingsNoCoincidenError):
        indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    resultado = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras, reindexar=True)
    assert resultado.omitida is False


def test_indexar_con_carpeta_vacia_no_borra_la_base(
    carpeta: Path, tmp_path: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    primera = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(ValueError):
        indexar(vacia, cliente, embeddings, MODELO, contar_palabras, reindexar=True)
    coleccion = leer_coleccion(cliente)
    assert coleccion is not None and coleccion.count() == primera.fragmentos


class EmbeddingsQueFallan(Embeddings):
    """Embeddings que fallan al calcular, como un modelo que no se pudo descargar."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Simula la falla al cargar el modelo."""
        raise OSError("sin conexión para descargar el modelo")

    def embed_query(self, text: str) -> list[float]:
        """Simula la falla al cargar el modelo."""
        raise OSError("sin conexión para descargar el modelo")


def test_reindexar_que_falla_no_borra_la_base_anterior(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    primera = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    with pytest.raises(OSError):
        indexar(carpeta, cliente, EmbeddingsQueFallan(), MODELO, contar_palabras, reindexar=True)
    coleccion = leer_coleccion(cliente)
    assert coleccion is not None and coleccion.count() == primera.fragmentos
    assert leer_coleccion(cliente, NOMBRE_COLECCION_TEMPORAL) is None


def test_ingesta_interrumpida_no_deja_una_base_parcial(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    with pytest.raises(OSError):
        indexar(carpeta, cliente, EmbeddingsQueFallan(), MODELO, contar_palabras)
    assert leer_coleccion(cliente) is None
    # La corrida siguiente no la toma por completa: indexa de verdad.
    assert indexar(carpeta, cliente, embeddings, MODELO, contar_palabras).omitida is False


def test_restos_de_una_indexacion_cortada_se_limpian(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    cliente.create_collection(NOMBRE_COLECCION_TEMPORAL)  # como si un Ctrl-C la hubiera dejado
    resultado = indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    assert resultado.omitida is False
    assert leer_coleccion(cliente, NOMBRE_COLECCION_TEMPORAL) is None


def test_fragmentos_respetan_el_limite_aunque_los_tokens_no_se_sumen() -> None:
    # Con este contador, el texto unido tiene más tokens que la suma de sus
    # pedazos, como puede pasar con un tokenizer real en los bordes.
    def contar_no_aditivo(texto: str) -> int:
        palabras = len(texto.split())
        return palabras + palabras * palabras // 10_000

    parrafos = "\n\n".join(" ".join(["palabra"] * 60) for _ in range(20))
    documento = Document(page_content=parrafos, metadata={"archivo": "a.md"})
    sin_verificar = crear_splitter(contar_no_aditivo).split_documents([documento])
    assert max(contar_no_aditivo(p.page_content) for p in sin_verificar) > TOKENS_POR_FRAGMENTO

    partes = partir_documento(documento, contar_no_aditivo)
    assert all(contar_no_aditivo(p.page_content) <= TOKENS_POR_FRAGMENTO for p in partes)


# --- Base vectorial --------------------------------------------------------


def test_abrir_cliente_existente_no_crea_la_carpeta(tmp_path: Path) -> None:
    with pytest.raises(BaseNoIndexadaError):
        abrir_cliente_existente(tmp_path / "no-existe")
    assert not (tmp_path / "no-existe").exists()


def test_abrir_para_consulta_no_modifica_la_coleccion(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    antes = leer_coleccion(cliente)
    assert antes is not None
    metadata, configuracion = antes.metadata, antes.configuration_json
    abrir_para_consulta(cliente, embeddings, MODELO)
    despues = leer_coleccion(cliente)
    assert despues is not None
    assert despues.metadata == metadata
    assert despues.configuration_json == configuracion


def test_abrir_para_consulta_exige_base_indexada(
    cliente: ClientAPI, embeddings: Embeddings
) -> None:
    with pytest.raises(BaseNoIndexadaError):
        abrir_para_consulta(cliente, embeddings, MODELO)


def test_abrir_para_consulta_detecta_embeddings_distintos(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    indexar(carpeta, cliente, embeddings, "otro-modelo", contar_palabras)
    with pytest.raises(EmbeddingsNoCoincidenError, match="otro-modelo"):
        abrir_para_consulta(cliente, embeddings, MODELO)


def test_retriever_devuelve_top_k_fragmentos(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    retriever = crear_retriever(abrir_para_consulta(cliente, embeddings, MODELO), k=2)
    documentos = retriever.invoke("¿Cuándo cambian las guardias?")
    assert len(documentos) == 2
    assert all({"archivo", "fragmento"} <= d.metadata.keys() for d in documentos)


# --- Contrato de salida ----------------------------------------------------


def test_respuesta_con_referencias_esta_encontrada() -> None:
    respuesta = RespuestaRAG.model_validate_json(respuesta_json("Los lunes.", [("a.md", 1)]))
    assert respuesta.encontrada is True
    assert respuesta.model_dump()["encontrada"] is True


@pytest.mark.parametrize("texto", ["No lo sé.", "no lo sé", "  NO LO SÉ.  "])
def test_no_lo_se_se_normaliza_y_descarta_referencias(texto: str) -> None:
    respuesta = RespuestaRAG.model_validate_json(respuesta_json(texto, [("a.md", 1)]))
    assert respuesta.respuesta == RESPUESTA_NO_SE
    assert respuesta.referencias == []
    assert respuesta.encontrada is False


def test_respuesta_sin_referencias_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="sin referencias"):
        RespuestaRAG.model_validate_json(respuesta_json("Los lunes.", []))


def test_referencias_repetidas_se_eliminan() -> None:
    respuesta = RespuestaRAG.model_validate_json(
        respuesta_json("Los lunes.", [("a.md", 1), ("a.md", 1), ("b.md", 2)])
    )
    assert respuesta.referencias == [
        Referencia(archivo="a.md", fragmento=1),
        Referencia(archivo="b.md", fragmento=2),
    ]


def test_esquema_para_el_modelo_no_incluye_encontrada_ni_docstrings() -> None:
    esquema = RespuestaRAG.model_json_schema()
    assert set(esquema["properties"]) == {"respuesta", "referencias"}
    assert "description" not in esquema
    assert "description" not in esquema["$defs"]["Referencia"]


# --- Piezas de la cadena ---------------------------------------------------


def test_formatear_contexto_incluye_archivo_y_fragmento() -> None:
    contexto = formatear_contexto([doc("a.md", 1, "uno"), doc("b.txt", 3, "tres")])
    assert "[archivo: a.md | fragmento: 1]\nuno" in contexto
    assert "[archivo: b.txt | fragmento: 3]\ntres" in contexto


@pytest.mark.parametrize(
    ("motivo", "error"),
    [("refusal", ModeloNegativaError), ("max_tokens", RespuestaInvalidaError)],
)
def test_verificar_fin_rechaza_negativas_y_cortes(motivo: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        verificar_fin(AIMessage(content="{}", response_metadata={"stop_reason": motivo}))


def test_validar_referencias_rechaza_fuentes_inventadas() -> None:
    respuesta = RespuestaRAG.model_validate_json(respuesta_json("Algo.", [("inventado.md", 9)]))
    with pytest.raises(RespuestaInvalidaError):
        validar_referencias({"documentos": [doc("a.md", 1)], "respuesta": respuesta})


# --- Cadena completa -------------------------------------------------------


async def test_cadena_arma_el_prompt_con_contexto_y_pregunta() -> None:
    modelo = ModeloFalso(respuestas=[respuesta_json("Los lunes a las 10:00.", [("runbook.md", 2)])])
    documentos = [doc("runbook.md", 2, "La rotación cambia los lunes.")]
    cadena = crear_cadena(retriever_fijo(documentos), modelo)

    respuesta = await get_rag_response("¿Cuándo cambia la rotación?", cadena)

    assert respuesta.encontrada is True
    assert respuesta.referencias == [Referencia(archivo="runbook.md", fragmento=2)]
    sistema, humano = modelo.prompts[0]
    assert '"No lo sé."' in sistema.content
    assert "La rotación cambia los lunes." in humano.content
    assert "PREGUNTA: ¿Cuándo cambia la rotación?" in humano.content


async def test_cadena_devuelve_no_lo_se_sin_error() -> None:
    modelo = ModeloFalso(respuestas=[respuesta_json("No lo sé.", [])])
    cadena = crear_cadena(retriever_fijo([doc("a.md", 1)]), modelo)
    respuesta = await get_rag_response("¿Pregunta trampa?", cadena)
    assert respuesta.encontrada is False
    assert respuesta.respuesta == RESPUESTA_NO_SE


async def test_cadena_acepta_json_dentro_de_un_bloque_de_codigo() -> None:
    texto = "```json\n" + respuesta_json("Sí.", [("a.md", 1)]) + "\n```"
    modelo = ModeloFalso(respuestas=[texto])
    cadena = crear_cadena(retriever_fijo([doc("a.md", 1)]), modelo)
    respuesta = await get_rag_response("¿?", cadena)
    assert respuesta.encontrada is True


@pytest.mark.parametrize(
    "texto",
    [
        "Esto no es JSON",
        respuesta_json("Sin fuentes.", []),
        respuesta_json("Fuente inventada.", [("otro.md", 7)]),
    ],
)
async def test_cadena_convierte_salidas_invalidas_en_error_propio(texto: str) -> None:
    modelo = ModeloFalso(respuestas=[texto])
    with pytest.raises(RespuestaInvalidaError):
        await get_rag_response("¿?", crear_cadena(retriever_fijo([doc("a.md", 1)]), modelo))


async def test_cadena_con_chroma_real(
    carpeta: Path, cliente: ClientAPI, embeddings: Embeddings
) -> None:
    indexar(carpeta, cliente, embeddings, MODELO, contar_palabras)
    retriever = crear_retriever(abrir_para_consulta(cliente, embeddings, MODELO))
    modelo = ModeloFalso(respuestas=[respuesta_json("Los lunes.", [("corto.txt", 1)])])

    cadena = crear_cadena(retriever, modelo)
    respuesta = await get_rag_response("¿Cuándo cambian las guardias?", cadena)

    assert respuesta.referencias == [Referencia(archivo="corto.txt", fragmento=1)]
    # El prompt recibió como mucho TOP_K fragmentos.
    assert modelo.prompts[0][1].content.count("[archivo:") <= TOP_K


# --- get_rag_response y configuración --------------------------------------


async def test_get_rag_response_rechaza_pregunta_vacia() -> None:
    with pytest.raises(ValueError):
        await get_rag_response("   ", RunnableLambda(lambda _: None))


async def test_get_rag_response_traduce_errores_de_anthropic() -> None:
    def falsa(_: str) -> RespuestaRAG:
        respuesta = httpx2.Response(429, request=httpx2.Request("POST", "https://api.example.com"))
        raise anthropic.RateLimitError("x", response=respuesta, body=None)

    with pytest.raises(RAGProveedorError) as info:
        await get_rag_response("¿?", RunnableLambda(falsa))
    assert info.value.error.kind == "rate_limit"


async def test_get_rag_response_no_oculta_bugs() -> None:
    def falsa(_: str) -> RespuestaRAG:
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        await get_rag_response("¿?", RunnableLambda(falsa))


def test_crear_modelo_sin_api_key_es_error_de_configuracion(
    sin_cadena_cacheada: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RAGConfigError, match="ANTHROPIC_API_KEY"):
        crear_modelo()


async def test_get_rag_response_sin_base_es_error_de_configuracion(
    sin_cadena_cacheada: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sin_cadena_cacheada.setenv("ANTHROPIC_API_KEY", "sk-test")
    sin_cadena_cacheada.setattr(chain, "DIRECTORIO_POR_DEFECTO", tmp_path / "vacia")
    falsos = DeterministicFakeEmbedding(size=8)
    sin_cadena_cacheada.setattr(chain, "crear_embeddings", lambda: falsos)
    with pytest.raises(BaseNoIndexadaError):
        await get_rag_response("¿?")
    assert not (tmp_path / "vacia").exists()


async def test_get_rag_response_envuelve_fallas_al_cargar_el_modelo(
    sin_cadena_cacheada: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def sin_conexion() -> Embeddings:
        raise OSError("no se pudo descargar el modelo")

    sin_cadena_cacheada.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "base").mkdir()
    sin_cadena_cacheada.setattr(chain, "DIRECTORIO_POR_DEFECTO", tmp_path / "base")
    sin_cadena_cacheada.setattr(chain, "crear_embeddings", sin_conexion)
    with pytest.raises(RecursosNoDisponiblesError) as info:
        await get_rag_response("¿?")
    assert isinstance(info.value, RAGConfigError)
    assert chain._cadena is None  # la falla no queda cacheada


async def test_get_rag_response_usa_la_cadena_creada_sin_ir_a_un_hilo(
    sin_cadena_cacheada: pytest.MonkeyPatch,
) -> None:
    esperada = RespuestaRAG.model_validate_json(respuesta_json("Sí.", [("a.md", 1)]))

    def prohibido(*args: object, **kwargs: object) -> None:
        raise AssertionError("no debería ir a un hilo si la cadena ya existe")

    sin_cadena_cacheada.setattr(chain, "_cadena", RunnableLambda(lambda _: esperada))
    sin_cadena_cacheada.setattr(chain.asyncio, "to_thread", prohibido)
    assert await get_rag_response("¿?") is esperada


def test_rutas_por_defecto_son_las_de_la_raiz_del_repo() -> None:
    assert DIRECTORIO_POR_DEFECTO == RAIZ_PROYECTO / "vectorstore"
    assert CARPETA_POR_DEFECTO == RAIZ_PROYECTO / "data"
    assert (RAIZ_PROYECTO / "pyproject.toml").is_file()
