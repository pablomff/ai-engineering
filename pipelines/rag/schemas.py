"""Contrato de salida del pipeline RAG.

``RespuestaRAG`` es lo que devuelve ``get_rag_response``: el texto de la
respuesta y las referencias a los fragmentos que la respaldan. El mismo
modelo cumple dos funciones:

- ``PydanticOutputParser`` lo convierte en las instrucciones de formato que
  van en el prompt (el esquema JSON, con las ``description`` de cada campo).
- Después valida el JSON que devolvió el modelo, incluida la coherencia entre
  la respuesta y sus referencias.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.config import JsonDict

RESPUESTA_NO_SE = "No lo sé."
"""Respuesta exacta que el prompt le pide al modelo cuando el contexto no alcanza."""


def _sin_docstring(esquema: JsonDict) -> None:
    """Quita la descripción de la clase del esquema JSON.

    Pydantic usa el docstring de la clase como ``description`` del esquema,
    y ese esquema va dentro del prompt. Los docstrings de este módulo están
    escritos para quien lee el código (detalles de implementación, por qué
    se valida algo), no para el modelo: si viajaran, serían instrucciones
    confusas. Las ``description`` de cada campo sí están escritas para el
    modelo y se conservan.
    """
    esquema.pop("description", None)


def es_no_se(texto: str) -> bool:
    """Indica si ``texto`` es la respuesta "No lo sé", tolerando mayúsculas y el punto final."""
    return texto.strip().rstrip(".").casefold() == RESPUESTA_NO_SE.rstrip(".").casefold()


class Referencia(BaseModel):
    """Un fragmento del contexto que respalda la respuesta.

    ``archivo`` + ``fragmento`` identifican un fragmento de forma única, igual
    que el ID con el que se guardó en ChromaDB (``"arquitectura.md#2"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_sin_docstring)

    archivo: str = Field(
        min_length=1,
        description="Nombre del archivo, tal como aparece en el contexto (ej. 'arquitectura.md').",
    )
    fragmento: int = Field(
        ge=1,
        description="Número de fragmento dentro del archivo, tal como aparece en el contexto.",
    )


class RespuestaRAG(BaseModel):
    """Respuesta basada en los documentos, con sus fuentes.

    Reglas de coherencia (se validan acá, no se le confían al modelo):

    - Si la respuesta es "No lo sé.", no lleva referencias: si el modelo
      citó alguna igual, se descartan, porque no respaldan nada.
    - Cualquier otra respuesta necesita al menos una referencia. Una respuesta
      sin fuentes no se puede verificar y es la forma típica de una
      alucinación, así que se rechaza.

    ``encontrada`` se calcula a partir de las referencias. Es un
    ``computed_field``: aparece en el JSON de salida, pero no en el esquema
    que se le manda al modelo, que no tiene que completarlo.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_sin_docstring)

    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta en español basada solo en el contexto, o exactamente "
            f'"{RESPUESTA_NO_SE}" si el contexto no contiene la respuesta.'
        ),
    )
    referencias: list[Referencia] = Field(
        description=(
            "Fragmentos del contexto que respaldan la respuesta. "
            f'Lista vacía si la respuesta es "{RESPUESTA_NO_SE}".'
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def encontrada(self) -> bool:
        """``True`` si la respuesta salió de los documentos, ``False`` si es "No lo sé"."""
        return bool(self.referencias)

    @field_validator("referencias")
    @classmethod
    def _sin_duplicados(cls, referencias: list[Referencia]) -> list[Referencia]:
        """Elimina referencias repetidas, conservando el orden."""
        return list(dict.fromkeys(referencias))

    @model_validator(mode="after")
    def _coherencia(self) -> Self:
        """Normaliza "No lo sé" y rechaza respuestas sin referencias."""
        self.respuesta = self.respuesta.strip()
        if es_no_se(self.respuesta):
            self.respuesta = RESPUESTA_NO_SE
            self.referencias = []
        elif not self.referencias:
            raise ValueError("una respuesta sin referencias no está respaldada por el contexto")
        return self
