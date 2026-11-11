"""Contrato de salida del pipeline de extracción.

Define con Pydantic la estructura que el LLM debe devolver. Por qué Pydantic:
el mismo modelo sirve para dos cosas a la vez. LangChain lo convierte en el
esquema que se le envía al modelo (``with_structured_output``), y además
valida la respuesta, así que un objeto que no cumple el contrato nunca llega
al resto de la aplicación.

Las ``description`` de cada campo no son solo documentación: viajan dentro del
esquema y el modelo las lee como instrucciones.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

ERROR_SIN_TECNOLOGIAS = "sin_tecnologias"
"""Tipo de error de Pydantic cuando ``tecnologias`` queda vacía.

Tiene un tipo propio para que la cadena lo distinga de los demás errores de
validación: una lista vacía significa que el texto no menciona tecnologías, y
volver a preguntarle al modelo no lo va a cambiar.
"""

class NivelCriticidad(StrEnum):
    """Nivel de criticidad técnica de lo descrito en el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto.

    Restricciones del contrato:

    - ``tecnologias`` no puede estar vacía: un texto sin ninguna tecnología
      identificable se trata como una extracción fallida.
    - ``resumen_tecnico`` no puede estar vacío.
    - No se aceptan campos extra, para detectar respuestas que no siguen
      el esquema.
    """

    model_config = ConfigDict(extra="forbid")

    # El mínimo de 1 se valida en _limpiar_tecnologias y no con min_length:
    # min_length viaja al esquema como "minItems", la API obliga al modelo a
    # cumplirlo y, ante un texto sin tecnologías, el modelo inventa una
    # ("No identificada") en lugar de devolver la lista vacía.
    tecnologias: list[str] = Field(
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios "
            "mencionados en el texto, con su nombre propio (ej. 'PostgreSQL')."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Gravedad técnica de la situación descrita: baja, media o alta.",
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Resumen técnico de una o dos oraciones, en español.",
    )

    @field_validator("tecnologias")
    @classmethod
    def _limpiar_tecnologias(cls, tecnologias: list[str]) -> list[str]:
        """Quita espacios, descarta strings vacíos y elimina duplicados.

        Se conserva el orden de aparición y los duplicados se comparan sin
        distinguir mayúsculas. Si la lista llega vacía, o queda vacía después
        de limpiar, falla con el tipo de error ``ERROR_SIN_TECNOLOGIAS``.
        """
        limpias: list[str] = []
        vistas: set[str] = set()
        for tecnologia in tecnologias:
            nombre = tecnologia.strip()
            clave = nombre.casefold()
            if nombre and clave not in vistas:
                vistas.add(clave)
                limpias.append(nombre)
        if not limpias:
            raise PydanticCustomError(
                ERROR_SIN_TECNOLOGIAS, "la lista de tecnologías no puede quedar vacía"
            )
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def _limpiar_resumen(cls, resumen: str) -> str:
        """Quita espacios sobrantes y rechaza un resumen en blanco."""
        resumen = resumen.strip()
        if not resumen:
            raise ValueError("el resumen técnico no puede estar vacío")
        return resumen
