"""Configuración del RAG híbrido, leída de variables de entorno.

Igual que ``llm_client.settings``, valida todo una sola vez al arrancar: si
falta una variable, el error aparece enseguida y con un mensaje claro, no en
la primera llamada a Pinecone.

Variables:

- ``PINECONE_API_KEY``: obligatoria, autentica contra Pinecone.
- ``INDEX_NAME``: obligatoria, nombre del índice serverless.
- ``ANTHROPIC_API_KEY``: obligatoria por consigna (la key del LLM del
  sistema RAG). Esta entrega solo recupera documentos, así que no la usa
  para llamar a la API: la valida para que el entorno quede listo para el
  paso de generación.
- ``PINECONE_NAMESPACE``: opcional, por defecto ``httpx-docs``.

Este módulo no carga el ``.env``: lo hace cada script al arrancar.
"""

import os
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr

NAMESPACE_POR_DEFECTO = "httpx-docs"
"""Namespace donde vive la documentación de httpx dentro del índice.

Separar por namespace permite que el mismo índice guarde otros conjuntos de
documentos (otra librería, otro cliente) sin que se mezclen en las búsquedas:
una consulta solo ve los vectores de su namespace.
"""

# Variable de entorno -> campo de ConfigRAG.
_ENV_VARS: dict[str, str] = {
    "PINECONE_API_KEY": "pinecone_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "INDEX_NAME": "index_name",
    "PINECONE_NAMESPACE": "namespace",
}


class ConfigRAG(BaseModel):
    """Configuración validada e inmutable del RAG sobre Pinecone.

    Las keys son ``SecretStr``: al imprimir el objeto se ven como
    ``'**********'``, así no terminan en un log por accidente.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pinecone_api_key: SecretStr
    anthropic_api_key: SecretStr
    # Pinecone solo acepta minúsculas, números y guiones, hasta 45 caracteres.
    index_name: str = Field(pattern=r"^[a-z0-9]([a-z0-9-]{0,43}[a-z0-9])?$")
    namespace: str = Field(default=NAMESPACE_POR_DEFECTO, min_length=1)

    @classmethod
    def from_env(cls) -> Self:
        """Construye la configuración a partir de ``os.environ``.

        Las variables vacías cuentan como ausentes (una línea ``INDEX_NAME=``
        en el ``.env`` no es un nombre válido), así Pydantic informa
        "Field required" en vez de aceptar un string vacío.

        Raises:
            pydantic.ValidationError: si falta una variable obligatoria o
                ``INDEX_NAME`` no es un nombre válido para Pinecone.
        """
        valores: dict[str, str] = {}
        for variable, campo in _ENV_VARS.items():
            valor = os.environ.get(variable, "").strip()
            if valor:
                valores[campo] = valor
        return cls.model_validate(valores)
