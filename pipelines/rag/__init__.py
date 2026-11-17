"""Pipeline RAG (Retrieval-Augmented Generation) sobre documentos locales.

Tiene dos etapas que se corren por separado:

1. **Ingesta** (``ingesta.py``): lee los ``.md`` y ``.txt`` de una carpeta, los
   parte en fragmentos y guarda sus embeddings en una base ChromaDB local.
2. **Consulta** (``chain.py``): ``get_rag_response`` busca los fragmentos más
   parecidos a la pregunta y le pide al LLM una respuesta basada solo en ellos,
   con las referencias a los fragmentos usados.

Las dos etapas comparten ``vectorstore.py``, que fija el modelo de embeddings
para que indexar y consultar usen siempre el mismo.
"""
