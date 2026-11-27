"""RAG con recuperación híbrida (BM25 + vectores) sobre Pinecone Serverless.

Tiene cuatro piezas que se usan en este orden:

1. **Índice** (``indice.py``): crea el índice serverless en Pinecone si no
   existe y verifica que su dimensión coincida con el modelo de embeddings.
2. **Ingesta** (``ingesta.py``): lee la documentación de httpx, la parte en
   fragmentos y los sube a Pinecone con el texto y la fuente en la metadata.
3. **Recuperador** (``retriever.py``): ``RAGSystem`` combina búsqueda léxica
   (BM25) y semántica (Pinecone) con un ``EnsembleRetriever`` y devuelve el
   top-5.
4. **Evaluación** (``evaluate.py``): mide Recall@5 y Precision@5 sobre un
   golden set de preguntas con su documento esperado.

Reutiliza el splitter del RAG local (``pipelines/rag``), así los fragmentos
se arman con las mismas reglas. Los embeddings, en cambio, los calcula
Pinecone (``embeddings.py``): no hay ningún modelo que descargar ni cargar.
"""
