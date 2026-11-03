# AI Engineering (Coderhouse)

Mi carpeta de trabajo del curso. Uso Python 3.12, con el proyecto gestionado por `uv`.

## Cómo correr un script

Desde esta carpeta:

```bash
uv run python modulo-01-asyncio/ejemplo_embeddings.py
```

Para agregar librerías cuando el curso las pida (por ejemplo, en el módulo 2):

```bash
uv add openai anthropic
```

## Estructura

- `modulo-01-asyncio/`: arquitectura asíncrona con Python 3.12

## Nota sobre iCloud

Documents se sincroniza con iCloud. Por eso el entorno virtual vive en `.venv.nosync`,
una carpeta que iCloud no sube, y `.venv` es un enlace simbólico a ella. Si alguna vez
se rompe, se recrea así:

```bash
rm -rf .venv .venv.nosync && uv venv .venv.nosync --python 3.12 && ln -s .venv.nosync .venv
```

Las API keys van en un archivo `.env`, que está en `.gitignore`. Nunca se suben al repo.
