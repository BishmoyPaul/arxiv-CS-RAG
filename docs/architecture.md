# Architecture

## Purpose

`Github_Repo` contains the engineering-focused implementation of the ArXiv CS RAG project.

It owns:

- the shared Python package
- the FastAPI service
- the local Gradio UI
- Docker packaging
- test coverage
- CI configuration

It does not own:

- the built retrieval indices
- the deployed Hugging Face Space runtime
- the Kaggle update workflow

## High-Level Shape

```text
User Query
  |
  +--> FastAPI: app/main.py
  |
  +--> Local Gradio UI: app/gradio_app.py
            |
            v
      Shared Package: src/arxiv_cs_rag/
            |
            +--> config.py
            +--> retrieval.py
            +--> pipeline.py
            +--> generation.py
            +--> formatters.py
```

Both delivery surfaces call the same shared package. UI code should stay thin. API code should stay thin.

## Core Package

The reusable application logic lives in `src/arxiv_cs_rag`.

- `config.py`
  Reads environment-driven settings such as `LOCAL_DATA_DIR`, `HF_TOKEN`, `GEMINI_API_KEY`, and `RAG_SOURCE`.
- `retrieval.py`
  Handles semantic retrieval, live arXiv search, source normalization, and fallback behavior.
- `pipeline.py`
  Orchestrates retrieval plus prompt construction.
- `generation.py`
  Wraps Gemini generation and streaming behavior.
- `formatters.py`
  Shapes prompt entries and markdown search output.

This package is the source of truth for business logic.

## Delivery Surfaces

### FastAPI

`app/main.py` exposes:

- `GET /health`
- `POST /search`
- `POST /ask`

FastAPI owns:

- HTTP request/response models
- validation
- dependency wiring
- stable error mapping

FastAPI should not own retrieval or generation logic.

### Local Gradio

`app/gradio_app.py` provides a direct local UI entrypoint.

It uses:

- `app/gradio_controller.py` for UI-facing orchestration
- the same shared package used by FastAPI

The local Gradio app is intended for local demo and manual validation, not as the primary hosted deployment path.

### Docker

Docker packages the FastAPI service, not the Hugging Face Space.

Large index artifacts are mounted at runtime instead of copied into the image.

Current Docker support includes:

- image build
- container startup
- `/health`
- live arXiv search
- semantic search against a mounted real index

## Data And Artifact Boundaries

The retrieval index lives outside `Github_Repo`.

Local semantic search expects:

- `LOCAL_DATA_DIR` to point at the parent directory of `arxiv_colbert`

Example:

```text
Arxiv-CS-RAG-indices/
  arxiv_colbert/
```

Set:

```powershell
$env:LOCAL_DATA_DIR="<INDEX_ROOT>"
```

`Github_Repo` should not absorb these artifacts into version control.

## Platform Support Model

### Native local

Exercised locally:

- FastAPI startup
- local Gradio startup
- live arXiv search

Important caveat:

- native semantic search may require a local compiler toolchain because ColBERT/Torch extensions can compile at runtime

On Windows, this usually means MSVC Build Tools and `cl`.
On Linux, this usually means `g++` or equivalent build tooling.

In practice, the project supports:

- local native mode for package, API, and live arXiv search
- Docker for predictable semantic retrieval without local toolchain setup

If semantic retriever initialization fails natively, the local Gradio app now falls back to live arXiv search with a warning.

### Docker

Docker is the most reliable path for semantic search.

It is the recommended option for users who want predictable semantic retrieval behavior without local compiler setup.

## Relationship To Other Workspace Folders

### HF_Space

`HF_Space` remains the public Hugging Face deployment surface.

The current sibling-path integration is a local development convenience, not the final standalone hosted packaging strategy.

### Arxiv-CS-RAG-indices

This folder is an artifact store for ColBERT indices.

### Kaggle_Update_Ipynb

This folder contains the legacy update workflow and still has drift that should be cleaned later.

## Open Items

These items are not required for the repo to be usable today, but remain open:

- real Gemini token-bearing validation
- observing GitHub Actions on a real runner after push
- final hosted HF Space reuse strategy
- Kaggle workflow cleanup
