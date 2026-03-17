# ArXiv CS RAG

This repository contains the engineering-focused implementation of ArXiv CS RAG. It includes the reusable RAG package, the FastAPI service, the local Gradio UI, Docker packaging, tests, and CI configuration behind the public Hugging Face demo.

The public demo lives at [ArXiv CS RAG](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG).

## How The Space Works
https://github.com/user-attachments/assets/160a0fbd-3d59-49b1-9d4f-7535a7044e80

- **Input a Question**: The user inputs a question into the interface.
- **Abstract Retrieval**: The system uses ColBERTv2 to search ArXiv for the most relevant paper abstracts related to the question.
- **Contextual Answer Generation**: The retrieved abstracts are then fed into a Gemma-family model served through the Gemini API to generate a detailed and accurate answer.
- **Output**: The final answer, along with the relevant abstracts, is displayed to the user.

## Key Features
- **Question-Based ArXiv Paper Retrieval**: Automatically fetches the most relevant ArXiv paper abstracts by using a question as input.
- **ColBERTv2 Retriever**: Employs ColBERTv2 to search a local semantic index when the retriever runtime is available.
- **Live ArXiv Search Fallback**: Falls back to live arXiv search when semantic retrieval is unavailable.
- **LLM-Powered Answers**: Uses Gemini-backed generation when configured, while keeping tests mock-driven by default.

## Codes
- [Create Embeddings from ArXiv Abstracts](https://www.kaggle.com/code/artemenon/arxiv-cs-rag-query-cs-papers-using-colbertv2/) | Alternatively, you can check out the code [here](https://github.com/BishmoyPaul/arxiv-CS-RAG/blob/main/generate_embeddings.ipynb) too
- [Build Huggingface Space](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG/tree/main)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the current package and delivery-surface layout.

The short version:

- shared core: `src/arxiv_cs_rag`
- API entrypoint: `app/main.py`
- local UI entrypoint: `app/gradio_app.py`
- Docker packages the FastAPI service
- retrieval indices stay outside the repo

## Runtime Modes

This project supports three local usage modes:

- shared package mode
  Import `arxiv_cs_rag` directly from your own Python code.
- API mode
  Run FastAPI directly or through Docker.
- UI mode
  Run the local Gradio app directly from the repo.

## Local Setup

Use a Python 3.11 environment for local execution. The examples below use Conda, but any isolated environment is fine.

Example:

```powershell
conda create -n arxiv-cs-rag python=3.11
conda activate arxiv-cs-rag
```

Install the package:

```powershell
pip install -e .
```

Installation choices:

- `pip install -e .`
  Installs the shared package plus the core retrieval, API, and Gemini runtime dependencies. This is the default because the repo is meant to support the package and FastAPI surfaces out of the box.
- `pip install -e ".[ui]"`
  Adds Gradio for the local UI entrypoint.
- Docker
  Is the recommended way to run semantic retrieval predictably if you do not want to manage native compiler/toolchain requirements locally.

If you want the local UI too:

```powershell
pip install -e ".[ui]"
```

Run tests:

```powershell
python -m pytest
```

## Local API

The FastAPI service lives in `app/main.py` and reuses the shared package in `src/arxiv_cs_rag`.

Run the API locally:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Local Gradio UI

The repo also includes a native local Gradio entrypoint at `app/gradio_app.py`. This lets you run the UI directly without starting FastAPI first.

If you are installing from scratch, include the optional UI extra:

```powershell
pip install -e ".[ui]"
```

If you want local retrieval against a real index, point `LOCAL_DATA_DIR` at the parent directory that contains `arxiv_colbert`:

```powershell
$env:LOCAL_DATA_DIR="<INDEX_ROOT>"
```

Example:

```text
<INDEX_ROOT>/
  arxiv_colbert/
```

Run the local Gradio app:

```powershell
python -m app.gradio_app
```

This local UI reuses the same shared package in `src/arxiv_cs_rag` that powers the FastAPI service.

Native local notes:

- live arXiv search has been exercised locally
- Gradio startup has been exercised locally
- native semantic search may require a compiler toolchain because ColBERT/Torch extensions can compile at runtime
- if semantic retriever initialization fails locally, the app falls back to live arXiv search with a warning

For predictable semantic search without native toolchain setup, prefer the Docker path.

## Docker

Docker support is intended for local reproducibility of the FastAPI app. Large index artifacts are not bundled into the image.

Prerequisite:

- Docker Desktop or another compatible Docker daemon must be running locally before `docker build` or `docker run`.

1. Copy `.env.example` to `.env` and fill in any runtime values you need.
2. Build the image from `Github_Repo`.

```powershell
docker build -t arxiv-cs-rag-api .
```

3. Run the container.

```powershell
docker run --rm -p 8000:8000 --env-file .env arxiv-cs-rag-api
```

4. Check the health endpoint.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

For a basic startup check, `.env` may keep `HF_TOKEN`, `GEMINI_API_KEY`, and `RAG_SOURCE` blank. The `/health` endpoint should still return `200 OK`.

If you already have index files locally, mount them into the container and set `LOCAL_DATA_DIR` accordingly. Keep Hugging Face tokens and Gemini keys outside version control.

Example mounted-index run:

```powershell
docker run --rm -p 8000:8000 `
  --env-file .env `
  -e LOCAL_DATA_DIR=/app/rag_index_data `
  -v "<INDEX_ROOT>:/app/rag_index_data:ro" `
  arxiv-cs-rag-api
```

With that mount in place, both `/health` and `POST /search` with `"source": "Semantic Search"` work against a mounted local index.

## CI

The repo includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

The CI workflow installs a lightweight dependency set and runs the mock-driven `pytest` suite. It does not perform real Gemini calls or require the full production index.

## Current Limitations

These items are currently outside the routine local and CI validation path:

- real Gemini token-bearing validation
- observing CI on a live GitHub Actions runner after push
- final standalone hosted HF Space reuse strategy
- Kaggle workflow cleanup
