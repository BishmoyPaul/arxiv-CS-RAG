from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException

from arxiv_cs_rag import AppConfig, PaperRecord, RAGPipeline
from .dependencies import get_config, get_pipeline
from .schemas import AskRequest, AskResponse, HealthResponse, PaperResponse, SearchRequest, SearchResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArXiv CS RAG API",
        version="0.2.0",
        description="FastAPI surface for the extracted ArXiv CS RAG core package.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health(config: AppConfig = Depends(get_config)) -> HealthResponse:
        index_path = config.retriever_index_path
        return HealthResponse(
            status="ok",
            has_gemini_api_key=bool(config.gemini_api_key),
            has_hf_token=bool(config.hf_token),
            rag_source_configured=bool(config.rag_source),
            retriever_index_path=str(index_path),
            retriever_index_present=index_path.exists(),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
        prepared = _run_service_call(
            lambda: pipeline.prepare_answer(
                query=request.query,
                context_results=min(request.top_k, 20),
                search_source=request.source,
                top_k=request.top_k,
            )
        )
        return SearchResponse(
            source_used=prepared.source_used,
            papers=_paper_responses(prepared.papers),
            search_results_markdown=prepared.search_results_markdown,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
        result = _run_service_call(
            lambda: pipeline.ask(
                query=request.query,
                context_results=request.context_results,
                model_name=request.llm_model,
                search_source=request.search_source,
                top_k=request.top_k,
            )
        )
        if isinstance(result, tuple):
            prepared, answer = result
        else:
            prepared = result
            answer = "Generation service is not configured."
        return AskResponse(
            answer=answer,
            prompt=prepared.prompt,
            source_used=prepared.source_used,
            papers=_paper_responses(prepared.papers),
            search_results_markdown=prepared.search_results_markdown,
        )

    return app


def _paper_responses(papers: list[PaperRecord]) -> list[PaperResponse]:
    return [PaperResponse(**paper.__dict__) for paper in papers]


def _run_service_call(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.") from exc


app = create_app()
