from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PaperResponse(BaseModel):
    title: str
    abstract: str
    authors: str
    updated_display: str
    paper_url: str
    pdf_url: str
    source: str
    document_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    source: str | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchResponse(BaseModel):
    source_used: str
    papers: list[PaperResponse]
    search_results_markdown: str


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    context_results: int = Field(default=5, ge=1, le=20)
    search_source: str | None = None
    llm_model: str = Field(default="google/gemma-3-4b-it", min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class AskResponse(BaseModel):
    answer: str
    prompt: str
    source_used: str
    papers: list[PaperResponse]
    search_results_markdown: str


class HealthResponse(BaseModel):
    status: str
    has_gemini_api_key: bool
    has_hf_token: bool
    rag_source_configured: bool
    retriever_index_path: str
    retriever_index_present: bool
