from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


DEFAULT_LLM_MODELS = (
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "None",
)

DEFAULT_ARXIV_CATEGORY_FILTER = (
    "(cat:cs.CV OR cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.NE OR cat:cs.RO)"
)


@dataclass(frozen=True)
class GeminiGenerationSettings:
    temperature: float = 0.2
    max_output_tokens: int = 450
    top_p: float = 0.95


@dataclass(frozen=True)
class AppConfig:
    hf_token: str | None = None
    gemini_api_key: str | None = None
    rag_source: str | None = None
    local_data_dir: Path = Path("./rag_index_data")
    retriever_dir_name: str = "arxiv_colbert"
    retrieve_results: int = 20
    default_llm_model: str = "google/gemma-3-4b-it"
    llm_models_to_choose: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LLM_MODELS)
    semantic_search_label: str = "Semantic Search"
    live_search_label: str = "Arxiv Search - Latest - (EXPERIMENTAL)"
    generation: GeminiGenerationSettings = field(default_factory=GeminiGenerationSettings)
    arxiv_category_filter: str = DEFAULT_ARXIV_CATEGORY_FILTER

    @property
    def retriever_index_path(self) -> Path:
        return self.local_data_dir / self.retriever_dir_name

    @classmethod
    def from_env(cls) -> "AppConfig":
        local_data_dir = Path(os.getenv("LOCAL_DATA_DIR", "./rag_index_data"))
        return cls(
            hf_token=os.getenv("HF_TOKEN"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            rag_source=os.getenv("RAG_SOURCE"),
            local_data_dir=local_data_dir,
        )
