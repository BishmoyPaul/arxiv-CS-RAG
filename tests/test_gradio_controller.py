from __future__ import annotations

from arxiv_cs_rag.config import AppConfig
from arxiv_cs_rag.pipeline import PreparedAnswer
from arxiv_cs_rag.retrieval import PaperRecord, RetrievalService
from app.gradio_controller import GradioController, SEARCH_RESULTS_HEADER


def build_paper(title: str) -> PaperRecord:
    return PaperRecord(
        title=title,
        abstract=f"{title} abstract",
        authors="Author One",
        updated_display="10 Mar 2026",
        paper_url=f"https://arxiv.org/abs/{title.lower().replace(' ', '-')}",
        pdf_url=f"https://arxiv.org/pdf/{title.lower().replace(' ', '-')}",
        source="Semantic Search",
    )


class StubPipeline:
    def __init__(self) -> None:
        self.retrieval_service = RetrievalService(config=AppConfig())

    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        return PreparedAnswer(
            query=query,
            prompt=f"Prompt for: {query}",
            source_used="Semantic Search",
            papers=[build_paper("Paper One")],
            search_results_markdown="# Search Results\n\n### Paper One",
        )


class FailingPipeline:
    def __init__(self) -> None:
        self.retrieval_service = RetrievalService(config=AppConfig())

    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        raise RuntimeError("backend offline")


class FallbackPipeline:
    def __init__(self) -> None:
        self.retrieval_service = RetrievalService(config=AppConfig())
        self.retrieval_service._retriever_load_error = RuntimeError("compiler not available")

    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        return PreparedAnswer(
            query=query,
            prompt=f"Prompt for: {query}",
            source_used="Arxiv Search - Latest - (EXPERIMENTAL)",
            papers=[build_paper("Paper One")],
            search_results_markdown="# Search Results\n\n### Paper One",
        )


class StubGenerationService:
    def generate_answer(self, prompt: str, model_name: str) -> str:
        return f"{model_name}: {prompt}"

    def stream_answer(self, prompt: str, model_name: str):
        yield f"{model_name}: {prompt[:10]}"
        yield f"{model_name}: {prompt}"


def test_prepare_search_returns_markdown_and_prompt():
    controller = GradioController(
        config=AppConfig(),
        pipeline=StubPipeline(),
        generation_service=StubGenerationService(),
    )

    state = controller.prepare_search("transformers", context_results=5, database_choice="Semantic Search")

    assert state.search_results_markdown == "# Search Results\n\n### Paper One"
    assert state.prompt == "Prompt for: transformers"
    assert state.error_message is None


def test_prepare_search_returns_error_state_on_failure():
    controller = GradioController(
        config=AppConfig(),
        pipeline=FailingPipeline(),
        generation_service=StubGenerationService(),
    )

    state = controller.prepare_search("transformers", context_results=5, database_choice="Semantic Search")

    assert state.search_results_markdown == SEARCH_RESULTS_HEADER
    assert state.prompt == ""
    assert state.error_message == "An error occurred during search: backend offline"


def test_prepare_search_returns_warning_when_semantic_search_falls_back():
    controller = GradioController(
        config=AppConfig(),
        pipeline=FallbackPipeline(),
        generation_service=StubGenerationService(),
    )

    state = controller.prepare_search("transformers", context_results=5, database_choice="Semantic Search")

    assert state.error_message is None
    assert state.warning_message is not None
    assert "Semantic Search is unavailable in this environment." in state.warning_message
    assert "compiler not available" in state.warning_message


def test_generate_answer_returns_single_message_when_streaming_disabled():
    controller = GradioController(
        config=AppConfig(),
        pipeline=StubPipeline(),
        generation_service=StubGenerationService(),
    )

    chunks = list(
        controller.generate_answer(
            prompt="Prompt for: transformers",
            model_name="mock-model",
            stream_outputs=False,
        )
    )

    assert chunks == ["mock-model: Prompt for: transformers"]


def test_generate_answer_streams_chunks_when_enabled():
    controller = GradioController(
        config=AppConfig(),
        pipeline=StubPipeline(),
        generation_service=StubGenerationService(),
    )

    chunks = list(
        controller.generate_answer(
            prompt="Prompt for: transformers",
            model_name="mock-model",
            stream_outputs=True,
        )
    )

    assert chunks == [
        "mock-model: Prompt for",
        "mock-model: Prompt for: transformers",
    ]
