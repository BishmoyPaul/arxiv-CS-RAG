from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from arxiv_cs_rag import AppConfig, GeminiGenerationService, RAGPipeline


SEARCH_RESULTS_HEADER = "# Search Results\n\n"
APP_HEADER_TEXT = (
    "# ArXiv CS RAG\n\n"
    "Local Gradio UI backed directly by the shared `arxiv_cs_rag` package."
)


@dataclass(frozen=True)
class SearchViewState:
    search_results_markdown: str
    prompt: str
    error_message: str | None = None
    warning_message: str | None = None


class GradioController:
    def __init__(
        self,
        config: AppConfig,
        pipeline: RAGPipeline,
        generation_service: GeminiGenerationService,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.generation_service = generation_service

    @property
    def llm_models_to_choose(self) -> tuple[str, ...]:
        return self.config.llm_models_to_choose

    @property
    def default_llm_model(self) -> str:
        return self.config.default_llm_model

    @property
    def default_retrieve_results(self) -> int:
        return self.config.retrieve_results

    @property
    def database_choices(self) -> list[str]:
        return self.pipeline.retrieval_service.get_database_choices()

    @property
    def default_database_choice(self) -> str:
        choices = self.database_choices
        return choices[0]

    def prepare_search(
        self,
        query: str,
        context_results: int,
        database_choice: str,
    ) -> SearchViewState:
        try:
            prepared = self.pipeline.prepare_answer(
                query=query,
                context_results=context_results,
                search_source=database_choice,
                top_k=self.default_retrieve_results,
            )
        except Exception as exc:
            return SearchViewState(
                search_results_markdown=SEARCH_RESULTS_HEADER,
                prompt="",
                error_message=f"An error occurred during search: {exc}",
            )

        warning_message = None
        retrieval_service = self.pipeline.retrieval_service
        if (
            self._is_semantic_selection(database_choice)
            and prepared.source_used != self.config.semantic_search_label
            and retrieval_service.retriever_load_error is not None
        ):
            warning_message = (
                "Semantic Search is unavailable in this environment. "
                f"Fell back to {prepared.source_used}. "
                f"Retriever load error: {retrieval_service.retriever_load_error}"
            )

        return SearchViewState(
            search_results_markdown=prepared.search_results_markdown,
            prompt=prepared.prompt,
            warning_message=warning_message,
        )

    def _is_semantic_selection(self, database_choice: str) -> bool:
        return self.config.semantic_search_label in database_choice

    def generate_answer(
        self,
        prompt: str,
        model_name: str,
        stream_outputs: bool,
    ) -> Iterator[str]:
        if stream_outputs:
            yield from self.generation_service.stream_answer(prompt, model_name)
            return

        yield self.generation_service.generate_answer(prompt, model_name)
