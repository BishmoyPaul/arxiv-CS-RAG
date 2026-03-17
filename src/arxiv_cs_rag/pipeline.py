from __future__ import annotations

from dataclasses import dataclass

from .formatters import format_prompt_entry, format_search_results_markdown
from .generation import GeminiGenerationService
from .retrieval import PaperRecord, RetrievalService


SYSTEM_INSTRUCTION = (
    "Based on the provided scientific paper abstracts, provide a comprehensive answer "
    "of 6-7 lines. Synthesize information from multiple sources if possible. "
    "Your answer must be grounded in the details found in the abstracts. "
    "Cite the titles of the papers you use as sources in your answer."
)


@dataclass(frozen=True)
class PreparedAnswer:
    query: str
    prompt: str
    source_used: str
    papers: list[PaperRecord]
    search_results_markdown: str


def build_prompt_text(question: str, context: str) -> str:
    message = f"Abstracts:\n{context}\n\nQuestion: {question}"
    return f"{SYSTEM_INSTRUCTION}\n\n{message}"


class RAGPipeline:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GeminiGenerationService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service

    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        source_used, papers = self.retrieval_service.search(
            query=query,
            top_k=top_k,
            source=search_source,
        )
        prompt_context = self._build_prompt_context(papers, context_results)
        return PreparedAnswer(
            query=query,
            prompt=build_prompt_text(query, prompt_context),
            source_used=source_used,
            papers=papers,
            search_results_markdown=format_search_results_markdown(papers),
        )

    def ask(
        self,
        query: str,
        context_results: int,
        model_name: str,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer | tuple[PreparedAnswer, str]:
        prepared = self.prepare_answer(
            query=query,
            context_results=context_results,
            search_source=search_source,
            top_k=top_k,
        )
        if self.generation_service is None:
            return prepared
        answer = self.generation_service.generate_answer(prepared.prompt, model_name)
        return prepared, answer

    @staticmethod
    def _build_prompt_context(papers: list[PaperRecord], context_results: int) -> str:
        return "\n".join(
            f"{index}. {format_prompt_entry(paper)}"
            for index, paper in enumerate(papers[:context_results], start=1)
        )
