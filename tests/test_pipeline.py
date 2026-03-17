from arxiv_cs_rag.pipeline import RAGPipeline, build_prompt_text
from arxiv_cs_rag.retrieval import PaperRecord


class StubRetrievalService:
    def search(self, query: str, top_k: int | None = None, source: str | None = None):
        return "Semantic Search", [
            PaperRecord(
                title="Paper One",
                abstract="First abstract",
                authors="Author One",
                updated_display="10 Mar 2026",
                paper_url="https://arxiv.org/abs/1111.1111",
                pdf_url="https://arxiv.org/pdf/1111.1111",
                source="Semantic Search",
            ),
            PaperRecord(
                title="Paper Two",
                abstract="Second abstract",
                authors="Author Two",
                updated_display="10 Mar 2026",
                paper_url="https://arxiv.org/abs/2222.2222",
                pdf_url="https://arxiv.org/pdf/2222.2222",
                source="Semantic Search",
            ),
        ]


class EmptyRetrievalService:
    def search(self, query: str, top_k: int | None = None, source: str | None = None):
        return "Semantic Search", []


class StubGenerationService:
    def generate_answer(self, prompt: str, model_name: str) -> str:
        return f"generated:{model_name}:{prompt}"


def test_build_prompt_text_contains_instruction_and_question():
    prompt = build_prompt_text("What is retrieval?", "1. Context")

    assert "Question: What is retrieval?" in prompt
    assert "scientific paper abstracts" in prompt


def test_prepare_answer_limits_prompt_context_to_requested_count():
    pipeline = RAGPipeline(retrieval_service=StubRetrievalService())

    prepared = pipeline.prepare_answer(query="Explain transformers", context_results=1)

    assert prepared.source_used == "Semantic Search"
    assert "Paper One" in prepared.prompt
    assert "Paper Two" not in prepared.prompt
    assert "Paper Two" in prepared.search_results_markdown


def test_prepare_answer_handles_zero_papers():
    pipeline = RAGPipeline(retrieval_service=EmptyRetrievalService())

    prepared = pipeline.prepare_answer(query="Explain nothing", context_results=3)

    assert prepared.papers == []
    assert "Question: Explain nothing" in prepared.prompt
    assert prepared.search_results_markdown.startswith("# Search Results")


def test_ask_returns_prepared_answer_when_generation_service_is_missing():
    pipeline = RAGPipeline(retrieval_service=StubRetrievalService(), generation_service=None)

    result = pipeline.ask(query="Explain transformers", context_results=1, model_name="google/gemma-3-4b-it")

    assert not isinstance(result, tuple)
    assert result.source_used == "Semantic Search"


def test_ask_uses_generation_service_when_configured():
    pipeline = RAGPipeline(
        retrieval_service=StubRetrievalService(),
        generation_service=StubGenerationService(),
    )

    prepared, answer = pipeline.ask(
        query="Explain transformers",
        context_results=1,
        model_name="mock-model",
    )

    assert prepared.source_used == "Semantic Search"
    assert answer.startswith("generated:mock-model:")
