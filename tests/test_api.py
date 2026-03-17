from pathlib import Path
import shutil

from fastapi.testclient import TestClient

from app.dependencies import get_config, get_pipeline
from app.main import app
from arxiv_cs_rag.config import AppConfig
from arxiv_cs_rag.pipeline import PreparedAnswer
from arxiv_cs_rag.retrieval import PaperRecord


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
    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        papers = [build_paper("Paper One"), build_paper("Paper Two")]
        return PreparedAnswer(
            query=query,
            prompt=f"Prompt for: {query}",
            source_used="Semantic Search",
            papers=papers[: top_k or len(papers)],
            search_results_markdown="# Search Results\n\n### Paper One",
        )

    def ask(
        self,
        query: str,
        context_results: int,
        model_name: str,
        search_source: str | None = None,
        top_k: int | None = None,
    ):
        prepared = self.prepare_answer(
            query=query,
            context_results=context_results,
            search_source=search_source,
            top_k=top_k,
        )
        return prepared, f"Answer for: {query}"


class PreparedOnlyPipeline:
    def ask(
        self,
        query: str,
        context_results: int,
        model_name: str,
        search_source: str | None = None,
        top_k: int | None = None,
    ):
        return PreparedAnswer(
            query=query,
            prompt=f"Prompt for: {query}",
            source_used="Semantic Search",
            papers=[build_paper("Paper One")],
            search_results_markdown="# Search Results\n\n### Paper One",
        )


class FailingPipeline:
    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        raise RuntimeError("backend offline")

    def ask(
        self,
        query: str,
        context_results: int,
        model_name: str,
        search_source: str | None = None,
        top_k: int | None = None,
    ):
        raise RuntimeError("backend offline")


class BadInputPipeline:
    def prepare_answer(
        self,
        query: str,
        context_results: int,
        search_source: str | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        raise ValueError("Invalid search request.")

    def ask(
        self,
        query: str,
        context_results: int,
        model_name: str,
        search_source: str | None = None,
        top_k: int | None = None,
    ):
        raise ValueError("Invalid ask request.")


def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_reflects_config_flags():
    test_root = Path("tests/.tmp-health")
    if test_root.exists():
        shutil.rmtree(test_root)
    index_dir = test_root / "rag_index_data" / "arxiv_colbert"
    index_dir.mkdir(parents=True)
    app.dependency_overrides[get_config] = lambda: AppConfig(
        hf_token="hf-token",
        gemini_api_key="gemini-key",
        rag_source="owner/index-dataset",
        local_data_dir=test_root / "rag_index_data",
    )
    client = TestClient(app)

    response = client.get("/health")

    app.dependency_overrides.clear()
    payload = response.json()
    shutil.rmtree(test_root)
    assert response.status_code == 200
    assert payload["has_gemini_api_key"] is True
    assert payload["has_hf_token"] is True
    assert payload["rag_source_configured"] is True
    assert payload["retriever_index_present"] is True


def test_search_endpoint_returns_normalized_papers():
    app.dependency_overrides[get_pipeline] = lambda: StubPipeline()
    client = TestClient(app)

    response = client.post("/search", json={"query": "transformers", "top_k": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_used"] == "Semantic Search"
    assert len(payload["papers"]) == 2
    assert payload["papers"][0]["title"] == "Paper One"


def test_search_endpoint_rejects_invalid_top_k():
    client = TestClient(app)

    response = client.post("/search", json={"query": "transformers", "top_k": 0})

    assert response.status_code == 422


def test_search_endpoint_maps_value_errors_to_400():
    app.dependency_overrides[get_pipeline] = lambda: BadInputPipeline()
    client = TestClient(app)

    response = client.post("/search", json={"query": "transformers", "top_k": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid search request."


def test_search_endpoint_maps_runtime_failures_to_503():
    app.dependency_overrides[get_pipeline] = lambda: FailingPipeline()
    client = TestClient(app)

    response = client.post("/search", json={"query": "transformers", "top_k": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable."


def test_ask_endpoint_returns_answer_and_prompt():
    app.dependency_overrides[get_pipeline] = lambda: StubPipeline()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "What is retrieval?",
            "top_k": 2,
            "context_results": 1,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for: What is retrieval?"
    assert payload["prompt"] == "Prompt for: What is retrieval?"
    assert payload["source_used"] == "Semantic Search"


def test_ask_endpoint_returns_placeholder_when_generation_is_not_configured():
    app.dependency_overrides[get_pipeline] = lambda: PreparedOnlyPipeline()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "What is retrieval?",
            "top_k": 1,
            "context_results": 1,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    app.dependency_overrides.clear()
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"] == "Generation service is not configured."


def test_ask_endpoint_rejects_invalid_context_results():
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "What is retrieval?",
            "top_k": 2,
            "context_results": 0,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_maps_value_errors_to_400():
    app.dependency_overrides[get_pipeline] = lambda: BadInputPipeline()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "What is retrieval?",
            "top_k": 2,
            "context_results": 1,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ask request."


def test_ask_endpoint_maps_runtime_failures_to_503():
    app.dependency_overrides[get_pipeline] = lambda: FailingPipeline()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "What is retrieval?",
            "top_k": 2,
            "context_results": 1,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable."


def test_ask_endpoint_rejects_blank_query():
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "query": "   ",
            "top_k": 2,
            "context_results": 1,
            "llm_model": "google/gemma-3-4b-it",
        },
    )

    assert response.status_code == 422
