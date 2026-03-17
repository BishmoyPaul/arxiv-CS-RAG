from pathlib import Path

from app import dependencies
from arxiv_cs_rag.config import AppConfig


def test_app_config_from_env_reads_expected_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("RAG_SOURCE", "owner/index-dataset")

    config = AppConfig.from_env()

    assert config.hf_token == "hf-test-token"
    assert config.gemini_api_key == "gemini-test-key"
    assert config.rag_source == "owner/index-dataset"
    assert config.retriever_index_path == Path("rag_index_data") / "arxiv_colbert"


def test_dependency_config_cache_can_be_refreshed(monkeypatch):
    dependencies.get_config.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "first-token")
    first = dependencies.get_config()

    monkeypatch.setenv("HF_TOKEN", "second-token")
    cached = dependencies.get_config()

    dependencies.get_config.cache_clear()
    refreshed = dependencies.get_config()
    dependencies.get_retrieval_service.cache_clear()
    dependencies.get_generation_service.cache_clear()

    assert first.hf_token == "first-token"
    assert cached.hf_token == "first-token"
    assert refreshed.hf_token == "second-token"


def test_dependency_pipeline_builds_without_eager_runtime_initialization(monkeypatch):
    dependencies.get_config.cache_clear()
    dependencies.get_retrieval_service.cache_clear()
    dependencies.get_generation_service.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_SOURCE", raising=False)

    pipeline = dependencies.get_pipeline()

    assert pipeline.retrieval_service.config.gemini_api_key is None
    assert pipeline.generation_service is not None
