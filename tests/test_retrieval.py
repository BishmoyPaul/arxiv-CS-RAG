from pathlib import Path

from arxiv_cs_rag.config import AppConfig
from arxiv_cs_rag.retrieval import RetrievalService, remove_punctuation, remove_stopwords, search_cleaner


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query: str, k: int):
        return self.results[:k]


def build_config() -> AppConfig:
    return AppConfig(
        local_data_dir=Path("./missing-index"),
        semantic_search_label="Semantic Search",
        live_search_label="Arxiv Search - Latest - (EXPERIMENTAL)",
    )


def test_search_cleaner_removes_common_stopwords_and_punctuation():
    cleaned = search_cleaner("What, exactly, is the role of retrieval?")

    assert cleaned == "what exactly role retrieval"
    assert remove_punctuation("paper's, test!") == "paper's test"
    assert remove_stopwords("this is a paper", {"this", "is", "a"}) == "paper"


def test_retrieval_rejects_blank_query():
    service = RetrievalService(config=build_config())

    try:
        service.search("   ")
    except ValueError as exc:
        assert str(exc) == "Query must not be empty."
    else:
        raise AssertionError("Expected ValueError for blank query")


def test_retrieval_exposes_database_choices():
    service = RetrievalService(config=build_config())

    assert service.get_database_choices() == [
        "Semantic Search",
        "Arxiv Search - Latest - (EXPERIMENTAL)",
    ]


def test_retrieval_normalizes_semantic_display_label():
    service = RetrievalService(config=build_config())

    assert service._normalize_source_label("Semantic Search - up to 10 Mar 2026") == "Semantic Search"
    assert service._normalize_source_label("Arxiv Search - Latest - (EXPERIMENTAL)") == "Arxiv Search - Latest - (EXPERIMENTAL)"


def test_retrieval_falls_back_to_live_search_when_semantic_index_is_unavailable():
    class LiveOnlyService(RetrievalService):
        def _get_retriever(self):
            return None

        def _live_search(self, query: str, top_k: int):
            return []

    service = LiveOnlyService(config=build_config())

    source_used, papers = service.search("graph neural networks")

    assert source_used == "Arxiv Search - Latest - (EXPERIMENTAL)"
    assert papers == []


def test_retrieval_falls_back_to_live_search_when_retriever_init_fails():
    class BrokenRetrieverInitService(RetrievalService):
        def _ensure_local_index(self):
            return Path("./present-index")

        def _live_search(self, query: str, top_k: int):
            return []

    service = BrokenRetrieverInitService(config=build_config())

    source_used, papers = service.search("graph neural networks")

    assert source_used == "Arxiv Search - Latest - (EXPERIMENTAL)"
    assert papers == []
    assert service.retriever_load_error is not None


def test_retrieval_falls_back_to_semantic_search_when_live_search_errors():
    semantic_results = [
        {
            "document_metadata": {
                "title": "Semantic Title",
                "_time": "10 Mar 2026",
                "authors": "Author One",
            },
            "content": "Abstract from semantic search",
            "document_id": "1234.5678",
        }
    ]

    class BrokenLiveSearchService(RetrievalService):
        def _live_search(self, query: str, top_k: int):
            raise RuntimeError("live search failed")

    service = BrokenLiveSearchService(config=build_config(), retriever=FakeRetriever(semantic_results))

    source_used, papers = service.search(
        "multimodal learning",
        source="Arxiv Search - Latest - (EXPERIMENTAL)",
    )

    assert source_used == "Semantic Search"
    assert len(papers) == 1
    assert papers[0].title == "Semantic Title"
