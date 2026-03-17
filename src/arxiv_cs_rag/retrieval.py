from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
from typing import Iterable

from .config import AppConfig


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "to",
        "was",
        "will",
        "with",
    }
)


@dataclass(frozen=True)
class PaperRecord:
    title: str
    abstract: str
    authors: str
    updated_display: str
    paper_url: str
    pdf_url: str
    source: str
    document_id: str | None = None


class RetrievalService:
    def __init__(
        self,
        config: AppConfig,
        retriever: object | None = None,
        arxiv_client: object | None = None,
    ) -> None:
        self.config = config
        self._retriever = retriever
        self._arxiv_client = arxiv_client
        self._retriever_load_error: Exception | None = None

    def get_database_choices(self) -> list[str]:
        return [self.config.semantic_search_label, self.config.live_search_label]

    @property
    def retriever_load_error(self) -> Exception | None:
        return self._retriever_load_error

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source: str | None = None,
    ) -> tuple[str, list[PaperRecord]]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty.")

        requested_source = self._normalize_source_label(source)
        limit = top_k or self.config.retrieve_results

        if requested_source == self.config.semantic_search_label:
            retriever = self._get_retriever()
            if retriever is not None:
                return self.config.semantic_search_label, self._semantic_search(clean_query, limit)
            return self.config.live_search_label, self._live_search(clean_query, limit)

        try:
            live_results = self._live_search(clean_query, limit)
            if live_results:
                return self.config.live_search_label, live_results
        except Exception:
            live_results = []

        retriever = self._get_retriever()
        if retriever is None:
            return self.config.live_search_label, live_results

        return self.config.semantic_search_label, self._semantic_search(clean_query, limit)

    def _normalize_source_label(self, source: str | None) -> str:
        if not source:
            return self.config.semantic_search_label

        if self.config.semantic_search_label in source:
            return self.config.semantic_search_label

        if self.config.live_search_label in source or "Arxiv Search" in source:
            return self.config.live_search_label

        return source

    def _get_retriever(self) -> object | None:
        if self._retriever is not None:
            return self._retriever

        index_path = self._ensure_local_index()
        if index_path is None:
            return None

        try:
            from ragatouille import RAGPretrainedModel

            self._retriever = RAGPretrainedModel.from_index(str(index_path))
            self._retriever_load_error = None
            return self._retriever
        except Exception as exc:
            self._retriever_load_error = exc
            return None

    def _ensure_local_index(self) -> Path | None:
        index_path = self.config.retriever_index_path
        if index_path.exists():
            return index_path

        if not self.config.rag_source or not self.config.hf_token:
            return None

        from huggingface_hub import snapshot_download

        self.config.local_data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.config.rag_source,
            repo_type="dataset",
            token=self.config.hf_token,
            local_dir=str(self.config.local_data_dir),
        )
        return index_path if index_path.exists() else None

    def _semantic_search(self, query: str, top_k: int) -> list[PaperRecord]:
        retriever = self._get_retriever()
        if retriever is None:
            return []

        raw_results = retriever.search(query, k=top_k)
        return [self._paper_from_semantic_result(item) for item in raw_results]

    def _live_search(self, query: str, top_k: int) -> list[PaperRecord]:
        client = self._get_arxiv_client()
        search_query = f"{search_cleaner(query)} AND {self.config.arxiv_category_filter}"

        import arxiv

        search = arxiv.Search(
            query=search_query,
            max_results=top_k,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return [self._paper_from_arxiv_result(item) for item in client.results(search)]

    def _get_arxiv_client(self) -> object:
        if self._arxiv_client is not None:
            return self._arxiv_client

        import arxiv

        self._arxiv_client = arxiv.Client()
        return self._arxiv_client

    def _paper_from_semantic_result(self, rag_answer: dict) -> PaperRecord:
        metadata = rag_answer["document_metadata"]
        document_id = rag_answer["document_id"]
        return PaperRecord(
            title=metadata["title"].replace("\n", ""),
            abstract=rag_answer["content"].strip(),
            authors=metadata["authors"].replace("\n", ""),
            updated_display=metadata["_time"],
            paper_url=f"https://arxiv.org/abs/{document_id}",
            pdf_url=f"https://arxiv.org/pdf/{document_id}",
            source=self.config.semantic_search_label,
            document_id=document_id,
        )

    def _paper_from_arxiv_result(self, result: object) -> PaperRecord:
        links = list(getattr(result, "links", []))
        paper_url = links[0].href if links else ""
        pdf_url = links[1].href if len(links) > 1 else paper_url
        authors = ", ".join(author.name for author in result.authors)

        return PaperRecord(
            title=result.title,
            abstract=result.summary.replace("\n", " ").strip(),
            authors=authors,
            updated_display=result.updated.strftime("%d %b %Y"),
            paper_url=paper_url,
            pdf_url=pdf_url,
            source=self.config.live_search_label,
            document_id=getattr(result, "entry_id", None),
        )


def remove_punctuation(text: str) -> str:
    punct_str = string.punctuation.replace("'", "")
    return text.translate(str.maketrans("", "", punct_str))


def get_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords
    except ImportError:
        return DEFAULT_STOPWORDS

    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        return DEFAULT_STOPWORDS


def remove_stopwords(text: str, stopwords_set: Iterable[str] | None = None) -> str:
    words = text.split()
    active_stopwords = set(stopwords_set or get_stopwords())
    return " ".join(word for word in words if word not in active_stopwords)


def search_cleaner(text: str) -> str:
    lowered = text.lower().strip()
    no_stopwords = remove_stopwords(lowered)
    return remove_punctuation(no_stopwords)
