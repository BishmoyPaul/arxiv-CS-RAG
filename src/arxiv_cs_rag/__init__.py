"""Core package for the ArXiv CS RAG project."""

from .config import AppConfig
from .generation import GeminiGenerationService
from .pipeline import PreparedAnswer, RAGPipeline, build_prompt_text
from .retrieval import PaperRecord, RetrievalService

__all__ = [
    "AppConfig",
    "GeminiGenerationService",
    "PaperRecord",
    "PreparedAnswer",
    "RAGPipeline",
    "RetrievalService",
    "build_prompt_text",
]
