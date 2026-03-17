from __future__ import annotations

from functools import lru_cache

from arxiv_cs_rag import AppConfig, GeminiGenerationService, RAGPipeline, RetrievalService


@lru_cache
def get_config() -> AppConfig:
    return AppConfig.from_env()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(config=get_config())


@lru_cache
def get_generation_service() -> GeminiGenerationService:
    return GeminiGenerationService(config=get_config())


def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retrieval_service=get_retrieval_service(),
        generation_service=get_generation_service(),
    )
