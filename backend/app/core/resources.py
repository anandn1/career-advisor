"""Shared, lazily created AI components.

Each factory builds its object on first use and caches it, so importing an
agent module stays cheap and needs no credentials.
"""

import logging
from functools import cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.embeddings import GemmaOnnxEmbeddings

logger = logging.getLogger(__name__)

RERANK_MODEL = "rerank-english-v3.0"


@cache
def get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )


RESUME_MAX_TOKENS = 6000


@cache
def get_resume_llm() -> ChatGroq:
    """Larger model with a long output budget, for full LaTeX documents."""
    settings = get_settings()
    return ChatGroq(
        model=settings.resume_llm_model,
        temperature=0.1,
        max_tokens=RESUME_MAX_TOKENS,
        api_key=settings.groq_api_key,
    )


@cache
def get_embeddings() -> GemmaOnnxEmbeddings:
    model_dir = get_settings().embedding_model_dir
    logger.info("Loading the embedding model from %s", model_dir)
    return GemmaOnnxEmbeddings.load(model_dir)


@cache
def get_chroma(persist_dir: Path) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(persist_directory=str(persist_dir), embedding_function=get_embeddings())


def get_reranker(top_n: int) -> BaseDocumentCompressor | None:
    """Cohere reranker, or None when no Cohere key is configured."""
    api_key = get_settings().cohere_api_key
    if api_key is None:
        return None
    from langchain_cohere import CohereRerank

    return CohereRerank(cohere_api_key=api_key, model=RERANK_MODEL, top_n=top_n)


@cache
def get_web_search() -> BaseTool | None:
    """Tavily search tool, or None when no Tavily key is configured."""
    api_key = get_settings().tavily_api_key
    if api_key is None:
        return None
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=5, tavily_api_key=api_key.get_secret_value())
