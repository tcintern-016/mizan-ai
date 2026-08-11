"""
API routes for the Pakistan Law Assistant.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.models import (
    AskRequest,
    AskResponse,
    SourceCitation,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
)
from retrieval.rag_chain import LawRAGChain
from retrieval.retriever import LawRetriever
from utils.config import settings
from utils.logger import logger

router = APIRouter()


@lru_cache(maxsize=1)
def get_rag_chain() -> LawRAGChain:
    """Build the RAG chain once and reuse it across requests."""
    return LawRAGChain()


@lru_cache(maxsize=1)
def get_retriever() -> LawRetriever:
    """Build the retriever once and reuse it across requests (for /search, /health)."""
    return LawRetriever()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    try:
        retriever = get_retriever()
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            version=settings.app_version,
            groq_configured=bool(settings.groq_api_key),
            chunk_count=retriever.collection_count(),
            available_sources=retriever.available_sources(),
        )
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return HealthResponse(
            status="degraded",
            app_name=settings.app_name,
            version=settings.app_version,
            groq_configured=bool(settings.groq_api_key),
            chunk_count=0,
            available_sources=[],
        )


@router.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(payload: AskRequest):
    try:
        chain = get_rag_chain()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result = chain.ask(
        question=payload.question,
        top_k=payload.top_k,
        source_filter=payload.source_filter,
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
    )


@router.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(payload: SearchRequest):
    retriever = get_retriever()
    chunks = retriever.retrieve_with_scores(
        payload.query, top_k=payload.top_k, source_filter=payload.source_filter
    )
    return SearchResponse(
        results=[
            SearchResult(
                citation=c.citation,
                source=c.source,
                page=c.page,
                confidence=c.confidence_pct,
                confidence_label=c.confidence_label,
                text=c.text,
            )
            for c in chunks
        ]
    )
