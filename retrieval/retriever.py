"""
Builds a genuine LangChain Retriever backed by the persisted ChromaDB
vector store, plus a small dataclass for formatting retrieved chunks
for API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from utils.config import settings
from utils.logger import logger
from utils.embeddings import LightweightEmbeddings


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata, for API responses."""

    text: str
    source: str          # e.g. "Pakistan Penal Code"
    page: int
    score: float          # similarity score, 0-1 (higher is better)

    @property
    def confidence_label(self) -> str:
        if self.score >= 0.75:
            return "Very High"
        elif self.score >= 0.55:
            return "High"
        elif self.score >= 0.35:
            return "Medium"
        return "Low"

    @property
    def confidence_pct(self) -> int:
        return min(100, max(0, int(self.score * 100)))

    @property
    def citation(self) -> str:
        return f"{self.source}, p.{self.page}"


class LawRetriever:
    """
    Wraps a LangChain VectorStoreRetriever with a similarity-score
    interface (LangChain's default retriever doesn't return scores),
    plus optional filtering by source document.
    """

    def __init__(self) -> None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embeddings = LightweightEmbeddings()

        self.vectordb = Chroma(
            persist_directory=str(settings.chroma_persist_path),
            embedding_function=self.embeddings,
            collection_name=settings.chroma_collection_name,
        )

    def as_langchain_retriever(self, top_k: Optional[int] = None) -> VectorStoreRetriever:
        """Return a real LangChain Retriever object (for use in LCEL chains)."""
        k = top_k or settings.top_k_results
        return self.vectordb.as_retriever(search_kwargs={"k": k})

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks with similarity scores attached, for display in
        the API response (source cards, confidence %, page references).
        """
        k = top_k or settings.top_k_results
        min_score = threshold if threshold is not None else settings.similarity_threshold

        filter_dict = {"source": source_filter} if source_filter else None

        # Chroma returns (Document, distance) pairs; distance is cosine
        # distance in [0, 2] when using cosine space — convert to a
        # 0-1 similarity score.
        results = self.vectordb.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )

        chunks: List[RetrievedChunk] = []
        for doc, score in results:
            if score < min_score:
                continue
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", 0),
                    score=max(0.0, min(1.0, score)),
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)
        logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}…'")
        return chunks

    def collection_count(self) -> int:
        try:
            return self.vectordb._collection.count()
        except Exception:
            return 0

    def available_sources(self) -> List[str]:
        """List distinct document sources currently in the collection."""
        try:
            data = self.vectordb._collection.get(include=["metadatas"])
            sources = {m.get("source") for m in data.get("metadatas", []) if m.get("source")}
            return sorted(sources)
        except Exception:
            return []