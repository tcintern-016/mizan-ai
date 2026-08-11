"""
Pydantic request/response models for the Pakistan Law Assistant API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The legal question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=15)
    source_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to one document, e.g. 'Pakistan Penal Code'",
    )


class SourceCitation(BaseModel):
    citation: str
    source: str
    page: int
    confidence: int
    confidence_label: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    disclaimer: str = (
        "This response is for educational and informational purposes only and "
        "does not constitute legal advice. For any real legal matter, please "
        "consult a licensed lawyer or legal professional in Pakistan."
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: Optional[int] = Field(default=10, ge=1, le=20)
    source_filter: Optional[str] = None


class SearchResult(BaseModel):
    citation: str
    source: str
    page: int
    confidence: int
    confidence_label: str
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    groq_configured: bool
    chunk_count: int
    available_sources: List[str]
