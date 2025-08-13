"""
Data models for retrieval and document processing.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from enum import Enum


class RetrievalQuery(BaseModel):
    query_text: str
    query_type: Literal["vector", "keyword", "regulatory", "entity", "fusion"]
    target_domains: List[str] = []
    max_results: int = 10
    min_relevance_score: float = 0.5


class DocumentSource(BaseModel):
    document_id: str
    title: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    effective_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    authority_level: int = Field(ge=1, le=4)  # 1=highest (law), 4=lowest (guidance)
    jurisdiction: str
    chunk_id: Optional[str] = None  # Unique chunk identifier for citation


class RetrievedDocument(BaseModel):
    source: DocumentSource
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieval_method: Literal["vector", "keyword", "regulatory", "entity", "fusion"]
    match_highlights: List[str] = []
    context_window: str = ""  # Broader context around the match


class HybridRetrievalResult(BaseModel):
    query: RetrievalQuery
    documents: List[RetrievedDocument]
    retrieval_stats: Dict[str, Any]  # method -> {count, avg_score, etc}
    coverage_assessment: float = Field(ge=0.0, le=1.0)
    fusion_confidence: float = Field(ge=0.0, le=1.0)


class EntityDefinition(BaseModel):
    entity_name: str
    regulatory_definition: str
    source_documents: List[DocumentSource]
    key_criteria: List[str] = []
    exclusions: List[str] = []
    related_entities: List[str] = []


class Exception(BaseModel):
    exception_name: str
    description: str
    qualification_criteria: List[str]
    source_documents: List[DocumentSource]
    applicability_assessment: Optional[float] = None  # 0-1 likelihood of applicability
