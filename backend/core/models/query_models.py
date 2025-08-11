"""
Data models for query processing and analysis.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class QueryClassification(BaseModel):
    is_compliance: bool
    complexity_level: QueryComplexity
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    detected_domains: List[str] = []


class RegulatoryContext(BaseModel):
    jurisdiction: str
    regulatory_domains: List[str]
    document_priority: List[str]  # Laws > Regulations > Rules > Guidance
    applicable_frameworks: List[str] = []


class SubQuestion(BaseModel):
    question: str
    question_type: Literal["definition", "requirement", "comparison", "analysis"]
    priority: int = Field(ge=1, le=10)  # 1 = highest priority
    entities_involved: List[str] = []
    regulatory_domains: List[str] = []


class QueryDecomposition(BaseModel):
    sub_questions: List[SubQuestion]
    entity_definitions_needed: List[str]
    regulatory_requirements_to_check: List[str]
    analysis_required: List[str]
    dependencies: Dict[str, List[str]] = {}  # question_id -> dependent_question_ids


class ClarificationQuestion(BaseModel):
    question: str
    context: str
    question_type: Literal["definition", "scenario", "preference", "constraint"]
    suggested_answers: List[str] = []
    is_required: bool = True


class KnowledgeGap(BaseModel):
    retrieval_queries: List[str]
    clarification_questions: List[ClarificationQuestion]
    question_dependencies: Dict[str, List[str]]
    coverage_assessment: Dict[str, bool]  # sub_question_id -> can_be_answered


class CustomerClarification(BaseModel):
    question_id: str
    user_response: str
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)


class ProcessingState(BaseModel):
    """Tracks the current state of query processing"""
    current_node: str
    completed_nodes: List[str] = []
    pending_clarifications: List[ClarificationQuestion] = []
    collected_clarifications: List[CustomerClarification] = []
    intermediate_results: Dict[str, Any] = {}
    requires_user_input: bool = False
