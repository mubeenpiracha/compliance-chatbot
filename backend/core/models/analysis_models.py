"""
Data models for analysis and response generation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from .retrieval_models import DocumentSource, EntityDefinition, Exception


class Definition(BaseModel):
    term: str
    regulatory_definition: str
    customer_context: str
    synthesized_understanding: str
    source_documents: List[DocumentSource]
    ambiguities: List[str] = []


class ExceptionAnalysis(BaseModel):
    exception: Exception
    applicability_score: float = Field(ge=0.0, le=1.0)
    qualification_assessment: Dict[str, bool]  # criteria -> meets_criteria
    missing_requirements: List[str] = []
    recommendation: str


class Requirement(BaseModel):
    requirement_id: str
    description: str
    source_document: DocumentSource
    compliance_level: Literal["mandatory", "recommended", "optional"]
    related_domains: List[str] = []


class Conflict(BaseModel):
    conflict_type: Literal["requirement", "definition", "jurisdiction"]
    description: str
    conflicting_sources: List[DocumentSource]
    resolution_guidance: Optional[str] = None


class RiskArea(BaseModel):
    area_name: str
    risk_level: Literal["high", "medium", "low"]
    description: str
    potential_consequences: List[str]
    mitigation_strategies: List[str]
    regulatory_basis: List[DocumentSource]


class ComplianceAssessment(BaseModel):
    overall_risk_level: Literal["high", "medium", "low"]
    classification_likelihood: Dict[str, float]  # regulation_type -> probability
    compliance_gaps: List[str]
    strengths: List[str]
    recommendations: List[str]


class CriticalRequirement(BaseModel):
    requirement: Requirement
    criticality_score: float = Field(ge=0.0, le=1.0)
    deadline: Optional[str] = None
    consequences_of_non_compliance: List[str]


class MitigationStrategy(BaseModel):
    strategy_name: str
    description: str
    implementation_steps: List[str]
    effectiveness_rating: float = Field(ge=0.0, le=1.0)
    cost_assessment: Literal["low", "medium", "high"]


class RiskProfile(BaseModel):
    overall_assessment: ComplianceAssessment
    risk_areas: List[RiskArea]
    critical_requirements: List[CriticalRequirement]
    mitigation_strategies: List[MitigationStrategy]
    monitoring_recommendations: List[str]


class Citation(BaseModel):
    source: DocumentSource
    quoted_text: str
    page_reference: Optional[str] = None
    context: str
    citation_type: Literal["definition", "requirement", "exception", "guidance"]


class ActionItem(BaseModel):
    action: str
    priority: Literal["high", "medium", "low"]
    timeline: Optional[str] = None
    responsible_party: Optional[str] = None
    dependencies: List[str] = []


class StructuredResponse(BaseModel):
    executive_summary: str
    detailed_analysis: str
    regulatory_classification: Dict[str, Any]
    compliance_requirements: List[str]
    risk_assessment: str
    recommendations: List[str]
    action_items: List[ActionItem]
    citations: List[Citation]
    areas_of_uncertainty: List[str]
    follow_up_questions: List[str] = []


class QualityAssessment(BaseModel):
    completeness_score: float = Field(ge=0.0, le=1.0)
    accuracy_confidence: float = Field(ge=0.0, le=1.0)
    citation_quality: float = Field(ge=0.0, le=1.0)
    response_coherence: float = Field(ge=0.0, le=1.0)
    issues_identified: List[str] = []
    recommendations_for_improvement: List[str] = []


class FinalResponse(BaseModel):
    response_id: str
    structured_response: StructuredResponse
    quality_assessment: QualityAssessment
    processing_metadata: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
