"""
Compliance Classification Node - Determines if query is compliance-related.
"""
import re
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from .base_node import ConditionalNode
from ..models.query_models import ProcessingState, QueryClassification, QueryComplexity


class ComplianceClassificationNode(ConditionalNode):
    """Classifies queries as compliance-related or general conversation."""
    
    # Legal/regulatory keywords that indicate compliance queries
    COMPLIANCE_KEYWORDS = {
        # Business structures
        'fund', 'spv', 'syndicate', 'investment', 'collective', 'scheme', 
        'vehicle', 'entity', 'company', 'partnership', 'trust',
        
        # Regulatory concepts
        'license', 'permit', 'registration', 'authorization', 'approval',
        'compliance', 'regulation', 'rule', 'law', 'requirement',
        'aml', 'kyc', 'cdd', 'sanctions', 'reporting',
        
        # Business activities
        'marketing', 'solicitation', 'offering', 'management',
        'custody', 'administration', 'advice', 'advisory',
        
        # Jurisdictions
        'adgm', 'uae', 'dubai', 'abu dhabi', 'difc',
        
        # Document types
        'rulebook', 'guidance', 'circular', 'policy',
        
        # Financial services
        'financial services', 'asset management', 'wealth management',
        'brokerage', 'dealing', 'arranging', 'advising'
    }
    
    COMPLEXITY_INDICATORS = {
        'simple': ['what is', 'define', 'definition of', 'meaning of'],
        'moderate': ['requirements for', 'how to', 'process for', 'when do i need'],
        'complex': ['structure that', 'avoid being classified', 'multiple', 'both', 'either']
    }
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("compliance_classification")
        self.client = client
    
    async def process(self, state: ProcessingState, query: str, **kwargs) -> Dict[str, Any]:
        """Classify the query as compliance-related and assess complexity."""
        
        # Initial keyword-based screening
        keyword_confidence = self._keyword_analysis(query)
        
        # LLM-based classification for nuanced understanding
        llm_classification = await self._llm_classification(query)
        
        # Combine results
        final_classification = self._combine_classifications(
            keyword_confidence, llm_classification, query
        )
        
        return {
            "classification": final_classification,
            "raw_query": query,
            "keyword_confidence": keyword_confidence,
            "llm_confidence": llm_classification.confidence_score
        }
    
    def _keyword_analysis(self, query: str) -> float:
        """Analyze query using compliance keywords."""
        query_lower = query.lower()
        
        # Count matches with different weights
        direct_matches = sum(1 for keyword in self.COMPLIANCE_KEYWORDS 
                           if keyword in query_lower)
        
        # Weight by keyword importance
        important_keywords = ['fund', 'license', 'compliance', 'regulation', 'spv']
        important_matches = sum(2 for keyword in important_keywords 
                              if keyword in query_lower)
        
        total_score = direct_matches + important_matches
        max_possible = len(self.COMPLIANCE_KEYWORDS) + len(important_keywords)
        
        return min(total_score / max_possible * 2, 1.0)  # Scale to 0-1
    
    async def _llm_classification(self, query: str) -> QueryClassification:
        """Use LLM for nuanced classification."""
        
        system_prompt = """You are an expert compliance analyst specializing in ADGM financial regulations.
        
        Analyze the user query and determine:
        1. Is this a compliance/regulatory question? (true/false)
        2. What is the complexity level? (simple/moderate/complex)
        3. What regulatory domains might be involved?
        4. Provide your reasoning
        
        Classification guidelines:
        - Compliance queries involve financial services, business structures, regulatory requirements, or legal obligations
        - Simple: Basic definitions or single concept questions
        - Moderate: Process questions, requirements for specific activities
        - Complex: Multi-faceted scenarios, comparative analysis, structure optimization
        
        Respond in JSON format only."""
        
        user_prompt = f"""Query: "{query}"
        
        Provide classification as JSON:
        {{
            "is_compliance": boolean,
            "complexity_level": "simple|moderate|complex",
            "confidence_score": float (0.0-1.0),
            "reasoning": "explanation of your classification",
            "detected_domains": ["domain1", "domain2", ...]
        }}"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return QueryClassification(
                is_compliance=result["is_compliance"],
                complexity_level=QueryComplexity(result["complexity_level"]),
                confidence_score=result["confidence_score"],
                reasoning=result["reasoning"],
                detected_domains=result.get("detected_domains", [])
            )
            
        except Exception as e:
            # Fallback to keyword-based classification
            keyword_conf = self._keyword_analysis(query)
            return QueryClassification(
                is_compliance=keyword_conf > 0.3,
                complexity_level=self._assess_complexity(query),
                confidence_score=keyword_conf,
                reasoning=f"Fallback classification due to error: {str(e)}",
                detected_domains=[]
            )
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on linguistic patterns."""
        query_lower = query.lower()
        
        # Check for complexity indicators
        for complexity, indicators in self.COMPLEXITY_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                return QueryComplexity(complexity)
        
        # Default based on query length and structure
        if len(query.split()) > 20 or '?' in query.count('?') > 1:
            return QueryComplexity.COMPLEX
        elif len(query.split()) > 10:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _combine_classifications(self, keyword_conf: float, 
                               llm_classification: QueryClassification, 
                               query: str) -> QueryClassification:
        """Combine keyword and LLM classifications."""
        
        # Weight LLM more heavily but use keyword as validation
        combined_confidence = (llm_classification.confidence_score * 0.7 + 
                             keyword_conf * 0.3)
        
        # If there's significant disagreement, be more conservative
        if abs(keyword_conf - llm_classification.confidence_score) > 0.4:
            combined_confidence *= 0.8
        
        # Final decision: if either method is highly confident it's compliance, classify as such
        is_compliance = (llm_classification.is_compliance and llm_classification.confidence_score > 0.6) or keyword_conf > 0.5
        
        return QueryClassification(
            is_compliance=is_compliance,
            complexity_level=llm_classification.complexity_level,
            confidence_score=combined_confidence,
            reasoning=f"Combined analysis: Keyword confidence: {keyword_conf:.2f}, LLM: {llm_classification.reasoning}",
            detected_domains=llm_classification.detected_domains
        )
    
    def get_next_node(self, state: ProcessingState) -> Optional[str]:
        """Determine next node based on classification."""
        classification = state.intermediate_results["compliance_classification"]["classification"]
        
        if classification.is_compliance:
            return "regulatory_context"
        else:
            return "direct_response"
