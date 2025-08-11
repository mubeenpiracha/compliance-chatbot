"""
Knowledge Gap Identification Node - Determines what needs retrieval vs clarification.
"""
from typing import Any, Dict, List
from openai import AsyncOpenAI
from .base_node import BaseNode
from ..models.query_models import ProcessingState, KnowledgeGap, ClarificationQuestion


class KnowledgeGapIdentificationNode(BaseNode):
    """Identifies what information can be retrieved vs what needs customer clarification."""
    
    # Terms that commonly need customer clarification
    AMBIGUOUS_TERMS = {
        'syndicate': "A business arrangement that could have various structures",
        'simple structure': "Could refer to different organizational forms",
        'key investment decisions': "The specific decisions that define fund-like activity",
        'minimal involvement': "The degree of management activity",
        'passive investment': "The level of active management involved"
    }
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("knowledge_gap_identification") 
        self.client = client
    
    async def process(self, state: ProcessingState, **kwargs) -> Dict[str, Any]:
        """Identify knowledge gaps and clarification needs."""
        
        decomposition = state.intermediate_results["query_decomposition"]["decomposition"]
        context = state.intermediate_results["regulatory_context"]["context"]
        original_query = state.intermediate_results["compliance_classification"]["raw_query"]
        
        # Analyze each sub-question
        retrieval_queries = []
        clarification_questions = []
        coverage_assessment = {}
        
        for sub_question in decomposition.sub_questions:
            analysis = await self._analyze_sub_question(
                sub_question, context, original_query
            )
            
            if analysis["can_retrieve"]:
                retrieval_queries.extend(analysis["retrieval_queries"])
                coverage_assessment[sub_question.question] = True
            else:
                clarification_questions.extend(analysis["clarification_questions"])
                coverage_assessment[sub_question.question] = False
                
        # Generate additional clarifications for ambiguous terms
        additional_clarifications = self._identify_ambiguous_terms(original_query)
        clarification_questions.extend(additional_clarifications)
        
        knowledge_gap = KnowledgeGap(
            retrieval_queries=retrieval_queries,
            clarification_questions=clarification_questions,
            question_dependencies=decomposition.dependencies,
            coverage_assessment=coverage_assessment
        )
        
        return {
            "knowledge_gap": knowledge_gap,
            "analysis_summary": {
                "retrievable_questions": sum(1 for v in coverage_assessment.values() if v),
                "clarification_needed": sum(1 for v in coverage_assessment.values() if not v),
                "total_retrieval_queries": len(retrieval_queries),
                "total_clarifications": len(clarification_questions)
            }
        }
    
    async def _analyze_sub_question(self, sub_question, context, original_query) -> Dict[str, Any]:
        """Analyze whether a sub-question can be answered through retrieval or needs clarification."""
        
        system_prompt = f"""You are analyzing whether a compliance sub-question can be answered through document retrieval or needs customer clarification.
        
        Regulatory Context:
        - Jurisdiction: {context.jurisdiction}
        - Domains: {', '.join(context.regulatory_domains)}
        
        Guidelines:
        - Regulatory definitions and requirements can usually be retrieved from documents
        - Customer-specific business models, structures, and interpretations need clarification
        - Comparative analysis may need both retrieval and clarification
        - Ambiguous terms in customer context need clarification
        
        Respond in JSON format."""
        
        user_prompt = f"""Original query: "{original_query}"
        
        Sub-question to analyze: "{sub_question.question}"
        Question type: {sub_question.question_type}
        Entities involved: {', '.join(sub_question.entities_involved)}
        
        Analyze and respond with JSON:
        {{
            "can_retrieve": boolean,
            "confidence": float (0.0-1.0),
            "reasoning": "explanation",
            "retrieval_queries": ["query1", "query2"] (if can_retrieve=true),
            "clarification_questions": [
                {{
                    "question": "clarification question",
                    "context": "why this clarification is needed",
                    "question_type": "definition|scenario|preference|constraint"
                }}
            ] (if can_retrieve=false)
        }}"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Convert clarification questions to proper model
            clarifications = []
            for cq in result.get("clarification_questions", []):
                clarifications.append(
                    ClarificationQuestion(
                        question=cq["question"],
                        context=cq["context"],
                        question_type=cq["question_type"],
                        is_required=True
                    )
                )
            
            return {
                "can_retrieve": result["can_retrieve"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "retrieval_queries": result.get("retrieval_queries", []),
                "clarification_questions": clarifications
            }
            
        except Exception as e:
            # Fallback analysis
            return self._fallback_analysis(sub_question, original_query)
    
    def _fallback_analysis(self, sub_question, original_query) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        
        # Simple heuristics
        question_lower = sub_question.question.lower()
        original_lower = original_query.lower()
        
        # If asking for regulatory definitions, can usually retrieve
        if sub_question.question_type == "definition" and any(
            term in question_lower for term in ["regulatory definition", "legal definition", "defined as"]
        ):
            return {
                "can_retrieve": True,
                "confidence": 0.8,
                "reasoning": "Regulatory definitions can typically be found in documents",
                "retrieval_queries": [sub_question.question],
                "clarification_questions": []
            }
        
        # If question contains customer-specific terms, needs clarification
        if any(term in original_lower for term in self.AMBIGUOUS_TERMS.keys()):
            clarification = ClarificationQuestion(
                question=f"Can you clarify what you mean by the term used in your question?",
                context="Customer-specific terms need clarification for accurate analysis",
                question_type="definition",
                is_required=True
            )
            return {
                "can_retrieve": False,
                "confidence": 0.7,
                "reasoning": "Contains customer-specific terms needing clarification",
                "retrieval_queries": [],
                "clarification_questions": [clarification]
            }
        
        # Default to retrieval for regulatory questions
        return {
            "can_retrieve": True,
            "confidence": 0.6,
            "reasoning": "Default assumption for regulatory questions",
            "retrieval_queries": [sub_question.question],
            "clarification_questions": []
        }
    
    def _identify_ambiguous_terms(self, query: str) -> List[ClarificationQuestion]:
        """Identify ambiguous terms that need clarification."""
        
        clarifications = []
        query_lower = query.lower()
        
        for term, description in self.AMBIGUOUS_TERMS.items():
            if term in query_lower:
                # Create specific clarification based on the term
                clarification = self._create_term_clarification(term, description, query)
                if clarification:
                    clarifications.append(clarification)
        
        return clarifications
    
    def _create_term_clarification(self, term: str, description: str, query: str) -> ClarificationQuestion:
        """Create specific clarification questions for ambiguous terms."""
        
        clarification_templates = {
            'syndicate': ClarificationQuestion(
                question="When you mention 'syndicate', can you describe the specific structure you have in mind? For example: How many parties are involved? What are their roles? How are investment decisions made?",
                context=f"The term 'syndicate' can refer to various business arrangements. Understanding your specific structure is crucial for regulatory analysis.",
                question_type="scenario",
                suggested_answers=[
                    "A group of 2-3 investors pooling resources with shared decision-making",
                    "A lead investor with several passive co-investors", 
                    "A formal partnership structure with defined roles",
                    "Other (please describe)"
                ]
            ),
            
            'key investment decisions': ClarificationQuestion(
                question="What do you consider to be the 'key investment decisions' in your structure? Please provide specific examples.",
                context="Understanding what constitutes key investment decisions helps determine if the structure resembles a collective investment scheme.",
                question_type="definition",
                suggested_answers=[
                    "Selecting which companies/assets to invest in",
                    "Determining investment amounts and timing",
                    "Exit strategy decisions",
                    "All major investment-related decisions"
                ]
            ),
            
            'simple structure': ClarificationQuestion(
                question="Can you describe what you mean by 'simple structure'? What specific legal form are you considering?",
                context="Different legal structures have different regulatory implications.",
                question_type="scenario",
                suggested_answers=[
                    "Limited liability company (LLC)",
                    "Partnership arrangement",
                    "Contractual agreement between parties",
                    "Other legal structure"
                ]
            )
        }
        
        return clarification_templates.get(term)
