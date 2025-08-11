"""
Query Decomposition Node - Breaks complex queries into atomic sub-questions.
"""
import re
from typing import Any, Dict, List
from openai import AsyncOpenAI
from .base_node import BaseNode
from ..models.query_models import ProcessingState, QueryDecomposition, SubQuestion


class QueryDecompositionNode(BaseNode):
    """Decomposes complex compliance queries into manageable sub-questions."""
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("query_decomposition")
        self.client = client
    
    async def process(self, state: ProcessingState, **kwargs) -> Dict[str, Any]:
        """Decompose the query into atomic sub-questions."""
        
        classification = state.intermediate_results["compliance_classification"]["classification"]
        context = state.intermediate_results["regulatory_context"]["context"]
        query = state.intermediate_results["compliance_classification"]["raw_query"]
        
        # For simple queries, minimal decomposition needed
        if classification.complexity_level.value == "simple":
            decomposition = await self._simple_decomposition(query, context)
        else:
            decomposition = await self._complex_decomposition(query, context, classification)
        
        # Validate and enrich the decomposition
        enriched_decomposition = self._enrich_decomposition(decomposition, context)
        
        return {
            "decomposition": enriched_decomposition,
            "complexity_assessment": {
                "original_complexity": classification.complexity_level.value,
                "sub_question_count": len(enriched_decomposition.sub_questions),
                "estimated_processing_time": self._estimate_processing_time(enriched_decomposition)
            }
        }
    
    async def _simple_decomposition(self, query: str, context) -> QueryDecomposition:
        """Handle simple queries with minimal decomposition."""
        
        # Extract the main entity/concept being asked about
        entities = self._extract_entities(query)
        
        # Create a single definition question
        main_question = SubQuestion(
            question=f"What is the regulatory definition of {entities[0] if entities else 'the concept in question'}?",
            question_type="definition",
            priority=1,
            entities_involved=entities,
            regulatory_domains=context.regulatory_domains
        )
        
        return QueryDecomposition(
            sub_questions=[main_question],
            entity_definitions_needed=entities,
            regulatory_requirements_to_check=[],
            analysis_required=["definition_lookup"]
        )
    
    async def _complex_decomposition(self, query: str, context, classification) -> QueryDecomposition:
        """Handle complex queries with comprehensive decomposition."""
        
        system_prompt = f"""You are an expert compliance analyst. Decompose this complex query into specific, atomic sub-questions that need to be answered to provide comprehensive compliance advice.

        Context:
        - Jurisdiction: {context.jurisdiction}
        - Regulatory domains: {', '.join(context.regulatory_domains)}
        - Query complexity: {classification.complexity_level.value}

        For each sub-question, determine:
        1. The specific question to be answered
        2. Question type: definition, requirement, comparison, analysis
        3. Priority (1=highest, 10=lowest) 
        4. Entities involved
        5. Regulatory domains relevant

        Focus on:
        - What definitions need to be clarified?
        - What regulatory requirements apply?
        - What comparisons or analyses are needed?
        - What are the compliance implications?

        Provide response as JSON with this structure:
        {{
            "sub_questions": [
                {{
                    "question": "specific question text",
                    "question_type": "definition|requirement|comparison|analysis", 
                    "priority": integer 1-10,
                    "entities_involved": ["entity1", "entity2"],
                    "regulatory_domains": ["domain1", "domain2"]
                }}
            ],
            "entity_definitions_needed": ["entity1", "entity2"],
            "regulatory_requirements_to_check": ["requirement1", "requirement2"],
            "analysis_required": ["type1", "type2"]
        }}"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query to decompose: '{query}'"}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Convert to Pydantic models
            sub_questions = [
                SubQuestion(
                    question=sq["question"],
                    question_type=sq["question_type"],
                    priority=sq["priority"],
                    entities_involved=sq.get("entities_involved", []),
                    regulatory_domains=sq.get("regulatory_domains", context.regulatory_domains)
                )
                for sq in result["sub_questions"]
            ]
            
            return QueryDecomposition(
                sub_questions=sub_questions,
                entity_definitions_needed=result.get("entity_definitions_needed", []),
                regulatory_requirements_to_check=result.get("regulatory_requirements_to_check", []),
                analysis_required=result.get("analysis_required", [])
            )
            
        except Exception as e:
            # Fallback to rule-based decomposition
            return self._fallback_decomposition(query, context)
    
    def _fallback_decomposition(self, query: str, context) -> QueryDecomposition:
        """Fallback decomposition using rule-based approach."""
        
        entities = self._extract_entities(query)
        sub_questions = []
        
        # Always start with definitions
        for i, entity in enumerate(entities):
            sub_questions.append(
                SubQuestion(
                    question=f"What is the regulatory definition of {entity}?",
                    question_type="definition",
                    priority=i + 1,
                    entities_involved=[entity],
                    regulatory_domains=context.regulatory_domains
                )
            )
        
        # Look for requirement questions
        if any(word in query.lower() for word in ['require', 'need', 'must', 'should']):
            sub_questions.append(
                SubQuestion(
                    question="What are the regulatory requirements for this scenario?",
                    question_type="requirement", 
                    priority=len(entities) + 1,
                    entities_involved=entities,
                    regulatory_domains=context.regulatory_domains
                )
            )
        
        # Look for comparison questions
        if any(word in query.lower() for word in ['vs', 'versus', 'compare', 'different', 'avoid']):
            sub_questions.append(
                SubQuestion(
                    question="How do these concepts compare from a regulatory perspective?",
                    question_type="comparison",
                    priority=len(entities) + 2, 
                    entities_involved=entities,
                    regulatory_domains=context.regulatory_domains
                )
            )
        
        return QueryDecomposition(
            sub_questions=sub_questions,
            entity_definitions_needed=entities,
            regulatory_requirements_to_check=context.regulatory_domains,
            analysis_required=["definition_synthesis", "requirement_analysis"]
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract business/legal entities from the query."""
        
        # Common financial/legal entities
        entity_patterns = [
            r'\b(spv|special purpose vehicle)\b',
            r'\b(fund|investment fund|collective investment)\b',
            r'\b(syndicate|investment syndicate)\b',
            r'\b(company|corporation|llc)\b',
            r'\b(partnership|limited partnership)\b',
            r'\b(trust|unit trust)\b',
            r'\b(scheme|investment scheme)\b',
            r'\b(vehicle|investment vehicle)\b',
            r'\b(entity|legal entity)\b'
        ]
        
        entities = []
        query_lower = query.lower()
        
        for pattern in entity_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity = match.group(0).strip()
                if entity not in entities:
                    entities.append(entity)
        
        return entities
    
    def _enrich_decomposition(self, decomposition: QueryDecomposition, context) -> QueryDecomposition:
        """Enrich decomposition with dependency analysis and validation."""
        
        # Sort questions by priority
        decomposition.sub_questions.sort(key=lambda x: x.priority)
        
        # Add dependencies (definitions before requirements/comparisons)
        dependencies = {}
        definition_questions = [q for q in decomposition.sub_questions if q.question_type == "definition"]
        other_questions = [q for q in decomposition.sub_questions if q.question_type != "definition"]
        
        for other_q in other_questions:
            # Other questions depend on relevant definition questions
            deps = []
            for def_q in definition_questions:
                if any(entity in other_q.entities_involved for entity in def_q.entities_involved):
                    deps.append(def_q.question)
            if deps:
                dependencies[other_q.question] = deps
        
        decomposition.dependencies = dependencies
        
        return decomposition
    
    def _estimate_processing_time(self, decomposition: QueryDecomposition) -> str:
        """Estimate processing time based on complexity."""
        question_count = len(decomposition.sub_questions)
        
        if question_count <= 2:
            return "30-60 seconds"
        elif question_count <= 5:
            return "1-2 minutes"
        else:
            return "2-3 minutes"
