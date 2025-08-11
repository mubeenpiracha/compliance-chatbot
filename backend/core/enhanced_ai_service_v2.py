"""
Enhanced AI service with chain-of-thought reasoning and hybrid retrieval.
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from .vector_service import VectorService
from .models.query_models import ProcessingState
from .nodes.classification import ComplianceClassificationNode
from .nodes.context_identification import RegulatoryContextNode
from .nodes.query_decomposition import QueryDecompositionNode
from .nodes.knowledge_gap_analysis import KnowledgeGapIdentificationNode
from .nodes.hybrid_retrieval import HybridRetrievalNode

logger = logging.getLogger(__name__)


class EnhancedAIServiceV2:
    """
    Enhanced AI service implementing chain-of-thought reasoning with hybrid retrieval.
    """
    
    def __init__(self, openai_api_key: str, vector_service: VectorService, document_corpus: List[Dict]):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.vector_service = vector_service
        self.document_corpus = document_corpus
        
        # Initialize nodes
        self.classification_node = ComplianceClassificationNode(self.client)
        self.context_node = RegulatoryContextNode(self.client)
        self.decomposition_node = QueryDecompositionNode(self.client)
        self.knowledge_gap_node = KnowledgeGapIdentificationNode(self.client)
        self.retrieval_node = HybridRetrievalNode(self.client, vector_service, document_corpus)
        
        # Processing graph
        self.node_graph = {
            "classification": self.classification_node,
            "regulatory_context": self.context_node,
            "query_decomposition": self.decomposition_node,
            "knowledge_gap_identification": self.knowledge_gap_node,
            "hybrid_retrieval": self.retrieval_node
        }
        
    async def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process user query with enhanced chain-of-thought reasoning.
        
        Args:
            query: User's question or request
            conversation_history: Previous conversation context
            
        Returns:
            Dict containing response, sources, reasoning steps, etc.
        """
        start_time = time.time()
        
        try:
            # Initialize processing state
            state = ProcessingState(
                current_node="classification",
                completed_nodes=[],
                pending_clarifications=[],
                collected_clarifications=[],
                intermediate_results={},
                requires_user_input=False
            )
            
            # Execute processing pipeline
            result = await self._execute_processing_pipeline(query, state)
            
            processing_time = time.time() - start_time
            
            # Check if clarifications are needed
            if state.requires_user_input:
                return self._format_clarification_response(state, processing_time)
            
            # Generate final response
            final_response = await self._generate_final_response(state, query)
            final_response["processing_time"] = processing_time
            
            return final_response
            
        except Exception as e:
            logger.error(f"Enhanced AI service error: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try rephrasing your question.",
                "sources": [],
                "reasoning": [],
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _execute_processing_pipeline(self, query: str, state: ProcessingState) -> ProcessingState:
        """Execute the main processing pipeline."""
        
        # Step 1: Classification
        logger.info("Starting compliance classification...")
        state = await self.classification_node.execute(state, query=query)
        
        classification = state.intermediate_results["compliance_classification"]["classification"]
        
        # If not a compliance query, skip to direct response
        if not classification.is_compliance:
            logger.info("Query classified as non-compliance, generating direct response")
            return state
        
        # Step 2: Regulatory Context Identification
        logger.info("Identifying regulatory context...")
        state.current_node = "regulatory_context"
        state = await self.context_node.execute(state)
        
        # Step 3: Query Decomposition
        logger.info("Decomposing query into sub-questions...")
        state.current_node = "query_decomposition"
        state = await self.decomposition_node.execute(state)
        
        # Step 4: Knowledge Gap Identification
        logger.info("Identifying knowledge gaps and clarification needs...")
        state.current_node = "knowledge_gap_identification"
        state = await self.knowledge_gap_node.execute(state)
        
        knowledge_gap = state.intermediate_results["knowledge_gap_identification"]["knowledge_gap"]
        
        # Check if clarifications are needed
        if knowledge_gap.clarification_questions:
            state.requires_user_input = True
            state.pending_clarifications = knowledge_gap.clarification_questions
            logger.info(f"Clarifications needed: {len(knowledge_gap.clarification_questions)} questions")
            return state
        
        # Step 5: Hybrid Retrieval
        logger.info("Executing hybrid retrieval...")
        state.current_node = "hybrid_retrieval"
        state = await self.retrieval_node.execute(state)
        
        logger.info("Processing pipeline completed successfully")
        return state
    
    def _format_clarification_response(self, state: ProcessingState, processing_time: float) -> Dict[str, Any]:
        """Format response when clarifications are needed."""
        
        classification = state.intermediate_results["compliance_classification"]["classification"]
        
        # Generate contextual message
        clarification_intro = self._generate_clarification_intro(classification, state.pending_clarifications)
        
        return {
            "response": clarification_intro,
            "requires_clarification": True,
            "clarification_questions": [
                {
                    "question": cq.question,
                    "context": cq.context,
                    "type": cq.question_type,
                    "suggested_answers": cq.suggested_answers,
                    "required": cq.is_required
                }
                for cq in state.pending_clarifications
            ],
            "sources": [],
            "reasoning": self._extract_reasoning_steps(state),
            "confidence": 0.0,  # No final confidence until clarifications provided
            "processing_time": processing_time,
            "state_id": id(state)  # For continuing the conversation
        }
    
    def _generate_clarification_intro(self, classification, clarification_questions) -> str:
        """Generate contextual introduction for clarification questions."""
        
        question_count = len(clarification_questions)
        
        intro = f"""I understand you're asking about {classification.detected_domains[0] if classification.detected_domains else 'compliance matters'}.
        
To provide you with accurate regulatory guidance, I need to clarify a few key points about your specific situation. """
        
        if question_count == 1:
            intro += "I have one important question:"
        else:
            intro += f"I have {question_count} important questions:"
        
        return intro
    
    async def _generate_final_response(self, state: ProcessingState, original_query: str) -> Dict[str, Any]:
        """Generate the final comprehensive response."""
        
        # For now, create a basic response - this will be enhanced with more analysis nodes
        classification = state.intermediate_results["compliance_classification"]["classification"]
        context = state.intermediate_results["regulatory_context"]["context"]
        decomposition = state.intermediate_results["query_decomposition"]["decomposition"]
        
        # Check if we have retrieval results
        if "hybrid_retrieval" in state.intermediate_results:
            retrieval_results = state.intermediate_results["hybrid_retrieval"]["consolidated_results"]
            documents = retrieval_results.get("documents", [])
        else:
            documents = []
        
        # Generate response based on available information
        if not documents:
            response = await self._generate_no_documents_response(classification, context, original_query)
            confidence = 0.3
        else:
            response = await self._generate_document_based_response(
                original_query, classification, context, decomposition, documents
            )
            confidence = min(retrieval_results.get("avg_confidence", 0.5), 0.9)
        
        return {
            "response": response,
            "sources": self._format_sources(documents),
            "reasoning": self._extract_reasoning_steps(state),
            "confidence": confidence,
            "requires_clarification": False,
            "sub_questions_analyzed": len(decomposition.sub_questions),
            "documents_found": len(documents)
        }
    
    async def _generate_no_documents_response(self, classification, context, query: str) -> str:
        """Generate response when no relevant documents are found."""
        
        response = f"""I understand you're asking about {classification.detected_domains[0] if classification.detected_domains else 'compliance matters'} in the {context.jurisdiction.upper()} jurisdiction.

While I wasn't able to find specific regulatory documents that directly address your query, I can provide some general guidance:

**Query Analysis:**
- Classification: {classification.complexity_level.value.title()} compliance question
- Regulatory domains: {', '.join(context.regulatory_domains)}
- Applicable frameworks: {', '.join(context.applicable_frameworks)}

**Recommendation:**
Given the specific nature of your query, I recommend:
1. Consulting directly with ADGM's Financial Services Regulatory Authority (FSRA)
2. Seeking advice from a qualified compliance consultant
3. Reviewing the specific regulatory modules that may apply to your situation

Would you like me to help you refine your question or provide more specific guidance on where to find the relevant regulatory information?"""
        
        return response
    
    async def _generate_document_based_response(self, query: str, classification, context, 
                                              decomposition, documents) -> str:
        """Generate comprehensive response based on retrieved documents."""
        
        # Prepare context for LLM
        document_context = "\n\n".join([
            f"**{doc.source.title}** ({doc.source.document_type.value})\n{doc.content}"
            for doc in documents[:5]  # Use top 5 documents
        ])
        
        system_prompt = f"""You are an expert ADGM compliance advisor. Using the provided regulatory documents, answer the user's question comprehensively.

Query Classification:
- Complexity: {classification.complexity_level.value}
- Domains: {', '.join(context.regulatory_domains)}
- Jurisdiction: {context.jurisdiction.upper()}

Sub-questions identified:
{chr(10).join([f"- {sq.question}" for sq in decomposition.sub_questions])}

Provide a structured response with:
1. Direct answer to the user's question
2. Relevant regulatory definitions
3. Specific requirements or compliance implications
4. Actionable recommendations
5. Areas where clarification may be needed

Always cite specific documents and sections. Be conservative in interpretation."""
        
        user_prompt = f"""User Query: "{query}"

Regulatory Documents:
{document_context}

Please provide a comprehensive compliance analysis."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"I found relevant regulatory information but encountered an error generating the response. Please try again or contact support. Error: {str(e)}"
    
    def _format_sources(self, documents) -> List[Dict[str, Any]]:
        """Format document sources for response."""
        
        sources = []
        for doc in documents:
            sources.append({
                "title": doc.source.title,
                "document_type": doc.source.document_type.value,
                "section": doc.source.section,
                "relevance_score": doc.relevance_score,
                "jurisdiction": doc.source.jurisdiction,
                "highlights": doc.match_highlights[:3]  # Top 3 highlights
            })
        
        return sources
    
    def _extract_reasoning_steps(self, state: ProcessingState) -> List[str]:
        """Extract reasoning steps from processing state."""
        
        reasoning = []
        
        if "compliance_classification" in state.intermediate_results:
            classification = state.intermediate_results["compliance_classification"]["classification"]
            reasoning.append(f"Classified as: {classification.complexity_level.value} compliance query (confidence: {classification.confidence_score:.2f})")
        
        if "regulatory_context" in state.intermediate_results:
            context = state.intermediate_results["regulatory_context"]["context"]
            reasoning.append(f"Identified jurisdiction: {context.jurisdiction.upper()}")
            reasoning.append(f"Regulatory domains: {', '.join(context.regulatory_domains)}")
        
        if "query_decomposition" in state.intermediate_results:
            decomposition = state.intermediate_results["query_decomposition"]["decomposition"]
            reasoning.append(f"Decomposed into {len(decomposition.sub_questions)} sub-questions")
        
        if "knowledge_gap_identification" in state.intermediate_results:
            analysis = state.intermediate_results["knowledge_gap_identification"]["analysis_summary"]
            reasoning.append(f"Knowledge gap analysis: {analysis['retrievable_questions']} retrievable, {analysis['clarification_needed']} need clarification")
        
        if "hybrid_retrieval" in state.intermediate_results:
            summary = state.intermediate_results["hybrid_retrieval"]["retrieval_summary"]
            reasoning.append(f"Retrieved {summary['total_documents_retrieved']} documents across {summary['queries_processed']} queries")
        
        return reasoning
    
    async def handle_clarification(self, original_query: str, clarifications: Dict[str, str], 
                                 state_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle user clarifications and continue processing.
        
        Args:
            original_query: The original user query
            clarifications: Dictionary of clarification responses
            state_id: ID of the processing state (for future use)
            
        Returns:
            Updated processing result
        """
        # For now, restart processing with clarifications incorporated
        # In the future, we could maintain state and continue from where we left off
        
        enhanced_query = self._incorporate_clarifications(original_query, clarifications)
        
        return await self.process_query(enhanced_query)
    
    def _incorporate_clarifications(self, original_query: str, clarifications: Dict[str, str]) -> str:
        """Incorporate user clarifications into the original query."""
        
        clarification_text = "\n\nClarifications:\n"
        for question, answer in clarifications.items():
            clarification_text += f"- {question}: {answer}\n"
        
        return original_query + clarification_text
