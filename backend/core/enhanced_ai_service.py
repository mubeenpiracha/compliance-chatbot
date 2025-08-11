"""
Enhanced AI service with a flexible, agent-based reasoning architecture.
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from pydantic import ValidationError

from .real_vector_service import RealVectorService
from .models.agent_models import QueryAnalysis, SearchPlan, ClarificationRequest
from .retrieval.vector_search import VectorSearchEngine
from .retrieval.keyword_search import KeywordSearchEngine
from .retrieval.result_fusion import ResultFusion
from .models.retrieval_models import RetrievalQuery, RetrievedDocument, DocumentType
from .models.agent_models import SearchQuery

logger = logging.getLogger(__name__)


class EnhancedAIService:
    """
    Enhanced AI service implementing a flexible, agent-based reasoning loop.
    """
    
    def __init__(self, openai_api_key: str, vector_service: RealVectorService, document_corpus: List[Dict]):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.vector_service = vector_service
        self.document_corpus = document_corpus
        
        # Initialize retrieval components
        self.vector_search = VectorSearchEngine(self.client, self.vector_service)
        self.keyword_search = KeywordSearchEngine(self.document_corpus)
        self.result_fusion = ResultFusion()
        
    async def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process user query using a flexible reasoning loop.
        
        Args:
            query: User's question or request
            conversation_history: Previous conversation context
            
        Returns:
            Dict containing the response, sources, reasoning steps, etc.
        """
        start_time = time.time()
        
        try:
            # 1. Holistic Query Analysis
            logger.info("Performing holistic query analysis...")
            analysis = await self._analyze_query(query, conversation_history)
            
            # 2. Decide and Act
            if isinstance(analysis.decision, ClarificationRequest):
                logger.info("Clarification needed. Pausing process.")
                return self._format_clarification_response(analysis, time.time() - start_time)
            
            if isinstance(analysis.decision, SearchPlan):
                logger.info("Executing search plan...")
                search_plan = analysis.decision
                
                # Execute searches in parallel
                search_tasks = []
                for search_query in search_plan.search_queries:
                    if search_query.search_type == "vector":
                        task = self.vector_search.search(self._to_retrieval_query(search_query))
                    else: # keyword
                        task = self.keyword_search.search(self._to_retrieval_query(search_query))
                    search_tasks.append(task)
                
                # Gather results
                all_search_results = await asyncio.gather(*search_tasks)
                
                # 3. Reflect and Synthesize
                logger.info("Reflecting on search results...")
                retrieved_docs = [doc for result_list in all_search_results for doc in result_list]

                if not retrieved_docs:
                    logger.warning("No documents found from initial search. Attempting to generate a response without documents.")
                    final_response = await self._generate_no_documents_response(query, analysis.reasoning)
                else:
                    logger.info(f"Found {len(retrieved_docs)} documents. Synthesizing final response.")
                    final_response = await self._generate_document_based_response(query, analysis.reasoning, retrieved_docs)

                final_response["processing_time"] = time.time() - start_time
                return final_response

        except Exception as e:
            logger.error(f"Enhanced AI service error: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try rephrasing your question.",
                "sources": [],
                "reasoning": [f"Error: {str(e)}"],
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def _analyze_query(self, query: str, history: Optional[List[Dict]]) -> QueryAnalysis:
        """
        Performs a single, holistic analysis of the user's query to decide the next step.
        """
        system_prompt = """You are an expert compliance analyst. Your task is to analyze a user's query and decide the best course of action.

You have two choices:
1.  **Create a Search Plan**: If the query is clear and has enough detail to be answered by searching regulatory documents. The plan should include multiple, diverse search queries (vector and keyword) to ensure comprehensive results.
2.  **Request Clarification**: If the query is ambiguous, vague, or lacks critical context that only the user can provide.

**Analysis Steps:**
1.  **Intent**: What is the user's core question?
2.  **Context**: Identify jurisdiction (assume ADGM if not specified) and key regulatory concepts.
3.  **Completeness**: Is the query self-contained? Are there ambiguous terms (e.g., "syndicate," "simple structure")?
4.  **Decision**: Based on the above, decide whether to create a `SearchPlan` or a `ClarificationRequest`.

**Respond in JSON format only, using one of the two schemas provided.**

**Schema for SearchPlan:**
```json
{
  "decision": {
    "search_plan": {
      "search_queries": [
        {
          "query_text": "detailed semantic query for vector search",
          "search_type": "vector",
          "purpose": "reason for this query"
        },
        {
          "query_text": "precise keyword query",
          "search_type": "keyword",
          "purpose": "reason for this query"
        }
      ]
    }
  },
  "reasoning": "Explanation of why you chose to create a search plan."
}
```

**Schema for ClarificationRequest:**
```json
{
  "decision": {
    "clarification_request": {
      "clarification_questions": [
        "Question 1 to ask the user.",
        "Question 2 to ask the user."
      ]
    }
  },
  "reasoning": "Explanation of why clarification is necessary."
}
```"""
        
        user_prompt = f"User Query: \"{query}\""
        if history:
            user_prompt += f"\n\nConversation History:\n{str(history)}"

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Pydantic will automatically validate and parse into the correct Union type
            if "search_plan" in result["decision"]:
                plan = SearchPlan(**result["decision"]["search_plan"])
                return QueryAnalysis(decision=plan, reasoning=result["reasoning"])
            elif "clarification_request" in result["decision"]:
                request = ClarificationRequest(**result["decision"]["clarification_request"])
                return QueryAnalysis(decision=request, reasoning=result["reasoning"])
            else:
                raise ValueError("Invalid decision structure in LLM response.")

        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse analysis from LLM: {e}")
            # Fallback: create a simple search plan
            fallback_plan = SearchPlan(search_queries=[
                SearchQuery(query_text=query, search_type="vector", purpose="Fallback vector search"),
                SearchQuery(query_text=query, search_type="keyword", purpose="Fallback keyword search")
            ])
            return QueryAnalysis(decision=fallback_plan, reasoning="Fell back to a default search plan due to a parsing error.")

    def _format_clarification_response(self, analysis: QueryAnalysis, processing_time: float) -> Dict[str, Any]:
        """Formats the response when the system needs to ask for clarification."""
        clarification_request = analysis.decision
        return {
            "response": "To provide you with the most accurate guidance, I need a bit more information. Could you please clarify the following points?",
            "requires_clarification": True,
            "clarification_questions": clarification_request.clarification_questions,
            "sources": [],
            "reasoning": [analysis.reasoning],
            "confidence": 0.0,
            "processing_time": processing_time,
        }

    async def _generate_no_documents_response(self, query: str, reasoning: str) -> Dict[str, Any]:
        """Generate a response when no relevant documents are found."""
        return {
            "response": "I was unable to find specific regulatory documents that directly address your query. This could be because the topic is highly specialized or requires interpretation beyond the scope of the available documents. I recommend consulting directly with a qualified compliance advisor.",
            "sources": [],
            "reasoning": [reasoning, "No relevant documents were found after executing the search plan."],
            "confidence": 0.2,
            "requires_clarification": False,
        }

    async def _generate_document_based_response(self, query: str, reasoning: str, documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Generate the final comprehensive response based on retrieved documents."""
        
        document_context = "\n\n".join([
            f"**Source: {doc.source.title} - Section: {doc.source.section}**\n{doc.content}"
            for doc in documents[:10] # Use top 10 documents for context
        ])

        system_prompt = f"""You are an expert ADGM compliance advisor. Your task is to synthesize the provided regulatory documents to answer the user's query.

**Your Reasoning So Far:**
{reasoning}

**Instructions:**
1.  Provide a direct and comprehensive answer to the user's query.
2.  Base your answer *only* on the information contained in the provided regulatory documents.
3.  Cite the specific source document for each piece of information you provide (e.g., "According to the COBS Rulebook...").
4.  If the documents do not fully answer the question, state that clearly. Do not invent information.
5.  Structure your response for clarity with headings and bullet points.
"""
        
        user_prompt = f"""User Query: "{query}"

**Retrieved Regulatory Documents:**
{document_context}

Please provide your comprehensive compliance analysis based *only* on these documents."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            final_answer = response.choices[0].message.content
            confidence = self._calculate_confidence(documents)

            return {
                "response": final_answer,
                "sources": self._format_sources(documents),
                "reasoning": [reasoning, f"Synthesized response from {len(documents)} retrieved document(s)."],
                "confidence": confidence,
                "requires_clarification": False,
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return {
                "response": f"I found relevant regulatory information but encountered an error generating the final response. Error: {str(e)}",
                "sources": self._format_sources(documents),
                "reasoning": [reasoning, f"Error during final synthesis: {str(e)}"],
                "confidence": 0.4,
                "requires_clarification": False,
            }

    def _to_retrieval_query(self, search_query: "SearchQuery") -> RetrievalQuery:
        """Converts a SearchQuery from the plan to a RetrievalQuery for the engines."""
        return RetrievalQuery(
            query_text=search_query.query_text,
            query_type=search_query.search_type,
            max_results=10,
            min_relevance_score=0.3 if search_query.search_type == "vector" else 3.0
        )

    def _format_sources(self, documents: List[RetrievedDocument]) -> List[Dict[str, Any]]:
        """Format document sources for the final response."""
        sources = []
        for doc in documents:
            sources.append({
                "title": doc.source.title,
                "document_type": doc.source.document_type.value,
                "section": doc.source.section,
                "relevance_score": doc.relevance_score,
                "jurisdiction": doc.source.jurisdiction,
            })
        # Deduplicate and return top 10
        unique_sources = {s['title']: s for s in sources}.values()
        return list(unique_sources)[:10]

    def _calculate_confidence(self, documents: List[RetrievedDocument]) -> float:
        """Calculate confidence score based on the quality of retrieved documents."""
        if not documents:
            return 0.0
        
        # Simple confidence: average relevance score, capped at 0.95
        avg_score = sum(doc.relevance_score for doc in documents) / len(documents)
        
        # Normalize for different search types
        # Assuming vector scores are 0-1 and keyword scores are higher
        if any(doc.retrieval_method == 'keyword' for doc in documents):
             # Heuristic normalization for keyword scores
            avg_score = min(avg_score / 10.0, 1.0)

        return min(avg_score * 0.9, 0.95)


# Global service instance
_service_instance = None


def get_ai_response(user_message: str, history: List[Dict] = None, jurisdiction: str = None) -> Dict[str, Any]:
    """
    Legacy interface for backwards compatibility with existing API.
    Now uses the new agent-based reasoning service.
    """
    import asyncio
    
    global _service_instance
    
    if _service_instance is None:
        from .real_vector_service import RealVectorService
        from .models.retrieval_models import RetrievalQuery, RetrievedDocument, DocumentType
        from .document_loader import load_document_corpus_from_content_store
        from .config import OPENAI_API_KEY
        
        # Initialize the real services
        vector_service = RealVectorService()
        document_corpus = load_document_corpus_from_content_store()
        
        _service_instance = EnhancedAIService(
            openai_api_key=OPENAI_API_KEY,
            vector_service=vector_service,
            document_corpus=document_corpus
        )
    
    # Run the async process_query method
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_service_instance.process_query(user_message, history))
        
        # Convert to legacy format if needed, though the new format is largely compatible
        return {
            'answer': result['response'],
            'sources': result.get('sources', []),
            'confidence': result.get('confidence', 0.0),
            'reasoning': result.get('reasoning', []),
            'requires_clarification': result.get('requires_clarification', False),
            'clarification_questions': result.get('clarification_questions', [])
        }
        
    finally:
        loop.close()
