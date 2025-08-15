# backend/core/agent/nodes.py
import os
import json
import logging
from typing import List, Dict, Any, Union
from openai import AsyncOpenAI
from pydantic import ValidationError

from backend.core.agent.state import AgentState
from backend.core.models.agent_models import QueryAnalysis, SearchPlan, ClarificationRequest, SearchQuery
from backend.core.retrieval.vector_search import VectorSearchEngine
from backend.core.retrieval.keyword_search import KeywordSearchEngine
from backend.core.real_vector_service import RealVectorService
from backend.core.document_loader import load_document_corpus_from_content_store
from backend.core.models.retrieval_models import RetrievalQuery
from backend.core.config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use async client for all OpenAI operations
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def analyze_query(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's query to decide the next step.
    """
    logger.info("Node: analyze_query")
    user_query = state["user_query"]
    # Correctly access messages from the state
    history = state.get("messages", [])
    jurisdiction = state["jurisdiction"]

    # Convert history to the format OpenAI expects
    messages_history = []
    for msg in history:
        sender = msg.get("sender")
        text = msg.get("text")
        if sender and text:
            role = "user" if sender == "user" else "assistant"
            messages_history.append({"role": role, "content": text})

    system_prompt = f"""
You are an expert compliance analyst AI. Your task is to analyze a user's query about financial regulations in the specified jurisdiction: {jurisdiction}.
Based on the user's query and the conversation history, you must decide on one of two possible actions:

1.  **Search**: If the query is clear, specific, and actionable, create a `SearchPlan`. A search plan consists of one or more search queries. Each query must have a `description`.
2.  **Clarify**: If the query is ambiguous, vague, or lacks essential details, create a `ClarificationRequest`.

**Conversation History Analysis**:
- Review the provided conversation history.
- If the AI's last message was a clarification request, the user's current message is likely an answer.
- Use this new information to re-evaluate the original query.

**Instructions**:
- Your output **MUST** be a single JSON object that validates against the `QueryAnalysis` Pydantic model.
- Provide a concise `reasoning` for your decision.
- For a `SearchPlan`, ensure your decision has a `type` of `search_plan` and each query in the `queries` list is a dictionary with both a `query` and a `description` key.
- For a `ClarificationRequest`, ensure your decision has a `type` of `clarification_request` and you provide a `clarification_questions` list of strings.

**Pydantic Models for your reference**:
```python
from typing import List, Union, Literal
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str
    description: str

class SearchPlan(BaseModel):
    type: Literal["search_plan"] = "search_plan"
    queries: List[SearchQuery]

class ClarificationRequest(BaseModel):
    type: Literal["clarification_request"] = "clarification_request"
    clarification_questions: List[str]

class QueryAnalysis(BaseModel):
    reasoning: str
    decision: Union[SearchPlan, ClarificationRequest] = Field(discriminator="type")
```

**Output Format Examples**:

*If you decide to search*:
```json
{{
  "reasoning": "The user's query is specific and requires looking up definitions in the glossary.",
  "decision": {{
    "type": "search_plan",
    "queries": [
      {{
        "query": "Definition of 'Authorised Firm' in DIFC glossary",
        "description": "This query will find the precise definition of 'Authorised Firm' which is central to the user's question."
      }}
    ]
  }}
}}
```

*If you decide to clarify*:
```json
{{
  "reasoning": "The user's query is too broad. I need to know which specific regulations they are interested in.",
  "decision": {{
    "type": "clarification_request",
    "clarification_questions": [
      "Which specific regulation are you asking about?",
      "Can you provide more context on what you are trying to achieve?"
    ]
  }}
}}
```
"""
    messages = [{"role": "system", "content": system_prompt}]
    # Append history messages
    messages.extend(messages_history)
    # Append the current user query
    messages.append({"role": "user", "content": user_query})

    try:
        response = await async_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=messages,
            response_format={"type": "json_object"}
        )
        response_json = json.loads(response.choices[0].message.content)
        analysis = QueryAnalysis(**response_json)
        # Return a dictionary that directly updates AgentState
        return {
            "analysis_reasoning": analysis.reasoning,
            "decision": analysis.decision,
        }
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Error in analyze_query: {e}")
        return {
            "decision": ClarificationRequest(clarification_questions=["Sorry, I had trouble understanding that. Could you rephrase?"]),
            "final_response": "Sorry, I had trouble understanding that. Could you rephrase?",
        }


async def execute_search(state: AgentState) -> Dict[str, Any]:
    """
    Executes the search plan.
    """
    logger.info("Node: execute_search")
    decision = state["decision"]
    if not isinstance(decision, SearchPlan):
        return {"search_results": []} # Should not happen due to conditional routing

    primary_query = decision.queries[0].query
    
    # Initialize the search engines with proper dependencies
    try:
        # Initialize services
        vector_service = RealVectorService()
        async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        document_corpus = load_document_corpus_from_content_store()
        
        # Create search engines
        vector_search = VectorSearchEngine(client=async_client, vector_service=vector_service)
        keyword_search = KeywordSearchEngine(document_corpus=document_corpus)
        
        # Create RetrievalQuery objects
        retrieval_query = RetrievalQuery(
            query_text=primary_query,
            query_type="fusion",  # Use fusion to combine both vector and keyword
            max_results=5,
            min_relevance_score=0.3,
            target_domains=[state["jurisdiction"].lower()]  # Use jurisdiction from state
        )
        
        # Perform searches
        vector_results = await vector_search.search(retrieval_query)
        keyword_results = await keyword_search.search(retrieval_query)
        
        # Convert results to the expected format and combine
        combined_results = []
        
        # Process vector results
        for doc in vector_results:
            combined_results.append({
                'id': doc.source.document_id,
                'content': doc.content,
                'score': doc.relevance_score,
                'metadata': {
                    'text': doc.content,
                    'title': doc.source.title,
                    'section': doc.source.section,
                    'authority_level': doc.source.authority_level,
                    'jurisdiction': doc.source.jurisdiction,
                    'source_collection': doc.source.document_id.split('_')[0],
                    'retrieval_method': 'vector'
                }
            })
        
        # Process keyword results
        for doc in keyword_results:
            combined_results.append({
                'id': doc.source.document_id,
                'content': doc.content,
                'score': doc.relevance_score,
                'metadata': {
                    'text': doc.content,
                    'title': doc.source.title,
                    'section': doc.source.section,
                    'authority_level': doc.source.authority_level,
                    'jurisdiction': doc.source.jurisdiction,
                    'source_collection': doc.source.document_id.split('_')[0],
                    'retrieval_method': 'keyword'
                }
            })
        
        # Remove duplicates based on content
        unique_results = {}
        for result in combined_results:
            content_key = result['content'][:100]  # Use first 100 chars as key
            if content_key not in unique_results or result['score'] > unique_results[content_key]['score']:
                unique_results[content_key] = result
        
        # Sort by score and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Retrieved {len(sorted_results)} unique results for query: {primary_query}")
        return {"search_results": sorted_results[:5]}
        
    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
        return {"search_results": []}


async def generate_response(state: AgentState) -> Dict[str, Any]:
    """
    Generates a final response based on search results.
    """
    logger.info("Node: generate_response")
    query = state["user_query"]
    search_results = state.get("search_results", [])
    
    if not search_results:
        return {"final_response": "I could not find any relevant information to answer your question."}

    context_str = "\n\n".join([f"Source {i+1} (Page {r['metadata'].get('page_number', 'N/A')} of {r['metadata'].get('filename', 'Unknown')}):\n{r['metadata']['text']}" for i, r in enumerate(search_results)])
    
    # Store the search results in state so they can be used for source citations later
    state["used_sources"] = search_results
    
    system_prompt = "You are a helpful AI assistant for compliance professionals. Answer the user's query based *only* on the provided search results. Cite sources using individual citation markers like [1], [2], etc. Do not use compound citations like [1-3] or [1,2,3]."
    user_prompt = f"Query: {query}\n\nSearch Results:\n{context_str}\n\nAnswer:"

    try:
        response = await async_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        final_response = response.choices[0].message.content
        return {"final_response": final_response}
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return {"final_response": "I found some information, but had trouble summarizing it."}


async def format_clarification(state: AgentState) -> Dict[str, Any]:
    """
    Formats the clarification request.
    """
    logger.info("Node: format_clarification")
    decision = state["decision"]
    if not isinstance(decision, ClarificationRequest):
        return {"final_response": "I had an issue formulating my clarification questions. Could you please rephrase your query?"}

    questions = decision.clarification_questions
    questions_text = "\n".join([f"- {q}" for q in questions])
    full_text = f"To provide you with the most accurate guidance, I need a bit more information. Could you please clarify the following points?\n\n{questions_text}"
    return {"final_response": full_text}
