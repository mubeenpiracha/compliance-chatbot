# backend/core/agent/nodes.py
import os
import json
import logging
from typing import List, Dict, Any, Union
from openai import OpenAI
from pydantic import ValidationError

from backend.core.agent.state import AgentState
from backend.core.models.agent_models import QueryAnalysis, SearchPlan, ClarificationRequest, SearchQuery
from backend.core.retrieval.vector_search import VectorSearch
from backend.core.retrieval.keyword_search import KeywordSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_query(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's query to decide the next step.
    """
    logger.info("Node: analyze_query")
    user_query = state["user_query"]
    # Correctly access messages from the state
    messages_history = state.get("messages", [])
    jurisdiction = state["jurisdiction"]

    system_prompt = f"""
You are an expert compliance analyst AI. Your task is to analyze a user's query about financial regulations in the specified jurisdiction: {jurisdiction}.
Based on the user's query and the conversation history, you must decide on one of two possible actions:

1.  **Search**: If the query is clear, specific, and actionable, create a `SearchPlan`.
2.  **Clarify**: If the query is ambiguous, vague, or lacks essential details, create a `ClarificationRequest`.

**Conversation History Analysis**:
- Review the provided conversation history.
- If the AI's last message was a clarification request, the user's current message is likely an answer.
- Use this new information to re-evaluate the original query.

**Instructions**:
- Your output **MUST** be a single JSON object that validates against the `QueryAnalysis` Pydantic model.
- Provide a concise `reasoning` for your decision.

**Pydantic Models for your reference**:
```python
class SearchQuery(BaseModel):
    query: str
    description: str

class SearchPlan(BaseModel):
    queries: List[SearchQuery]

class ClarificationRequest(BaseModel):
    clarification_questions: List[str]

class QueryAnalysis(BaseModel):
    reasoning: str
    decision: Union[SearchPlan, ClarificationRequest]
```
"""
    messages = [{"role": "system", "content": system_prompt}]
    # Append history messages
    messages.extend(messages_history)
    # Append the current user query
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
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
            "decision": "error",
            "final_response": "Sorry, I had trouble understanding that. Could you rephrase?",
        }


def execute_search(state: AgentState) -> Dict[str, Any]:
    """
    Executes the search plan.
    """
    logger.info("Node: execute_search")
    decision = state["decision"]
    if not isinstance(decision, SearchPlan):
        return {"search_results": []} # Should not happen due to conditional routing

    primary_query = decision.queries[0].query
    
    vector_search = VectorSearch()
    keyword_search = KeywordSearch()
    
    vector_results = vector_search.search(primary_query, top_k=5)
    keyword_results = keyword_search.search(primary_query, top_k=5)
    
    combined_results = vector_results + keyword_results
    unique_results = list({r['metadata']['text']: r for r in combined_results}.values())
    sorted_results = sorted(unique_results, key=lambda x: x.get('score', 0.0), reverse=True)
    
    return {"search_results": sorted_results[:5]}


def generate_response(state: AgentState) -> Dict[str, Any]:
    """
    Generates a final response based on search results.
    """
    logger.info("Node: generate_response")
    query = state["user_query"]
    search_results = state["search_results"]
    
    if not search_results:
        return {"final_response": "I could not find any relevant information to answer your question."}

    context_str = "\n\n".join([f"Source {i+1} (Page {r['metadata'].get('page_number', 'N/A')} of {r['metadata'].get('filename', 'Unknown')}):\n{r['metadata']['text']}" for i, r in enumerate(search_results)])
    
    # Store the search results in state so they can be used for source citations later
    state["used_sources"] = search_results
    
    system_prompt = "You are a helpful AI assistant for compliance professionals. Answer the user's query based *only* on the provided search results. Cite sources using individual citation markers like [1], [2], etc. Do not use compound citations like [1-3] or [1,2,3]."
    user_prompt = f"Query: {query}\n\nSearch Results:\n{context_str}\n\nAnswer:"

    try:
        response = client.chat.completions.create(
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


def format_clarification(state: AgentState) -> Dict[str, Any]:
    """
    Formats the clarification request.
    """
    logger.info("Node: format_clarification")
    decision = state["decision"]
    if not isinstance(decision, ClarificationRequest):
        return {"final_response": "I had an issue formulating my clarification questions. Could you please rephrase your query?"}

    questions = decision.clarification_questions
    questions_text = "\n".join(f"- {q}" for q in questions)
    full_text = f"To provide you with the most accurate guidance, I need a bit more information. Could you please clarify the following points?\n\n{questions_text}"
    return {"final_response": full_text}
