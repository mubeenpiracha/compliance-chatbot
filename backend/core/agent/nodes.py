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
You are an expert AI compliance analyst responsible for interpreting user queries about financial regulations within the {jurisdiction} jurisdiction and determining the appropriate response path.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.


Based on the user's message and the conversation history, select one of these two actions:


1. **Search**: If the user query is clear, actionable, and specific, generate a `SearchPlan`. This consists of one or more search queries. Each search query must include both a `query` (the search string) and a `description` (which clarifies the query's focus). Maintain the logical or user-requested order for multiple queries.
2. **Clarify**: If the user's input is ambiguous, vague, incomplete, or lacks critical information, generate a `ClarificationRequest`.

Guidance for Conversation History:
- Analyze the full conversation history.
- If the AI's prior message requested clarification, treat the user's current message as an answer and reassess the original request with this new information.
- If the user has provided additional context or information, incorporate that into your analysis.

After generating your output, validate that your JSON matches the required structure, and self-correct if you detect formatting or conformity issues before returning your final output.

Output Requirements:
- Your output must be a single JSON object that strictly conforms to the following structure (as defined by the `QueryAnalysis` schema):
- `reasoning`: string. A brief rationale for your chosen action.
- `decision`: object. One of:
- `type: "search_plan"` with a `queries` list: Each item must be a dictionary with both `query` and `description` (both strings). If any search is missing these fields, flag as a formatting error and provide an empty `queries` list.
- `type: "clarification_request"` with a `clarification_questions` list: Each question must be a non-empty string.
- If the input is unexpected, invalid, or unmappable to search or clarification, generate a clarification request to elicit the required information.


Output Format Example:
Search example:
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

*Clarification Example*:
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
            model="gpt-5-2025-08-07",
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

    # Initialize the search engines with proper dependencies
    try:
        # Initialize services
        vector_service = RealVectorService()
        async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        document_corpus = load_document_corpus_from_content_store()
        
        # Create search engines (pass vector_service to keyword_search for metadata consistency)
        vector_search = VectorSearchEngine(client=async_client, vector_service=vector_service)
        keyword_search = KeywordSearchEngine(document_corpus=document_corpus, vector_service=vector_service)
        
        # Execute searches for all queries in the search plan
        all_combined_results = []
        
        for i, search_query in enumerate(decision.queries):
            logger.info(f"Executing query {i+1}/{len(decision.queries)}: {search_query.query}")
            
            # Create RetrievalQuery objects
            retrieval_query = RetrievalQuery(
                query_text=search_query.query,
                query_type="fusion",  # Use fusion to combine both vector and keyword
                max_results=10,
                min_relevance_score=0.3,
                target_domains=[state["jurisdiction"].lower()]  # Use jurisdiction from state
            )
            
            # Perform searches
            vector_results = await vector_search.search(retrieval_query)
            keyword_results = await keyword_search.search(retrieval_query)
            
            # Convert results to the expected format and combine
            query_results = []
            
            # Process vector results
            for doc in vector_results:
                query_results.append({
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
                        'retrieval_method': 'vector',
                        'query_description': search_query.description
                    }
                })
            
            # Process keyword results
            for doc in keyword_results:
                query_results.append({
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
                        'retrieval_method': 'keyword',
                        'query_description': search_query.description
                    }
                })
            
            all_combined_results.extend(query_results)
            logger.info(f"Query {i+1} returned {len(query_results)} results")
        
        # Remove duplicates based on content
        unique_results = {}
        for result in all_combined_results:
            content_key = result['content'][:100]  # Use first 100 chars as key
            if content_key not in unique_results or result['score'] > unique_results[content_key]['score']:
                unique_results[content_key] = result
        
        # Sort by score and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Retrieved {len(sorted_results)} unique results across {len(decision.queries)} queries")
        return {"search_results": sorted_results[:10]}  # Increased limit since we're searching multiple queries
        
    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
        return {"search_results": []}


async def generate_response(state: AgentState) -> Dict[str, Any]:
    """
    Synthesis node that generates a comprehensive response based on conversation history,
    current query, and retrieved search results. This node considers the full context
    of the conversation to provide contextually aware responses.
    """
    logger.info("Node: generate_response (synthesis)")
    
    current_query = state["user_query"]
    search_results = state.get("search_results", [])
    conversation_history = state.get("messages", [])
    jurisdiction = state["jurisdiction"]
    
    if not search_results:
        return {"final_response": "I could not find any relevant information to answer your question based on the available regulatory documents."}

    # Build conversation context from history
    conversation_context = ""
    if conversation_history:
        conversation_context = "Previous conversation:\n"
        for i, msg in enumerate(conversation_history[-6:]):  # Last 6 messages for context
            sender = msg.get("sender", "unknown")
            content = msg.get("text", "")
            role_label = "User" if sender == "user" else "Assistant"
            conversation_context += f"{role_label}: {content}\n"
        conversation_context += f"\nCurrent User Query: {current_query}\n"
    else:
        conversation_context = f"User Query: {current_query}\n"

    # Format search results with enhanced metadata
    sources_context = ""
    for i, result in enumerate(search_results):
        metadata = result.get('metadata', {})
        source_info = (
            f"**Source {i+1}** "
            f"[{metadata.get('title', 'Unknown Document')} - "
            f"Section: {metadata.get('section', 'N/A')}]\n"
            f"Authority Level: {metadata.get('authority_level', 'N/A')}\n"
            f"Jurisdiction: {metadata.get('jurisdiction', 'N/A')}\n"
            f"Content: {metadata.get('text', result.get('content', ''))}\n"
        )
        
        # Add query description if available (helps understand why this source was retrieved)
        if metadata.get('query_description'):
            source_info += f"Retrieved for: {metadata.get('query_description')}\n"
            
        sources_context += source_info + "\n---\n\n"
    
    # Store the search results in state for source citations
    state["used_sources"] = search_results
    
    system_prompt = f"""You are an expert AI compliance advisor specializing in {jurisdiction} financial regulations. 

Your task is to synthesize information from the conversation history and retrieved regulatory sources to provide a comprehensive, contextually-aware response.

Key Instructions:
1. **Context Awareness**: Consider the full conversation history to understand the user's broader needs and any previous clarifications or follow-up questions.
2. **Source-Based Responses**: Base your answer ONLY on the provided search results from official regulatory documents.
3. **Citation Requirements**: 
   - Use individual citation markers like [1], [2], etc. 
   - Do not use compound citations like [1-3] or [1,2,3]
   - Each citation should correspond to a specific source
4. **Synthesis Approach**:
   - Connect information across multiple sources when relevant
   - Address the current query while considering previous conversation context
   - Identify patterns, relationships, or contradictions across sources
   - Provide actionable guidance where appropriate
5. **Regulatory Precision**: Be precise about regulatory requirements, noting any jurisdictional specifics, effective dates, or conditional applications.
6. **Clarity**: Structure your response clearly with headings, bullet points, or numbered lists when appropriate.

If the search results don't fully address the query, acknowledge the limitations and suggest what additional information might be needed."""

    user_prompt = f"""Please analyze the following conversation and provide a comprehensive response:

{conversation_context}

Available Regulatory Sources:
{sources_context}

Provide a detailed, well-structured response that synthesizes the available information to address the user's current query while considering the conversation context."""

    try:
        response = await async_client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
    
        )
        final_response = response.choices[0].message.content
        logger.info("Successfully generated synthesis response")
        return {"final_response": final_response}
        
    except Exception as e:
        logger.error(f"Error in generate_response synthesis: {e}")
        return {"final_response": "I found relevant information in the regulatory documents, but encountered an issue while synthesizing the response. Please try rephrasing your question or contact support if the issue persists."}


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
