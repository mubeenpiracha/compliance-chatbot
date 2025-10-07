# backend/core/agent/nodes.py
import os
import json
import logging
import asyncio
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


def calculate_rrf_scores(vector_results: List, keyword_results: List, k: int = 60) -> List[Dict[str, Any]]:
    """
    Calculate Reciprocal Rank Fusion (RRF) scores for combining search results.
    RRF_score = 1 / (k + rank) where rank is 1-indexed position in each result list.
    """
    rrf_scores = {}  # document_id -> {"doc": doc_data, "rrf_score": float, "methods": set}
    
    # Process vector results (rank 1 = highest score)
    for rank, doc in enumerate(vector_results, 1):
        doc_id = doc.source.document_id
        rrf_contribution = 1.0 / (k + rank)
        
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "doc": {
                    'id': doc.source.document_id,
                    'content': doc.content,
                    'score': rrf_contribution,  # Start with RRF score
                    'metadata': {
                        'text': doc.content,
                        'title': doc.source.title,
                        'section': doc.source.section,
                        'authority_level': doc.source.authority_level,
                        'jurisdiction': doc.source.jurisdiction,
                        'checksum': doc.source.chunk_id,
                        'source_collection': doc.source.document_id.split('_')[0],
                        'retrieval_method': 'fusion',  # Mark as fusion result
                        'original_vector_score': doc.relevance_score,
                        'vector_rank': rank
                    }
                },
                "rrf_score": rrf_contribution,
                "methods": {"vector"}
            }
        else:
            # Document found in multiple methods - add RRF scores
            rrf_scores[doc_id]["rrf_score"] += rrf_contribution
            rrf_scores[doc_id]["methods"].add("vector")
            rrf_scores[doc_id]["doc"]["metadata"]["original_vector_score"] = doc.relevance_score
            rrf_scores[doc_id]["doc"]["metadata"]["vector_rank"] = rank
    
    # Process keyword results
    for rank, doc in enumerate(keyword_results, 1):
        doc_id = doc.source.document_id
        rrf_contribution = 1.0 / (k + rank)
        
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "doc": {
                    'id': doc.source.document_id,
                    'content': doc.content,
                    'score': rrf_contribution,
                    'metadata': {
                        'text': doc.content,
                        'title': doc.source.title,
                        'section': doc.source.section,
                        'authority_level': doc.source.authority_level,
                        'jurisdiction': doc.source.jurisdiction,
                        'checksum': doc.source.chunk_id,
                        'source_collection': doc.source.document_id.split('_')[0],
                        'retrieval_method': 'fusion',
                        'original_keyword_score': doc.relevance_score,
                        'keyword_rank': rank
                    }
                },
                "rrf_score": rrf_contribution,
                "methods": {"keyword"}
            }
        else:
            # Document found in multiple methods - add RRF scores
            rrf_scores[doc_id]["rrf_score"] += rrf_contribution
            rrf_scores[doc_id]["methods"].add("keyword")
            rrf_scores[doc_id]["doc"]["metadata"]["original_keyword_score"] = doc.relevance_score
            rrf_scores[doc_id]["doc"]["metadata"]["keyword_rank"] = rank
    
    # Update final scores and method information
    final_results = []
    for doc_data in rrf_scores.values():
        doc_data["doc"]["score"] = doc_data["rrf_score"]
        
        # Update retrieval method based on which methods found this document
        methods = doc_data["methods"]
        if len(methods) > 1:
            doc_data["doc"]["metadata"]["retrieval_method"] = "fusion"
            doc_data["doc"]["metadata"]["fusion_methods"] = list(methods)
        else:
            doc_data["doc"]["metadata"]["retrieval_method"] = list(methods)[0]
        
        final_results.append(doc_data["doc"])
    
    # Sort by RRF score (higher is better)
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results


async def execute_search(state: AgentState) -> Dict[str, Any]:
    """
    Executes the search plan using proper Reciprocal Rank Fusion (RRF).
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
        
        # Execute searches for all queries in the search plan and apply RRF per query
        all_rrf_results = []
        
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
            
            # Perform searches in parallel for better performance
            try:
                vector_results, keyword_results = await asyncio.gather(
                    vector_search.search(retrieval_query),
                    keyword_search.search(retrieval_query),
                    return_exceptions=True
                )
                
                # Handle partial failures gracefully
                if isinstance(vector_results, Exception):
                    logger.warning(f"Vector search failed for query {i+1}: {vector_results}")
                    vector_results = []
                    
                if isinstance(keyword_results, Exception):
                    logger.warning(f"Keyword search failed for query {i+1}: {keyword_results}")
                    keyword_results = []
                    
            except Exception as e:
                logger.error(f"Parallel search failed for query {i+1}, falling back to sequential: {e}")
                # Fallback to sequential execution
                try:
                    vector_results = await vector_search.search(retrieval_query)
                except Exception:
                    logger.error(f"Vector search fallback failed for query {i+1}")
                    vector_results = []
                    
                try:
                    keyword_results = await keyword_search.search(retrieval_query)
                except Exception:
                    logger.error(f"Keyword search fallback failed for query {i+1}")
                    keyword_results = []
            
            logger.info(f"Query {i+1}: Vector={len(vector_results)}, Keyword={len(keyword_results)} results")
            
            # Apply RRF to combine results for this query
            rrf_results = calculate_rrf_scores(vector_results, keyword_results)
            
            # Add query description to metadata
            for result in rrf_results:
                result['metadata']['query_description'] = search_query.description
            
            all_rrf_results.extend(rrf_results)
            logger.info(f"Query {i+1} RRF fusion produced {len(rrf_results)} results")
        
        # Final deduplication and ranking across all queries
        # Remove duplicates based on document ID (not content)
        unique_results = {}
        for result in all_rrf_results:
            doc_id = result['id']
            if doc_id not in unique_results or result['score'] > unique_results[doc_id]['score']:
                unique_results[doc_id] = result
        
        # If this is a reflection-triggered search, combine with previous results
        previous_results = state.get("search_results", [])
        if previous_results and state.get("needs_additional_search", False):
            logger.info(f"Combining {len(previous_results)} previous results with {len(unique_results)} new reflection results")
            # Add previous results to unique_results if not already present
            for prev_result in previous_results:
                prev_id = prev_result.get('id')
                if prev_id and prev_id not in unique_results:
                    unique_results[prev_id] = prev_result
        
        # Sort by RRF score and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Final RRF results: {len(sorted_results)} unique documents across {len(decision.queries)} queries")
        return {
            "search_results": sorted_results[:15],  # Increased to 15 to accommodate additional reflection results
            "needs_additional_search": False  # Reset the flag after processing
        }
        
    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
    reflection_analysis = state.get("reflection_analysis")
    previous_response = state.get("final_response") if reflection_analysis else None
    
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
    
    # Add reflection context if this is a regeneration
    if reflection_analysis and previous_response:
        conversation_context += f"\n--- REFLECTION CONTEXT ---\n"
        conversation_context += f"Previous response had incomplete information. Additional documents were retrieved to address:\n"
        for item in reflection_analysis.get("missing_items", []):
            conversation_context += f"- {item.get('description', 'Missing information')}\n"
        conversation_context += f"Previous response: {previous_response}\n"
        conversation_context += f"--- END REFLECTION CONTEXT ---\n\n"

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

{f'''
IMPORTANT - REFLECTION MODE ACTIVE:
This response is being regenerated because the previous response contained incomplete information. Additional regulatory documents have been retrieved to provide complete answers. Please:
- Integrate the new information with any relevant details from the previous response
- Provide complete and precise information rather than referring to "partial extracts"
- Include exact figures, complete definitions, and full regulatory requirements where available
- Focus on completing the missing information that was identified
''' if reflection_analysis else ''}

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
        
        # Check if the response indicates incomplete information that might need reflection
        incomplete_indicators = [
            "extract is partial", "should be confirmed against the full", "need the full",
            "complete text", "not visible in the provided extract", "requires the full",
            "must be verified", "detailed text", "entire document"
        ]
        
        response_lower = final_response.lower()
        needs_reflection = any(indicator.lower() in response_lower for indicator in incomplete_indicators)
        
        return {
            "final_response": final_response,
            "needs_additional_search": needs_reflection
        }
        
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


async def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the generated response to detect incomplete information and trigger additional searches.
    This node examines the response for phrases indicating partial extracts or missing documents.
    """
    logger.info("Node: reflection_node")
    
    final_response = state.get("final_response", "")
    jurisdiction = state["jurisdiction"]
    
    # Define patterns that indicate incomplete information
    incomplete_patterns = [
        "extract is partial",
        "should be confirmed against the full",
        "need the full",
        "complete text",
        "entire document",
        "full table",
        "complete formula",
        "full section",
        "detailed text",
        "complete definition",
        "exact amount",
        "precise figure",
        "specific number",
        "we need the",
        "requires the full",
        "must be verified",
        "not visible in the provided extract"
    ]
    
    # Check if the response contains any incomplete information indicators
    response_lower = final_response.lower()
    has_incomplete_info = any(pattern.lower() in response_lower for pattern in incomplete_patterns)
    
    if not has_incomplete_info:
        # No additional search needed
        return {"needs_additional_search": False}
    
    # Extract specific document references and section numbers
    system_prompt = f"""You are an expert at analyzing compliance responses to identify missing information and generate targeted search queries.

Analyze the following response and identify:
1. Specific documents mentioned that need to be retrieved in full
2. Section numbers, tables, or rules that are referenced but incomplete
3. Exact regulatory definitions or calculations that are missing

For each piece of missing information, generate a precise search query that would retrieve the complete document or section.

Response to analyze:
{final_response}

Generate your output as a JSON object with this structure:
{{
  "has_incomplete_info": true/false,
  "missing_items": [
    {{
      "type": "document|section|table|definition|calculation",
      "description": "What is missing",
      "search_query": "Precise search query to find the complete information",
      "priority": "high|medium|low"
    }}
  ]
}}"""

    try:
        response = await async_client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this response for incomplete information: {final_response}"}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        if not analysis.get("has_incomplete_info", False) or not analysis.get("missing_items"):
            return {"needs_additional_search": False}
        
        # Generate new search queries based on the missing information
        new_queries = []
        for item in analysis["missing_items"]:
            if item.get("priority") in ["high", "medium"]:  # Only pursue high/medium priority items
                query = SearchQuery(
                    query=item["search_query"],
                    description=f"Retrieve complete {item['type']}: {item['description']}"
                )
                new_queries.append(query)
        
        if new_queries:
            # Create a new search plan for the missing information
            search_plan = SearchPlan(queries=new_queries)
            logger.info(f"Reflection identified {len(new_queries)} additional searches needed")
            
            return {
                "decision": search_plan,
                "search_plan": search_plan,
                "needs_additional_search": True,
                "reflection_analysis": analysis
            }
        else:
            return {"needs_additional_search": False}
            
    except Exception as e:
        logger.error(f"Error in reflection_node: {e}")
        # If reflection fails, don't block the response
        return {"needs_additional_search": False}
