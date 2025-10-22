# backend/core/agent/nodes.py
import os
import json
import logging
import asyncio
import hashlib
import time
import traceback
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
from backend.core.performance_config import (
    get_optimized_semaphore, 
    PerformanceTimer, 
    time_async_operation,
    ASYNC_TIMEOUT,
    LLM_TIMEOUT,
    VECTOR_SEARCH_TIMEOUT,
    KEYWORD_SEARCH_TIMEOUT
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use async client for all OpenAI operations with optimized settings
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=ASYNC_TIMEOUT,
    max_retries=2
)

# Separate client for LLM calls with longer timeout
llm_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=LLM_TIMEOUT,  # 120 seconds for complex LLM queries
    max_retries=3
)

def create_content_hash(content: str) -> str:
    """
    Create a consistent hash for document content to use as deduplication key.
    Normalizes whitespace and creates a hash to handle slight content variations.
    """
    # Normalize whitespace and strip
    normalized_content = ' '.join(content.strip().split())
    # Create a hash for consistent deduplication
    return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()

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
        response = await llm_client.chat.completions.create(
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
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"Error in analyze_query ({error_type}): {error_msg}")
        
        # Handle timeout errors specifically
        if "timeout" in error_msg.lower() or "APITimeoutError" in error_type:
            return {
                "decision": ClarificationRequest(
                    clarification_questions=["The query analysis is taking longer than expected. Could you try rephrasing your question more concisely?"]
                ),
                "final_response": "The query analysis is taking longer than expected. Could you try rephrasing your question more concisely?",
            }
        
        # Handle other errors
        return {
            "decision": ClarificationRequest(
                clarification_questions=["Sorry, I had trouble understanding that. Could you rephrase?"]
            ),
            "final_response": "Sorry, I had trouble understanding that. Could you rephrase?",
        }


def calculate_rrf_scores(vector_results: List, keyword_results: List, k: int = 60) -> List[Dict[str, Any]]:
    """
    Calculate Reciprocal Rank Fusion (RRF) scores for combining search results.
    RRF_score = 1 / (k + rank) where rank is 1-indexed position in each result list.
    Uses content hash as deduplication key for reliable duplicate detection.
    Optimized for performance with larger result sets.
    """
    rrf_scores = {}  # content_hash -> {"doc": doc_data, "rrf_score": float, "methods": set}
    
    # Pre-calculate content hashes to avoid repeated computation
    vector_hashes = {}
    keyword_hashes = {}
    
    # Process vector results (rank 1 = highest score)
    for rank, doc in enumerate(vector_results, 1):
        content_key = create_content_hash(doc.content)
        vector_hashes[content_key] = doc
        rrf_contribution = 1.0 / (k + rank)
        
        rrf_scores[content_key] = {
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
    
    # Process keyword results
    for rank, doc in enumerate(keyword_results, 1):
        content_key = create_content_hash(doc.content)
        keyword_hashes[content_key] = doc
        rrf_contribution = 1.0 / (k + rank)
        
        if content_key not in rrf_scores:
            rrf_scores[content_key] = {
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
            rrf_scores[content_key]["rrf_score"] += rrf_contribution
            rrf_scores[content_key]["methods"].add("keyword")
            rrf_scores[content_key]["doc"]["metadata"]["original_keyword_score"] = doc.relevance_score
            rrf_scores[content_key]["doc"]["metadata"]["keyword_rank"] = rank
    
    # Update final scores and method information more efficiently
    final_results = []
    for content_key, doc_data in rrf_scores.items():
        doc_data["doc"]["score"] = doc_data["rrf_score"]
        
        # Update retrieval method based on which methods found this document
        methods = doc_data["methods"]
        if len(methods) > 1:
            doc_data["doc"]["metadata"]["retrieval_method"] = "fusion"
            doc_data["doc"]["metadata"]["fusion_methods"] = list(methods)
        else:
            doc_data["doc"]["metadata"]["retrieval_method"] = list(methods)[0]
        
        final_results.append(doc_data["doc"])
    
    # Sort by RRF score (higher is better) - using key function for better performance
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Log deduplication statistics
    total_inputs = len(vector_results) + len(keyword_results)
    fusion_count = sum(1 for doc_data in rrf_scores.values() if len(doc_data["methods"]) > 1)
    logger.info(f"RRF Stats: {total_inputs} inputs → {len(final_results)} unique ({fusion_count} fusion matches)")
    
    return final_results


async def execute_single_query_search(search_query, vector_search, keyword_search, jurisdiction: str, query_index: int) -> List[Dict[str, Any]]:
    """
    Execute search for a single query with both vector and keyword search in parallel.
    Returns RRF-combined results for this query.
    """
    query_start_time = time.time()
    logger.info(f"Executing query {query_index}: {search_query.query}")
    
    # Create RetrievalQuery object
    retrieval_query = RetrievalQuery(
        query_text=search_query.query,
        query_type="fusion",  # Use fusion to combine both vector and keyword
        max_results=5,
        min_relevance_score=0.3,
        target_domains=[jurisdiction.lower()]
    )
    
    # Perform searches in parallel for better performance with timeout
    search_start_time = time.time()
    try:
        # Add timeout to prevent hanging queries
        vector_results, keyword_results = await asyncio.wait_for(
            asyncio.gather(
                vector_search.search(retrieval_query),
                keyword_search.search(retrieval_query),
                return_exceptions=True
            ),
            timeout=25.0  # 25 second timeout per query (5s buffer from individual timeouts)
        )
        
        # Handle partial failures with better error classification
        if isinstance(vector_results, Exception):
            error_type = type(vector_results).__name__
            logger.error(f"Vector search failed for query {query_index} [{error_type}]: {vector_results}")
            vector_results = []
            
        if isinstance(keyword_results, Exception):
            error_type = type(keyword_results).__name__
            logger.error(f"Keyword search failed for query {query_index} [{error_type}]: {keyword_results}")
            keyword_results = []
            
    except asyncio.TimeoutError:
        logger.error(f"Query {query_index} timed out after 25s - both searches failed")
        vector_results = []
        keyword_results = []
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Parallel search failed for query {query_index} [{error_type}]: {e}")
        vector_results = []
        keyword_results = []
    
    search_end_time = time.time()
    search_duration = search_end_time - search_start_time
    logger.info(f"Query {query_index}: Vector={len(vector_results)}, Keyword={len(keyword_results)} results | Search time: {search_duration:.3f}s")
    
    # Apply RRF to combine results for this query
    rrf_start_time = time.time()
    rrf_results = calculate_rrf_scores(vector_results, keyword_results)
    rrf_duration = time.time() - rrf_start_time
    
    # Add query description to metadata
    for result in rrf_results:
        result['metadata']['query_description'] = search_query.description
        result['metadata']['query_index'] = query_index
    
    query_total_time = time.time() - query_start_time
    logger.info(f"Query {query_index} completed: RRF fusion produced {len(rrf_results)} results | RRF time: {rrf_duration:.3f}s | Total query time: {query_total_time:.3f}s")
    
    return rrf_results


async def execute_search(state: AgentState) -> Dict[str, Any]:
    """
    Executes the search plan using proper Reciprocal Rank Fusion (RRF) with full parallelization.
    All queries are executed in parallel for maximum performance.
    """
    logger.info("Node: execute_search")
    decision = state["decision"]
    if not isinstance(decision, SearchPlan):
        return {"search_results": []} # Should not happen due to conditional routing

    # Initialize the search engines with proper dependencies
    try:
        # Initialize services
        vector_service = RealVectorService()
        # Use module-level async_client (already configured with timeout and retries)
        # DO NOT create a new client here - it bypasses connection pooling
        
        # Load document corpus filtered by user's selected jurisdiction for efficiency
        jurisdiction = state.get("jurisdiction", "ADGM")  # Default to ADGM if not specified
        document_corpus = load_document_corpus_from_content_store(jurisdiction=jurisdiction)
        logger.info(f"Loaded document corpus filtered by jurisdiction: {jurisdiction}")
        
        # Create semaphore to limit concurrent OpenAI embedding API calls (prevents rate limiting)
        embedding_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent embeddings
        logger.info(f"Created embedding semaphore with limit of 3 concurrent API calls")
        
        # Create search engines (pass vector_service to keyword_search for metadata consistency)
        vector_search = VectorSearchEngine(
            client=async_client, 
            vector_service=vector_service,
            semaphore=embedding_semaphore
        )
        keyword_search = KeywordSearchEngine(document_corpus=document_corpus, vector_service=vector_service)
        
        total_search_start = time.time()
        logger.info(f"Starting parallel execution of {len(decision.queries)} queries")
        
        # Execute ALL queries in parallel for maximum performance
        query_tasks = []
        for i, search_query in enumerate(decision.queries, 1):
            task = execute_single_query_search(
                search_query, 
                vector_search, 
                keyword_search, 
                state["jurisdiction"], 
                i
            )
            query_tasks.append(task)
        
        # Wait for all queries to complete in parallel with global timeout
        parallel_start_time = time.time()
        try:
            all_query_results = await asyncio.wait_for(
                asyncio.gather(*query_tasks, return_exceptions=True),
                timeout=90.0  # 90 second global timeout for all parallel queries
            )
        except asyncio.TimeoutError:
            logger.error(f"Global timeout: Parallel execution exceeded 90s for {len(query_tasks)} queries")
            # Gather completed tasks if any
            all_query_results = [task.result() if task.done() else [] for task in query_tasks]
        parallel_duration = time.time() - parallel_start_time
        
        # Process results and handle any exceptions
        all_rrf_results = []
        successful_queries = 0
        
        for i, result in enumerate(all_query_results, 1):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
            else:
                all_rrf_results.extend(result)
                successful_queries += 1
        
        logger.info(f"PARALLEL EXECUTION: {successful_queries}/{len(decision.queries)} queries successful | Parallel time: {parallel_duration:.3f}s")
        
        # Final deduplication and ranking across all queries
        dedup_start_time = time.time()
        unique_results = {}
        duplicates_found = 0
        
        for result in all_rrf_results:
            content_key = create_content_hash(result.get('content', ''))
            if content_key not in unique_results:
                unique_results[content_key] = result
            elif result['score'] > unique_results[content_key]['score']:
                logger.debug(f"Replacing duplicate with higher score: {result['score']} > {unique_results[content_key]['score']}")
                unique_results[content_key] = result
                duplicates_found += 1
            else:
                duplicates_found += 1
        
        dedup_duration = time.time() - dedup_start_time
        
        if duplicates_found > 0:
            logger.info(f"Deduplication: Found {duplicates_found} duplicates, final unique count: {len(unique_results)} | Dedup time: {dedup_duration:.3f}s")
        
        # If this is a reflection-triggered search, combine with previous results
        previous_results = state.get("search_results", [])
        if previous_results and state.get("needs_additional_search", False):
            logger.info(f"Combining {len(previous_results)} previous results with {len(unique_results)} new reflection results")
            # Add previous results to unique_results if not already present (by content hash)
            for prev_result in previous_results:
                prev_content_hash = create_content_hash(prev_result.get('content', ''))
                if prev_content_hash and prev_content_hash not in unique_results:
                    unique_results[prev_content_hash] = prev_result
        
        # Sort by RRF score and limit results
        sort_start_time = time.time()
        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0.0), reverse=True)
        sort_duration = time.time() - sort_start_time
        
        total_search_time = time.time() - total_search_start
        
        logger.info(f"SEARCH PERFORMANCE SUMMARY:")
        logger.info(f"  - Total execution time: {total_search_time:.3f}s")
        logger.info(f"  - Parallel queries time: {parallel_duration:.3f}s ({parallel_duration/total_search_time*100:.1f}%)")
        logger.info(f"  - Deduplication time: {dedup_duration:.3f}s ({dedup_duration/total_search_time*100:.1f}%)")
        logger.info(f"  - Sorting time: {sort_duration:.3f}s ({sort_duration/total_search_time*100:.1f}%)")
        logger.info(f"  - Final results: {len(sorted_results)} unique documents from {len(decision.queries)} parallel queries")
        logger.info(f"  - Speedup: ~{len(decision.queries)}x faster than sequential execution")
        
        return {
            "search_results": sorted_results[:15],  # Increased to 15 to accommodate additional reflection results
            "needs_additional_search": False  # Reset the flag after processing
        }
        
    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
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

    # Limit search results to prevent token overflow
    MAX_SEARCH_RESULTS = 15  # Limit to prevent prompt from being too large
    if len(search_results) > MAX_SEARCH_RESULTS:
        logger.warning(f"Truncating search results from {len(search_results)} to {MAX_SEARCH_RESULTS} to prevent token overflow")
        search_results = search_results[:MAX_SEARCH_RESULTS]

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
    MAX_CONTENT_LENGTH = 2000  # Max characters per source content
    for i, result in enumerate(search_results):
        metadata = result.get('metadata', {})
        
        # Truncate content if it's too long
        content = metadata.get('text', result.get('content', ''))
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "... [content truncated]"
            logger.debug(f"Truncated source {i+1} content from {len(metadata.get('text', result.get('content', '')))} to {MAX_CONTENT_LENGTH} chars")
        
        source_info = (
            f"**Source {i+1}** "
            f"[{metadata.get('title', 'Unknown Document')} - "
            f"Section: {metadata.get('section', 'N/A')}]\n"
            f"Authority Level: {metadata.get('authority_level', 'N/A')}\n"
            f"Jurisdiction: {metadata.get('jurisdiction', 'N/A')}\n"
            f"Content: {content}\n"
        )
        
        # Add query description if available (helps understand why this source was retrieved)
        if metadata.get('query_description'):
            source_info += f"Retrieved for: {metadata.get('query_description')}\n"
            
        sources_context += source_info + "\n---\n\n"
    
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

    # Synthesis-specific retry logic with extended timeout
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Log prompt sizes for debugging
            if retry_count == 0:
                system_prompt_tokens = len(system_prompt) // 4  # Rough estimate
                user_prompt_tokens = len(user_prompt) // 4
                logger.info(f"Synthesis prompt sizes - System: ~{system_prompt_tokens} tokens, User: ~{user_prompt_tokens} tokens, Total: ~{system_prompt_tokens + user_prompt_tokens} tokens")
                logger.info(f"Number of search results: {len(search_results)}")
            else:
                logger.info(f"Synthesis retry attempt {retry_count + 1}/{max_retries}")
            
            # Create a custom client with extended timeout for synthesis (90 seconds)
            from openai import AsyncOpenAI
            synthesis_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=90.0,  # Extended timeout for synthesis
                max_retries=0  # We handle retries ourselves
            )
            
            response = await synthesis_client.chat.completions.create(
                model="gpt-5-2025-08-07",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            final_response = response.choices[0].message.content
            logger.info(f"Successfully generated synthesis response on attempt {retry_count + 1}")
            
            # Only trigger reflection if this is not already a reflection-generated response
            reflection_count = state.get("reflection_count", 0)
            logger.info(f"🔄 Reflection check - current reflection_count: {reflection_count}")
            
            # Skip reflection decision - will be handled by separate reflection_decision node
            # This keeps synthesis focused on generating the response
            
            return {
                "final_response": final_response,
                "needs_additional_search": False,  # Will be set by reflection_decision node
                "used_sources": search_results  # Store sources for citation tracking
            }
            
        except Exception as e:
            last_error = e
            retry_count += 1
            error_type = type(e).__name__
            error_msg = str(e).lower()
            
            logger.error(f"Synthesis attempt {retry_count}/{max_retries} failed: {error_type} - {e}")
            
            # For timeout errors, retry with progressively reduced context
            if "timeout" in error_msg and retry_count < max_retries:
                # Reduce number of search results by 30% each retry
                reduction_factor = 0.7 ** retry_count
                max_results_for_retry = max(3, int(len(search_results) * reduction_factor))
                
                if max_results_for_retry < len(search_results):
                    logger.info(f"Reducing search results from {len(search_results)} to {max_results_for_retry} for retry {retry_count}")
                    search_results = search_results[:max_results_for_retry]
                    
                    # Rebuild sources_context with reduced results
                    sources_context = ""
                    MAX_CONTENT_LENGTH = 2000
                    for i, result in enumerate(search_results):
                        metadata = result.get('metadata', {})
                        content = metadata.get('text', result.get('content', ''))
                        if len(content) > MAX_CONTENT_LENGTH:
                            content = content[:MAX_CONTENT_LENGTH] + "... [content truncated]"
                        
                        source_info = (
                            f"**Source {i+1}** "
                            f"[{metadata.get('title', 'Unknown Document')} - "
                            f"Section: {metadata.get('section', 'N/A')}]\n"
                            f"Authority Level: {metadata.get('authority_level', 'N/A')}\n"
                            f"Jurisdiction: {metadata.get('jurisdiction', 'N/A')}\n"
                            f"Content: {content}\n"
                        )
                        if metadata.get('query_description'):
                            source_info += f"Retrieved for: {metadata.get('query_description')}\n"
                        sources_context += source_info + "\n---\n\n"
                    
                    # Rebuild user prompt
                    user_prompt = f"""Please analyze the following conversation and provide a comprehensive response:

{conversation_context}

Available Regulatory Sources:
{sources_context}

Provide a detailed, well-structured response that synthesizes the available information to address the user's current query while considering the conversation context."""
                    
                    continue  # Retry with reduced content
            
            # If not a timeout or out of retries, break
            if retry_count >= max_retries:
                break
    
    # All retries exhausted - return error with sources intact
    logger.error(f"Synthesis failed after {max_retries} attempts")
    logger.error(f"Final error type: {type(last_error).__name__}")
    logger.error(f"Full traceback:", exc_info=True)
    
    # Check for specific error types and provide helpful messages
    error_msg = str(last_error).lower() if last_error else ""
    if "timeout" in error_msg:
        error_response = "I retrieved relevant documents but the response generation timed out. The documents are complex - please try:\n\n1. Asking a more specific question\n2. Focusing on one aspect at a time\n3. Checking the sources below for direct information"
    elif "token" in error_msg or "length" in error_msg:
        error_response = "The retrieved documents contain extensive information that exceeded processing limits. Please:\n\n1. Narrow your query to a specific topic\n2. Ask about one regulation at a time\n3. Review the sources below for relevant sections"
    elif "rate" in error_msg or "quota" in error_msg:
        error_response = "API rate limit reached. Please wait a moment and try again, or review the sources below directly."
    else:
        error_response = "I found relevant regulatory information (see sources below), but encountered a technical issue generating the synthesis. Please try:\n\n1. Rephrasing your question\n2. Being more specific about what you need\n3. Reviewing the source documents directly"
    
    # CRITICAL: Always return used_sources so sources are displayed even on error
    return {
        "final_response": error_response,
        "needs_additional_search": False,
        "used_sources": search_results  # Ensure sources are preserved in error cases
    }


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


async def reflection_decision_node(state: AgentState) -> Dict[str, Any]:
    """
    Lightweight decision node: Determines if the response needs additional searches.
    Uses a mini OpenAI call to intelligently assess if reflection is needed.
    This is more reliable than pattern matching and cheaper than full reflection analysis.
    """
    logger.info("Node: reflection_decision_node")
    
    final_response = state.get("final_response", "")
    user_query = state.get("user_query", "")
    history = state.get("messages", [])
    reflection_count = state.get("reflection_count", 0)
    
    # Never reflect more than once to prevent loops
    if reflection_count > 0:
        logger.info(f"🔄 Skipping reflection - already performed {reflection_count} time(s)")
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }
    
    # Quick sanity checks before making OpenAI call
    if len(final_response) < 100:
        logger.info("🔄 Response too short, likely an error - skipping reflection")
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }
    
    # Build conversation context for better evaluation
    conversation_context = ""
    if history and len(history) > 0:
        conversation_context = "Conversation History:\n"
        for msg in history[-4:]:  # Last 4 messages for context (keep it concise)
            sender = msg.get("sender", "unknown")
            text = msg.get("text", "")
            if text:
                conversation_context += f"{sender.upper()}: {text[:200]}{'...' if len(text) > 200 else ''}\n"
        conversation_context += f"\nUSER (current): {user_query}\n"
    else:
        conversation_context = f"User's Question: {user_query}"
    
    # Mini OpenAI call to decide if reflection is needed
    decision_prompt = f"""You are a quality checker for regulatory compliance responses. Your job is to quickly determine if a response is INCOMPLETE and needs additional document retrieval.

{conversation_context}

Generated Response: {final_response}

Respond with a JSON object:
{{
  "needs_reflection": true/false,
  "reason": "brief explanation (1 sentence)",
  "confidence": "high/medium/low"
}}

Mark needs_reflection=true ONLY if:
1. Response explicitly states information is partial/incomplete/missing
2. Response mentions needing full documents/sections/tables that weren't provided
3. Response indicates specific regulatory text is needed but not available
4. Response is suspiciously vague despite specific query

Mark needs_reflection=false if:
- Response fully answers the question with cited sources
- Response acknowledges limitations but provides available information
- Response uses standard regulatory language ("complete definition", "full scope") without indicating missing data
- Any uncertainty is about interpretation, not missing documents"""

    try:
        # Lightweight call with short response - use custom client with extended timeout
        # for conversation history processing (60s is generous for gpt-4o-mini but safe)
        from openai import AsyncOpenAI
        reflection_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,  # 60 second timeout for reflection decision (includes history context)
            max_retries=2
        )
        
        decision_response = await reflection_client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini model for speed and cost
            messages=[
                {"role": "user", "content": decision_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=100  # Keep it short - just need yes/no + reason
        )
        
        decision = json.loads(decision_response.choices[0].message.content)
        needs_reflection = decision.get("needs_reflection", False)
        reason = decision.get("reason", "No reason provided")
        confidence = decision.get("confidence", "medium")
        
        logger.info(f"🔄 Reflection decision: {needs_reflection} ({confidence} confidence) - {reason}")
        
        if needs_reflection:
            # Increment reflection count
            return {
                "needs_additional_search": True,
                "reflection_count": reflection_count,
                "reflection_reason": reason
            }
        else:
            return {
                "needs_additional_search": False,
                "reflection_count": reflection_count
            }
            
    except Exception as e:
        logger.error(f"Error in reflection_decision_node: {e}")
        # On error, don't block the response - assume no reflection needed
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }


async def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the response to generate specific search queries for missing information.
    Called ONLY if reflection_decision_node determined reflection is needed.
    Uses targeted OpenAI call to extract what's missing and generate search queries.
    """
    logger.info("Node: reflection_node")
    
    final_response = state.get("final_response", "")
    user_query = state.get("user_query", "")
    history = state.get("messages", [])
    jurisdiction = state.get("jurisdiction", "DIFC")
    reflection_reason = state.get("reflection_reason", "Information appears incomplete")
    
    # Increment reflection count
    reflection_count = state.get("reflection_count", 0) + 1
    
    # Build conversation context
    conversation_context = ""
    if history and len(history) > 0:
        conversation_context = "Conversation History:\n"
        for msg in history[-4:]:  # Last 4 messages for context
            sender = msg.get("sender", "unknown")
            text = msg.get("text", "")
            if text:
                conversation_context += f"{sender.upper()}: {text[:300]}{'...' if len(text) > 300 else ''}\n"
        conversation_context += f"\nUSER (current): {user_query}\n"
    else:
        conversation_context = f"User's Question: {user_query}"

    # Targeted prompt to extract missing information and generate queries
    system_prompt = f"""You are an expert at analyzing compliance responses to identify missing information and generate targeted search queries.

{conversation_context}

Generated Response: {final_response}

Reflection Trigger: {reflection_reason}

Analyze the response in the context of the full conversation and identify what specific information is missing. For each missing piece, generate a precise search query.

Output JSON format:
{{
  "missing_items": [
    {{
      "type": "document|section|table|definition|calculation",
      "description": "Brief description of what's missing",
      "search_query": "Precise search query including jurisdiction and specific reference",
      "priority": "high|medium"
    }}
  ]
}}

Focus on:
1. Specific document names, codes, or section numbers mentioned as needed
2. Tables or schedules referenced but not provided
3. Definitions explicitly stated as incomplete
4. Calculations or formulas mentioned but not shown

Generate 1-3 targeted queries maximum. Be specific - include document codes, section numbers, and {jurisdiction} jurisdiction."""

    try:
        # Use custom client with extended timeout for reflection analysis with conversation history
        from openai import AsyncOpenAI
        reflection_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,  # 60 second timeout for reflection analysis (includes history + query generation)
            max_retries=2
        )
        
        response = await reflection_client.chat.completions.create(
            model="gpt-4o",  # Use faster model for extraction
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract missing information and generate search queries."}
            ],
            response_format={"type": "json_object"},
            max_tokens=500  # Limit to keep focused
        )
        
        analysis = json.loads(response.choices[0].message.content)
        missing_items = analysis.get("missing_items", [])
        
        if not missing_items:
            logger.info("Reflection analysis found no specific missing items")
            return {
                "needs_additional_search": False,
                "reflection_count": reflection_count
            }
        
        # Generate new search queries (only high/medium priority, max 3)
        new_queries = []
        for item in missing_items[:3]:
            if item.get("priority") in ["high", "medium"]:
                query = SearchQuery(
                    query=item["search_query"],
                    description=f"Reflection: {item.get('description', 'Additional information needed')}"
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
                "reflection_analysis": analysis,
                "reflection_count": reflection_count
            }
        else:
            return {
                "needs_additional_search": False,
                "reflection_count": reflection_count
            }
            
    except Exception as e:
        logger.error(f"Error in reflection_node: {e}")
        # If reflection fails, don't block the response
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }
