# backend/core/graph_agent.py
import os
from typing import List, TypedDict, Literal, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Configuration ---
# It's good practice to load API keys from environment variables.
# Ensure the OPENAI_API_KEY environment variable is set.
from backend.core.config import OPENAI_API_KEY # Assuming this works
# For standalone testing, you can use:
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Agent State ---
# Updated to support more flexible, natural conversation flow
class AgentState(TypedDict):
    """
    Represents the state of our agent. This state is passed between nodes in the graph.
    """
    original_query: str
    jurisdiction: str
    chat_history: List[BaseMessage]
    contextualized_query: str
    query_confidence: float  # 0.0-1.0 confidence in understanding the query
    query_type: Literal["GREETING", "COMPLIANCE", "EXPLORATORY", "CLARIFICATION"]
    retrieved_docs: List[Document]
    conversational_response: str
    compliance_response: str
    final_response: str
    final_response_with_sources: dict
    should_retrieve: bool  # Whether to attempt document retrieval
    should_converse: bool  # Whether to provide conversational response
    exploration_suggestions: List[str]  # Suggestions for further exploration

# --- Pydantic Models for Structured LLM Output ---
# Updated models for more flexible conversation handling

class QueryAnalysis(BaseModel):
    """Analysis of user query with confidence scoring."""
    query_type: Literal["GREETING", "COMPLIANCE", "EXPLORATORY", "CLARIFICATION"] = Field(
        description="The type of query: GREETING (hello, thanks), COMPLIANCE (specific regulatory question), EXPLORATORY (general browsing), CLARIFICATION (follow-up)"
    )
    confidence: float = Field(
        description="Confidence in understanding the query (0.0-1.0)"
    )
    should_retrieve: bool = Field(
        description="Whether this query would benefit from document retrieval"
    )
    should_converse: bool = Field(
        description="Whether this query would benefit from conversational response"
    )
    key_topics: List[str] = Field(
        description="Key compliance topics mentioned in the query"
    )

class ExplorationSuggestions(BaseModel):
    """Suggestions for further exploration of compliance topics."""
    suggestions: List[str] = Field(
        description="List of helpful follow-up questions or topics the user might explore"
    )


# --- LLM and Tool Initialization ---
# Updated chains for flexible conversation handling
llm_reasoning = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
llm_utility = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Structured LLM callers for new flexible approach
query_analysis_chain = llm_utility.with_structured_output(QueryAnalysis)
exploration_chain = llm_utility.with_structured_output(ExplorationSuggestions)


# --- Graph Nodes ---
# Each node is a function that takes the state and returns a dictionary to update the state.

# --- Graph Nodes ---
# Redesigned for natural, flexible conversation flow

def contextualize_query(state: AgentState) -> dict:
    """
    If there is chat history, rephrase the user's query to be a standalone question.
    """
    print("--- NODE: contextualize_query ---")
    if not state.get("chat_history"):
        print("  - No chat history, using original query.")
        return {"contextualized_query": state["original_query"]}

    print("  - Rephrasing query based on history.")
    response = llm_reasoning.invoke([
        SystemMessage(content="You are an expert DFSA/ADGM compliance officer. Rephrase the latest user question as a standalone compliance query based on chat history."),
        HumanMessage(content=f"Chat History:\n{state['chat_history']}\n\nLatest Question: {state['original_query']}")
    ])
    
    contextualized = response.content
    print(f"  - Contextualized Query: {contextualized}")
    return {"contextualized_query": contextualized}


def analyze_query(state: AgentState) -> dict:
    """
    Analyze the query with confidence scoring and flexible routing.
    """
    print("--- NODE: analyze_query ---")
    
    # First, do a quick heuristic check for obviously non-compliance queries
    query_lower = state['contextualized_query'].lower()
    
    # Personal/romantic queries
    personal_indicators = ["darling", "love", "honey", "baby", "sweetheart", "about you", "who are you", "your name", "how are you feeling"]
    if any(indicator in query_lower for indicator in personal_indicators):
        print("  - Heuristic: Personal/romantic query detected - conversational only")
        return {
            "query_type": "GREETING",
            "query_confidence": 0.9,
            "should_retrieve": False,
            "should_converse": True
        }
    
    # Small talk/casual queries
    casual_indicators = ["weather", "time", "date", "joke", "story", "chat", "talk", "hello", "hi", "thanks", "thank you"]
    if any(indicator in query_lower for indicator in casual_indicators) and len(query_lower.split()) < 6:
        print("  - Heuristic: Casual query detected - conversational only")
        return {
            "query_type": "GREETING", 
            "query_confidence": 0.9,
            "should_retrieve": False,
            "should_converse": True
        }
    
    prompt = [
        SystemMessage(content="""You are an expert compliance assistant. Analyze the user's query and determine:

1. Query Type:
   - GREETING: Personal questions, small talk, thanks, hellos
   - COMPLIANCE: Questions about regulations, rules, requirements, procedures
   - EXPLORATORY: General questions about compliance topics that might benefit from documents
   - CLARIFICATION: Follow-up questions about previous compliance discussions

2. Confidence in understanding (0.0-1.0)

3. Should retrieve documents: ONLY if the query relates to specific regulatory content, rules, procedures, or compliance requirements. DO NOT retrieve for:
   - Personal questions about the assistant
   - General greetings or thanks  
   - Casual conversation
   - Questions that can be answered conversationally

4. Should provide conversational response: Almost always true unless purely technical document lookup

Be conservative with document retrieval - only use it for actual compliance/regulatory questions."""),
        HumanMessage(content=f"User Query: {state['contextualized_query']}")
    ]
    
    try:
        result = query_analysis_chain.invoke(prompt)
        print(f"  - Query Type: {result.query_type}")
        print(f"  - Confidence: {result.confidence}")
        print(f"  - Should Retrieve: {result.should_retrieve}")
        print(f"  - Should Converse: {result.should_converse}")
        
        return {
            "query_type": result.query_type,
            "query_confidence": result.confidence,
            "should_retrieve": result.should_retrieve,
            "should_converse": result.should_converse
        }
    except Exception as e:
        print(f"  - Error in analysis, using fallback: {e}")
        # More conservative fallback - default to conversation only unless clearly compliance-related
        compliance_keywords = ["regulation", "rule", "requirement", "compliance", "dfsa", "adgm", "difc", "license", "capital", "aml", "fund"]
        is_compliance = any(keyword in query_lower for keyword in compliance_keywords)
        
        return {
            "query_type": "COMPLIANCE" if is_compliance else "GREETING",
            "query_confidence": 0.5,
            "should_retrieve": is_compliance,
            "should_converse": True
        }


def generate_conversational_response(state: AgentState) -> dict:
    """
    Generate a natural, helpful conversational response.
    """
    print("--- NODE: generate_conversational_response ---")
    try:
        messages = state.get("chat_history", []) + [HumanMessage(content=state.get("original_query", ""))]
        
        system_prompt = """You are a friendly, knowledgeable compliance assistant for DIFC and ADGM regulations. 
Be conversational, helpful, and encouraging. If the user is exploring or learning, guide them naturally. 
If they're asking specific compliance questions, acknowledge that while being personable."""
        
        response = llm_utility.invoke([SystemMessage(content=system_prompt)] + messages)
        print(f"  - Generated conversational response")
        
        return {"conversational_response": response.content}
    except Exception as e:
        print(f"  - Error generating conversational response: {e}")
        return {"conversational_response": "I'm here to help with your compliance questions! What would you like to know?"}


def retrieve_documents(state: AgentState) -> dict:
    """
    Retrieve relevant documents using enhanced retrieval if available.
    """
    print("--- NODE: retrieve_documents ---")
    
    if not state.get("should_retrieve", False):
        print("  - Skipping retrieval based on analysis")
        return {"retrieved_docs": []}
    
    try:
        # Try enhanced retrieval first
        from backend.core.ai_service import get_enhanced_documents
        
        jurisdiction = state.get("jurisdiction")
        query = state.get("contextualized_query", "")
        
        print(f"  - Enhanced retrieval for: '{query}' in {jurisdiction}")
        
        # Use enhanced retrieval
        enhanced_docs = get_enhanced_documents(query, jurisdiction)
        
        if enhanced_docs:
            # Convert to Document format for compatibility
            retrieved_docs = []
            for doc_data in enhanced_docs:
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                score = doc_data.get('score', 0.0)
                
                if content and jurisdiction.upper() in metadata.get("jurisdiction", "").upper():
                    from langchain.schema import Document
                    retrieved_docs.append(Document(page_content=content, metadata=metadata))
            
            print(f"  - Enhanced retrieval: {len(retrieved_docs)} relevant documents")
            return {"retrieved_docs": retrieved_docs}
        
    except Exception as e:
        print(f"  - Enhanced retrieval failed: {e}")
        print("  - Falling back to basic retrieval")
    
    # Fallback to original retrieval method
    try:
        from backend.core.ai_service import index
        if index is None:
            print("  - Index unavailable, skipping retrieval")
            return {"retrieved_docs": []}

        jurisdiction = state.get("jurisdiction")
        query = state.get("contextualized_query", "")
        
        print(f"  - Basic retrieval for: '{query}' in {jurisdiction}")
        
        # Use query engine for basic retrieval
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            mode="hybrid"
        )
        
        response = query_engine.query(query)
        retrieved_docs = []
        
        for node in getattr(response, "source_nodes", []):
            content = getattr(node.node, "get_content", lambda: None)()
            metadata = getattr(node.node, "metadata", {})
            if content and jurisdiction.upper() in metadata.get("jurisdiction", "").upper():
                from langchain.schema import Document
                retrieved_docs.append(Document(page_content=content, metadata=metadata))
        
        print(f"  - Basic retrieval: {len(retrieved_docs)} relevant documents")
        return {"retrieved_docs": retrieved_docs}
        
    except Exception as e:
        print(f"  - Error retrieving documents: {e}")
        return {"retrieved_docs": []}


def synthesize_response(state: AgentState) -> dict:
    """
    Create a natural hybrid response combining conversation and compliance info.
    """
    print("--- NODE: synthesize_response ---")
    
    conversational = state.get("conversational_response", "")
    docs = state.get("retrieved_docs", [])
    query_type = state.get("query_type", "COMPLIANCE")
    should_retrieve = state.get("should_retrieve", False)
    
    # For greetings without conversational response, generate one now
    if query_type == "GREETING" and not conversational:
        print("  - Generating conversational response for greeting")
        try:
            messages = state.get("chat_history", []) + [HumanMessage(content=state.get("original_query", ""))]
            system_prompt = """You are a friendly, knowledgeable compliance assistant for DIFC and ADGM regulations. 
Respond naturally to greetings, thanks, and personal questions. Be warm but professional."""
            
            response = llm_utility.invoke([SystemMessage(content=system_prompt)] + messages)
            conversational = response.content
        except Exception as e:
            print(f"  - Error generating conversational response: {e}")
            conversational = "Hello! I'm here to help with your DIFC and ADGM compliance questions. What would you like to know?"
    
    # For pure greetings, just return conversational response
    if query_type == "GREETING" and not should_retrieve:
        return {
            "final_response": conversational,
            "final_response_with_sources": {"answer": conversational, "sources": []}
        }
    
    # If we have both conversation and docs, create a hybrid response
    if conversational and docs:
        doc_context = "\n---\n".join([
            f"Source: {d.metadata.get('file_name', 'Unknown')}\nContent: {d.page_content}" 
            for d in docs[:3]  # Limit to top 3 docs
        ])
        
        prompt = [
            SystemMessage(content="""You are a compliance expert. Create a natural response that:
1. Starts conversationally and acknowledges the user's question
2. Provides specific regulatory guidance from the documents
3. Uses inline citations [Source: filename] when referencing documents
4. Ends helpfully, offering to explore more if relevant

Be natural and flowing, not robotic."""),
            HumanMessage(content=f"User Question: {state['contextualized_query']}\n\nConversational Context: {conversational}\n\nRegulatory Documents:\n{doc_context}")
        ]
        
        try:
            response = llm_reasoning.invoke(prompt)
            final_answer = response.content
        except Exception as e:
            print(f"  - Error synthesizing, using fallback: {e}")
            final_answer = f"{conversational}\n\nI also found some relevant regulatory information that might help."
            
    elif docs:
        # Only compliance info available
        doc_context = "\n---\n".join([f"Source: {d.metadata.get('file_name', 'Unknown')}\nContent: {d.page_content}" for d in docs])
        prompt = [
            SystemMessage(content="Provide helpful compliance guidance based on these documents. Use inline citations [Source: filename]."),
            HumanMessage(content=f"Question: {state['contextualized_query']}\n\nDocuments:\n{doc_context}")
        ]
        try:
            response = llm_reasoning.invoke(prompt)
            final_answer = response.content
        except Exception as e:
            final_answer = "I found some relevant information but encountered an error processing it."
    else:
        # Only conversational available or fallback
        final_answer = conversational or "I'm here to help with compliance questions. What would you like to know?"
    
    print("  - Synthesized hybrid response")
    return {
        "final_response": final_answer,
        "final_response_with_sources": {"answer": final_answer, "sources": docs}
    }


# --- Conditional Edges ---
# Simplified decision making for natural flow

def decide_next_steps(state: AgentState) -> Literal["conversational_only", "retrieval_and_conversation", "synthesis"]:
    """
    Decide what to do based on query analysis.
    """
    print("--- CONDITIONAL: decide_next_steps ---")
    
    should_retrieve = state.get("should_retrieve", True)
    should_converse = state.get("should_converse", True)
    query_type = state.get("query_type", "COMPLIANCE")
    
    # For greetings and personal queries, only conversational response
    if query_type == "GREETING" or not should_retrieve:
        print("  - Decision: Conversational only (no document retrieval needed)")
        return "conversational_only"
    # For compliance queries that need both conversation and documents
    elif should_retrieve and should_converse:
        print("  - Decision: Hybrid approach - retrieve documents and converse")
        return "retrieval_and_conversation"  
    # For pure document lookup (rare)
    elif should_retrieve and not should_converse:
        print("  - Decision: Document retrieval only")
        return "synthesis"
    else:
        print("  - Decision: Fallback to conversational")
        return "conversational_only"


# --- Assemble Graph ---
# New simplified, flexible workflow
workflow = StateGraph(AgentState)

# Add nodes for new approach
workflow.add_node("contextualize_query", contextualize_query)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("generate_conversational_response", generate_conversational_response)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("synthesize_response", synthesize_response)

# Define the flow - much simpler and more natural
workflow.set_entry_point("contextualize_query")

workflow.add_edge("contextualize_query", "analyze_query")

# Route based on analysis
workflow.add_conditional_edges(
    "analyze_query",
    decide_next_steps,
    {
        "conversational_only": "synthesize_response",  # Skip retrieval, go straight to synthesis
        "retrieval_and_conversation": "generate_conversational_response", 
        "synthesis": "retrieve_documents"  # Pure document lookup
    },
)

# For hybrid approach, go: conversation -> retrieval -> synthesis
workflow.add_edge("generate_conversational_response", "retrieve_documents")
workflow.add_edge("retrieve_documents", "synthesize_response")

# All paths end at synthesis
workflow.add_edge("synthesize_response", END)


# Compile the graph
agent = workflow.compile()
print("Graph compiled successfully.")


# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Running Agent Example ---")
    
    # Mock inputs
    inputs = {
        "original_query": "What are the capital requirements for a Category 3 firm?",
        "jurisdiction": "DFSA",
        "chat_history": []
    }

    # The `stream` method is useful for observing the flow of state through the graph
    for event in agent.stream(inputs):
        for key, value in event.items():
            print(f"--- Event: {key} ---")
            print(value)
            print("\n")

    # Example of a final result
    final_state = agent.invoke(inputs)
    print("--- FINAL RESPONSE ---")
    print(final_state.get("final_response"))
