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
# We've added clarification_attempts to prevent infinite loops.
class AgentState(TypedDict):
    """
    Represents the state of our agent. This state is passed between nodes in the graph.
    """
    original_query: str
    jurisdiction: str
    chat_history: List[BaseMessage]
    contextualized_query: str
    clarity_grade: Literal["CLEAR", "AMBIGUOUS"]
    sub_questions: List[str]
    retrieved_docs: List[Document]
    final_response: str
    route: Literal["COMPLEX", "SIMPLE", "CONVERSATIONAL"]
    clarification_question: str
    clarification_attempts: int # New field to prevent infinite loops

# --- Pydantic Models for Structured LLM Output ---
# Using Pydantic models ensures the LLM output is structured and validated.

class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate workflow."""
    route: Literal["COMPLEX", "SIMPLE", "CONVERSATIONAL"] = Field(
        description="The workflow to route the query to based on its nature."
    )

class DecomposedQuestions(BaseModel):
    """A list of sub-questions decomposed from the user's main query."""
    questions: List[str] = Field(
        description="A list of specific, answerable sub-questions that break down the main query."
    )

class ClarityGrade(BaseModel):
    """The clarity grade of the user's query."""
    grade: Literal["CLEAR", "AMBIGUOUS"] = Field(
        description="The clarity of the query. AMBIGUOUS if it lacks key details, CLEAR if it is specific."
    )

class Clarification(BaseModel):
    """A question to ask the user for clarification."""
    question: str = Field(
        description="A precise follow-up question to the user to clarify their ambiguous query."
    )


# --- LLM and Tool Initialization ---
# We use two LLMs: one for complex reasoning and one for simpler utility tasks.
# Using .with_structured_output makes the LLM reliably return a Pydantic model.
llm_reasoning = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
llm_utility = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY) # Cheaper/faster for utility

# Structured LLM callers
route_chain = llm_utility.with_structured_output(RouteQuery)
decompose_chain = llm_reasoning.with_structured_output(DecomposedQuestions)
grade_clarity_chain = llm_utility.with_structured_output(ClarityGrade)
clarification_chain = llm_utility.with_structured_output(Clarification)


# --- Graph Nodes ---
# Each node is a function that takes the state and returns a dictionary to update the state.

def contextualize_query(state: AgentState) -> dict:
    """
    If there is chat history, rephrase the user's query to be a standalone question.
    """
    print("--- NODE: contextualize_query ---")
    if not state.get("chat_history"):
        print("  - No chat history, using original query.")
        return {"contextualized_query": state["original_query"], "clarification_attempts": 0}

    print("  - Rephrasing query based on history.")
    response = llm_reasoning.invoke([
        SystemMessage(content="You are an expert DFSA/ADGM compliance officer. Your task is to rephrase the latest user question into a standalone, self-contained compliance query based on the preceding chat history."),
        HumanMessage(content=f"Chat History:\n{state['chat_history']}\n\nLatest User Question: {state['original_query']}")
    ])
    
    contextualized = response.content
    print(f"  - Contextualized Query: {contextualized}")
    return {"contextualized_query": contextualized, "clarification_attempts": 0}


def route_query(state: AgentState) -> dict:
    """
    Routes the query to the appropriate workflow: COMPLEX, SIMPLE, or CONVERSATIONAL.
    """
    print("--- NODE: route_query ---")
    prompt = [
        SystemMessage(content="You are an expert compliance officer in DFSA and ADGM. Your task is to classify the user's query into one of three categories: COMPLEX (requires breaking down and deep research), SIMPLE (can be answered directly with a quick search), or CONVERSATIONAL (a greeting or off-topic chat)."),
        HumanMessage(content=f"Chat History: {state['chat_history']}\n\nUser Query: {state['original_query']}")
    ]
    result = route_chain.invoke(prompt)
    print(f"  - Route decided: {result.route}")
    return {"route": result.route}


def grade_query_clarity(state: AgentState) -> dict:
    """
    Assesses whether the query is CLEAR or AMBIGUOUS.
    """
    print("--- NODE: grade_query_clarity ---")
    prompt = [
        SystemMessage(content="You are an expert compliance assistant. Your task is to assess the clarity of the user's compliance query. A query is AMBIGUOUS if it is vague or missing critical details (like entity type, specific activity, or jurisdiction). A query is CLEAR if it is specific and directly answerable."),
        HumanMessage(content=f"Assess the following query: {state['contextualized_query']}")
    ]
    result = grade_clarity_chain.invoke(prompt)
    print(f"  - Clarity Grade: {result.grade}")
    return {"clarity_grade": result.grade}


def request_clarification(state: AgentState) -> dict:
    """
    If the query is ambiguous, generate a question to ask the user for clarification.
    """
    print("--- NODE: request_clarification ---")
    prompt = [
        SystemMessage(content="You are an expert compliance assistant. The user has provided an ambiguous query. Your task is to formulate a single, precise follow-up question to get the information needed to provide a complete answer."),
        HumanMessage(content=f"The ambiguous query is: {state['contextualized_query']}")
    ]
    result = clarification_chain.invoke(prompt)
    print(f"  - Clarification Question: {result.question}")
    # We also increment the attempt counter here.
    return {
        "clarification_question": result.question,
        "clarification_attempts": state.get("clarification_attempts", 0) + 1
    }


def decompose_query(state: AgentState) -> dict:
    """
    Breaks down a complex query into smaller, manageable sub-questions.
    """
    print("--- NODE: decompose_query ---")
    prompt = [
        SystemMessage(content="You are an expert compliance analyst. Your task is to break down the user's complex compliance goal into a series of specific, answerable sub-questions that can be used for targeted document retrieval."),
        HumanMessage(content=f"Break down this goal: {state['contextualized_query']}")
    ]
    result = decompose_chain.invoke(prompt)
    print(f"  - Sub-questions: {result.questions}")
    return {"sub_questions": result.questions}


def retrieve_documents(state: AgentState) -> dict:
    """
    Retrieves documents based on the sub-questions (for complex queries) or the main query (for simple queries).
    
    NOTE: This is a placeholder for your actual retrieval logic using the 'index'.
    """
    print("--- NODE: retrieve_documents ---")
    # This is a mock implementation. Replace with your actual retrieval logic.
    # from backend.core.ai_service import index
    # if index is None:
    #     raise RuntimeError("Retrieval index is unavailable")
    
    questions = state.get("sub_questions") or [state.get("contextualized_query")]
    print(f"  - Retrieving docs for {len(questions)} question(s).")
    
    # Mock documents
    docs = [
        Document(page_content=f"This is a mock document about '{q}'. In a real scenario, this would be content from your knowledge base.", metadata={"source": "mock_db"})
        for q in questions
    ]
    
    print(f"  - Retrieved {len(docs)} docs.")
    return {"retrieved_docs": docs}


def synthesize_response(state: AgentState) -> dict:
    """
    Synthesizes a final answer to the user based on the retrieved documents.
    """
    print("--- NODE: synthesize_response ---")
    doc_text = "\n---\n".join(d.page_content for d in state.get("retrieved_docs", []))
    prompt = [
        SystemMessage(content="You are an expert DFSA/ADGM compliance officer. Your task is to provide a clear, concise, and actionable compliance guidance based *solely* on the provided documents. Do not use any prior knowledge. Cite the source of the information if available."),
        HumanMessage(content=f"Based on these documents:\n\n{doc_text}\n\nAnswer this query: {state['contextualized_query']}")
    ]
    response = llm_reasoning.invoke(prompt)
    print("  - Synthesized response.")
    return {"final_response": response.content}


def generate_conversational_response(state: AgentState) -> dict:
    """
    Generates a simple conversational response for non-compliance-related queries.
    """
    print("--- NODE: generate_conversational_response ---")
    response = llm_utility.invoke(state["chat_history"] + [HumanMessage(content=state["original_query"])])
    return {"final_response": response.content}


# --- Conditional Edges ---
# These functions determine the next step in the graph based on the current state.

def decide_clarity_path(state: AgentState) -> Literal["request_clarification", "continue_to_route"]:
    """
    If the query is ambiguous and we haven't tried clarifying too many times, ask for clarification.
    Otherwise, continue the main workflow.
    """
    print("--- CONDITIONAL: decide_clarity_path ---")
    if state.get("clarity_grade") == "AMBIGUOUS" and state.get("clarification_attempts", 0) < 2:
        print("  - Decision: Query is AMBIGUOUS. Requesting clarification.")
        return "request_clarification"
    print("  - Decision: Query is CLEAR or max retries reached. Continuing.")
    return "continue_to_route"


def decide_main_workflow(state: AgentState) -> Literal["decompose_query", "retrieve_documents", "generate_conversational_response"]:
    """
    Directs the flow based on the 'route' determined earlier.
    """
    print("--- CONDITIONAL: decide_main_workflow ---")
    route = state.get("route")
    print(f"  - Decision: Routing to '{route}' workflow.")
    if route == "CONVERSATIONAL":
        return "generate_conversational_response"
    if route == "COMPLEX":
        return "decompose_query"
    # If SIMPLE
    return "retrieve_documents"

# --- Assemble Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("contextualize_query", contextualize_query)
workflow.add_node("grade_query_clarity", grade_query_clarity)
workflow.add_node("request_clarification", request_clarification)
workflow.add_node("route_query", route_query)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("synthesize_response", synthesize_response)
workflow.add_node("generate_conversational_response", generate_conversational_response)

# Define the flow
workflow.set_entry_point("contextualize_query")

workflow.add_edge("contextualize_query", "grade_query_clarity")

# Add the first conditional branch for clarity
workflow.add_conditional_edges(
    "grade_query_clarity",
    decide_clarity_path,
    {
        "request_clarification": "request_clarification",
        "continue_to_route": "route_query",
    },
)
# This edge creates the clarification loop. The conditional logic prevents it from being infinite.
workflow.add_edge("request_clarification", END) # In a real app, you'd wait for user input here. For now, it ends.

workflow.add_conditional_edges(
    "route_query",
    decide_main_workflow,
    {
        "decompose_query": "decompose_query",
        "retrieve_documents": "retrieve_documents",
        "generate_conversational_response": "generate_conversational_response",
    },
)

workflow.add_edge("decompose_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "synthesize_response")

# All final nodes lead to the end
workflow.add_edge("synthesize_response", END)
workflow.add_edge("generate_conversational_response", END)


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
