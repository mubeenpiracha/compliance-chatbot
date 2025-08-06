# backend/core/graph_agent.py
import os
from typing import List, TypedDict, Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import json

from backend.core.config import OPENAI_API_KEY

# --- 0. Models & Parsers ---
# Define the models we'll use, according to our tiered strategy.
# We also define output parsers to get structured data from the LLMs.

class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate workflow."""
    route: Literal["COMPLEX", "SIMPLE", "CONVERSATIONAL"] = Field(
        description="The workflow to route the query to. One of: COMPLEX, SIMPLE, or CONVERSATIONAL."
    )

class DecomposedQuestions(BaseModel):
    """A list of sub-questions decomposed from the user's main query."""
    questions: List[str] = Field(
        description="A list of specific, answerable sub-questions."
    )

class ClarityGrade(BaseModel):
    """The clarity grade of the user's query."""
    grade: Literal["CLEAR", "AMBIGUOUS"] = Field(
        description="The clarity of the query. One of: CLEAR or AMBIGUOUS."
    )

# Tiered model strategy - now using gpt-4o for all LLM calls for debugging
llm_reasoning = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
llm_utility = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY) # Changed to gpt-4o

# Structured output chains - gpt-4o supports direct structured output
route_chain = llm_utility.with_structured_output(RouteQuery)
decompose_chain = llm_reasoning.with_structured_output(DecomposedQuestions)
grade_clarity_chain = llm_utility.with_structured_output(ClarityGrade)


# --- 1. Define the State for our graph ---
class AgentState(TypedDict):
    """
    Represents the state of our agent. It's passed between nodes.
    """
    original_query: str
    jurisdiction: str
    chat_history: List[BaseMessage]
    contextualized_query: str
    clarity_grade: str
    sub_questions: List[str]
    retrieved_docs: List[Document]
    final_response: str
    route: str
    clarification_question: str

# --- 2. Define the Nodes of the Graph ---

def route_query(state: AgentState) -> AgentState:
    """
    Node: Decides the best path (COMPLEX, SIMPLE, CONVERSATIONAL) using gpt-4o.
    """
    print("--- Executing Node: route_query ---")
    prompt = PromptTemplate.from_template(
        """You are an expert query router. Based on the user's query and chat history,
        classify the query into one of the following categories:
        - COMPLEX: For broad, open-ended questions that require breaking down into multiple steps.
        - SIMPLE: For specific, factual questions that can likely be answered with a single search.
        - CONVERSATIONAL: For greetings, thank yous, or other non-question messages.

        Chat History:
        {chat_history}

        User Query: {query}
        """
    )
    chain = prompt | route_chain
    result = chain.invoke({
        "chat_history": state["chat_history"],
        "query": state["original_query"]
    })
    print(f"  - Route decided: {result.route}")
    return {**state, "route": result.route}

def contextualize_query(state: AgentState) -> AgentState:
    """
    Node: Rephrases the query to be self-contained using gpt-4o.
    """
    print("--- Executing Node: contextualize_query ---")
    if not state.get("chat_history"):
        # If no history, the original query is already contextualized
        return {**state, "contextualized_query": state["original_query"]}

    prompt = PromptTemplate.from_template(
        """Given the chat history and the latest user question,
        rephrase the user question to be a standalone, self-contained question.

        Chat History:
        {chat_history}

        Latest User Question: {query}

        Standalone Question:
        """
    )
    chain = prompt | llm_utility
    result = chain.invoke({
        "chat_history": state["chat_history"],
        "query": state["original_query"]
    })
    print(f"  - Contextualized Query: {result.content}")
    return {**state, "contextualized_query": result.content}

def grade_query_clarity(state: AgentState) -> AgentState:
    """
    Node: Grades the query's clarity (CLEAR or AMBIGUOUS) using gpt-4o.
    """
    print("--- Executing Node: grade_query_clarity ---")
    prompt = PromptTemplate.from_template(
        """You are a clarity grader. Assess the following user query. Return a JSON object
        with a single key 'grade' and its value.
        A query is AMBIGUOUS if it lacks specific details needed to find a precise answer
        (e.g., uses vague terms, refers to "it" or "that" without context, is overly broad).
        A query is CLEAR if it is specific and actionable.

        User Query: {query}

        JSON Output:
        """
    )
    chain = prompt | grade_clarity_chain
    result = chain.invoke({"query": state["contextualized_query"]})
    print(f"  - Clarity Grade: {result.grade}")
    return {**state, "clarity_grade": result.grade}

def request_clarification(state: AgentState) -> AgentState:
    """
    Node (Fallback): Asks the user for more information using gpt-4o.
    """
    print("--- Executing Node: request_clarification ---")
    prompt = PromptTemplate.from_template(
        """The user's query is ambiguous. Ask a single, direct question to clarify their intent.
        Do not apologize.

        Ambiguous Query: {query}

        Clarification Question:
        """
    )
    chain = prompt | llm_utility
    result = chain.invoke({"query": state["contextualized_query"]})
    print(f"  - Generated Clarification: {result.content}")
    return {**state, "clarification_question": result.content}


def decompose_query(state: AgentState) -> AgentState:
    """
    Node: Breaks down a complex query into sub-questions using gpt-4o.
    """
    print("--- Executing Node: decompose_query ---")
    prompt = PromptTemplate.from_template(
        """You are an expert financial regulatory analyst. A user wants to understand a complex topic.
        Break down their goal into a series of specific, answerable sub-questions that can be
        researched in a regulatory rulebook.

        User's Goal: {query}
        """
    )
    chain = prompt | decompose_chain
    result = chain.invoke({"query": state["contextualized_query"]})
    print(f"  - Decomposed Questions: {result.questions}")
    return {**state, "sub_questions": result.questions}

def retrieve_documents(state: AgentState) -> AgentState:
    """
    Node: Fetches documents from Pinecone using Hybrid Search via LlamaIndex.
    """
    print("--- Executing Node: retrieve_documents ---")
    from backend.core.ai_service import index
    sub_questions = state.get("sub_questions", [])
    retrieved_docs = []
    if index is None:
        print("  - ERROR: Index is not initialized!")
        return {**state, "retrieved_docs": []}
    query_engine = index.as_query_engine(similarity_top_k=5, mode="hybrid")
    # TODO: Add metadata filtering (e.g., jurisdiction) in future
    for q in sub_questions:
        result = query_engine.query(q)
        # result is a Response object; extract source nodes
        for node in getattr(result, "source_nodes", []):
            # node.node is a BaseNode, node.score is similarity
            doc_content = getattr(node.node, "get_content", lambda: None)()
            if doc_content:
                retrieved_docs.append(Document(page_content=doc_content))
    print(f"  - Retrieved {len(retrieved_docs)} documents.")
    return {**state, "retrieved_docs": retrieved_docs}

def synthesize_response(state: AgentState) -> AgentState:
    """
    Node: Generates the final answer from retrieved documents using gpt-4o.
    """
    print("--- Executing Node: synthesize_response ---")
    docs = state["retrieved_docs"]
    print(f"  - Synthesizing from {len(docs)} documents.")
    for i, doc in enumerate(docs[:10]):
        print(f"    [Doc {i+1}] {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
    if len(docs) > 10:
        print(f"    ...and {len(docs)-10} more documents.")
    prompt = PromptTemplate.from_template(
        """You are an AI Regulatory Assistant. Your task is to provide a clear, factual answer
        to the user's question based *only* on the provided context documents.
        Synthesize the information from the documents into a single, coherent response.
        If the documents do not contain an answer, state that clearly.

        Context Documents:
        {documents}

        User's Question: {query}

        Final Answer:
        """
    )
    chain = prompt | llm_reasoning
    doc_content = "\n---\n".join([d.page_content for d in docs])
    result = chain.invoke({
        "documents": doc_content,
        "query": state["contextualized_query"]
    })
    print("  - Synthesized response.")
    return {**state, "final_response": result.content}

def generate_conversational_response(state: AgentState) -> AgentState:
    """
    Node: Generates a simple conversational response using gpt-4o.
    """
    print("--- Executing Node: generate_conversational_response ---")
    # This node is for simple chat messages and doesn't require complex prompting
    chain = llm_utility
    result = chain.invoke(state["chat_history"] + [state["original_query"]])
    return {**state, "final_response": result.content}


# --- 3. Define the Edges of the Graph ---

def _decide_clarity_path_router(state: AgentState) -> Literal["request_clarification", "continue_to_main_flow_decision"]:
    """
    Conditional Edge: Decides whether to ask for clarification or proceed.
    """
    print("--- Making Decision: Clarity Path ---")
    if state.get("clarity_grade") == "AMBIGUOUS":
        print("  - Decision: AMBIGUOUS, requesting clarification.")
        return "request_clarification"
    else:
        print("  - Decision: CLEAR, continuing to main flow.")
        return "continue_to_main_flow_decision"

def _decide_main_flow_path_router(state: AgentState) -> Literal["decompose_query", "retrieve_documents", "generate_conversational_response"]:
    """
    Conditional Edge: Based on the route, decides the next step.
    """
    print("--- Making Decision: Main Flow Path ---")
    route = state.get("route")
    if route == "CONVERSATIONAL":
        print("  - Decision: CONVERSATIONAL, generating chat response.")
        return "generate_conversational_response"
    elif route == "COMPLEX":
        print("  - Decision: COMPLEX, decomposing query.")
        return "decompose_query"
    else: # SIMPLE
        print("  - Decision: SIMPLE, retrieving documents directly.")
        return "retrieve_documents"

# --- 4. Assemble the Graph ---

workflow = StateGraph(AgentState)

# Add all the nodes
workflow.add_node("route_query", route_query)
workflow.add_node("contextualize_query", contextualize_query)
workflow.add_node("grade_query_clarity", grade_query_clarity)
workflow.add_node("request_clarification", request_clarification)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("synthesize_response", synthesize_response)
workflow.add_node("generate_conversational_response", generate_conversational_response)

# Add a dummy node for branching the main flow
workflow.add_node("main_flow_router", lambda state: state)

# Define the graph's flow
workflow.set_entry_point("route_query")

workflow.add_edge("route_query", "contextualize_query")
workflow.add_edge("contextualize_query", "grade_query_clarity")

# After grading, decide on clarity
workflow.add_conditional_edges(
    "grade_query_clarity",
    _decide_clarity_path_router,
    {
        "request_clarification": "request_clarification",
        "continue_to_main_flow_decision": "main_flow_router" # Map to the dummy router node
    }
)

# From the main router, decide the path based on the initial route
workflow.add_conditional_edges(
    "main_flow_router",
    _decide_main_flow_path_router,
    {
        "decompose_query": "decompose_query",
        "retrieve_documents": "retrieve_documents",
        "generate_conversational_response": "generate_conversational_response"
    }
)

# Define the rest of the flow
workflow.add_edge("decompose_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "synthesize_response")

# Define end points
workflow.add_edge("synthesize_response", END)
workflow.add_edge("request_clarification", END)
workflow.add_edge("generate_conversational_response", END)


# Compile the graph
agent = workflow.compile()
print("Graph compiled successfully with corrected flow.")