# backend/core/graph_agent.py
import os
import json
from typing import List, TypedDict, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from backend.core.config import OPENAI_API_KEY

# --- 0. Models & Parsers ---
class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate workflow."""
    route: Literal["COMPLEX", "SIMPLE", "CONVERSATIONAL"] = Field(
        description="The workflow to route the query to."
    )

class DecomposedQuestions(BaseModel):
    """Sub-questions decomposed from the user's main query."""
    questions: List[str] = Field(
        description="List of specific, answerable sub-questions."
    )

class ClarityGrade(BaseModel):
    """Clarity grade of the user's query."""
    grade: Literal["CLEAR", "AMBIGUOUS"] = Field(
        description="The clarity of the query."
    )

# Initialize deterministic LLM clients
llm_reasoning = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
llm_utility   = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# --- 1. Define Chains ---
route_chain = LLMChain(
    llm=llm_utility,
    prompt=PromptTemplate(
        template=(
            "You are an expert compliance officer in DFSA and ADGM. "
            "Classify the user's query into COMPLEX, SIMPLE, or CONVERSATIONAL. "
            "Output ONLY JSON matching the RouteQuery schema.\n"
            "Chat History: {chat_history}\n"
            "User Query: {query}"
        ),
        input_variables=["chat_history", "query"],
    ),
    output_key="route",
)
decompose_chain     = llm_reasoning.with_structured_output(DecomposedQuestions)
grade_clarity_chain = llm_utility.with_structured_output(ClarityGrade)

# --- 2. Define Agent State ---
class AgentState(TypedDict):
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

# --- 3. Nodes ---

def route_query(state: AgentState) -> AgentState:
    print("--- Executing Node: route_query ---")
    result = route_chain.invoke({
        "chat_history": state["chat_history"],
        "query": state["original_query"],
    })
    route = result.get("route")
    print(f"  - Route decided: {route}")
    return {**state, "route": route}


def contextualize_query(state: AgentState) -> AgentState:
    print("--- Executing Node: contextualize_query ---")
    if not state.get("chat_history"):
        return {**state, "contextualized_query": state["original_query"]}
    prompt = PromptTemplate(
        template=(
            "As an expert DFSA/ADGM compliance officer, rephrase the latest user question as a standalone compliance query.\n"
            "Chat History: {chat_history}\n"
            "Latest User Question: {query}"            
        ),
        input_variables=["chat_history", "query"],
    )
    chain = LLMChain(llm=llm_utility, prompt=prompt)
    result = chain.invoke({
        "chat_history": state["chat_history"],
        "query": state["original_query"],
    })
    content = result.get("text")
    print(f"  - Contextualized Query: {content}")
    return {**state, "contextualized_query": content}


def grade_query_clarity(state: AgentState) -> AgentState:
    print("--- Executing Node: grade_query_clarity ---")
    prompt = PromptTemplate(
        template=(
            "Assess clarity of the compliance query. Return only JSON with 'grade'.\n"
            "AMBGIUOUS if missing details; CLEAR if specific.\n"
            "Query: {query}"
        ),
        input_variables=["query"],
    )
    chain = LLMChain(llm=llm_utility, prompt=prompt, output_key="grade")
    raw = chain.invoke({"query": state["contextualized_query"]})
    result = grade_clarity_chain.invoke({"text": raw})
    grade = result.grade
    print(f"  - Clarity Grade: {grade}")
    return {**state, "clarity_grade": grade}


def request_clarification(state: AgentState) -> AgentState:
    print("--- Executing Node: request_clarification ---")
    prompt = PromptTemplate(
        template=(
            "The compliance query is ambiguous. Ask one precise follow-up question.\n"
            "Ambiguous Query: {query}"
        ),
        input_variables=["query"],
    )
    chain = LLMChain(llm=llm_utility, prompt=prompt)
    result = chain.invoke({"query": state["contextualized_query"]})
    question = result.get("text")
    print(f"  - Clarification Question: {question}")
    return {**state, "clarification_question": question}


def decompose_query(state: AgentState) -> AgentState:
    print("--- Executing Node: decompose_query ---")
    prompt = PromptTemplate(
        template=(
            "Break down the compliance goal into specific sub-questions.\n"
            "Goal: {query}"
        ),
        input_variables=["query"],
    )
    try:
        result = decompose_chain.invoke({"query": state["contextualized_query"]})
    except Exception:
        correction = PromptTemplate(
            template="Fix JSON to match DecomposedQuestions schema: {raw}",
            input_variables=["raw"],
        )
        corr_chain = LLMChain(llm=llm_utility, prompt=correction)
        # re-invoke
        result = decompose_chain.invoke({"query": state["contextualized_query"]})
    print(f"  - Sub-questions: {result.questions}")
    return {**state, "sub_questions": result.questions}


def retrieve_documents(state: AgentState) -> AgentState:
    print("--- Executing Node: retrieve_documents ---")
    from backend.core.ai_service import index
    if index is None:
        raise RuntimeError("Retrieval index is unavailable")
    filters = {"jurisdiction": state.get("jurisdiction")}
    qe = index.as_query_engine(similarity_top_k=5, mode="hybrid", metadata_filter=filters)
    docs: List[Document] = []
    for q in state.get("sub_questions", []):
        resp = qe.query(q)
        for node in getattr(resp, "source_nodes", []):
            content = getattr(node.node, "get_content", lambda: None)()
            if content:
                docs.append(Document(page_content=content))
    print(f"  - Retrieved {len(docs)} docs")
    return {**state, "retrieved_docs": docs}


def synthesize_response(state: AgentState) -> AgentState:
    print("--- Executing Node: synthesize_response ---")
    docs = state.get("retrieved_docs", [])
    prompt = PromptTemplate(
        template=(
            "Provide concise compliance guidance based solely on these docs.\n"
            "Docs: {documents}\n"
            "Query: {query}"
        ),
        input_variables=["documents", "query"],
    )
    chain = LLMChain(llm=llm_reasoning, prompt=prompt)
    doc_text = "\n---\n".join(d.page_content for d in docs)
    result = chain.invoke({"documents": doc_text, "query": state["contextualized_query"]})
    answer = result.get("text")
    print("  - Synthesized response.")
    return {**state, "final_response": answer}


def generate_conversational_response(state: AgentState) -> AgentState:
    print("--- Executing Node: generate_conversational_response ---")
    chain = llm_utility
    result = chain.invoke(state["chat_history"] + [state["original_query"]])
    text = result.get("text")
    return {**state, "final_response": text}

# --- 4. Assemble Graph ---
workflow = StateGraph(AgentState)
for name, fn in [
    ("route_query", route_query),
    ("contextualize_query", contextualize_query),
    ("grade_query_clarity", grade_query_clarity),
    ("request_clarification", request_clarification),
    ("decompose_query", decompose_query),
    ("retrieve_documents", retrieve_documents),
    ("synthesize_response", synthesize_response),
    ("generate_conversational_response", generate_conversational_response),
    ("main_flow_router", lambda s: s),
]:
    workflow.add_node(name, fn)

workflow.set_entry_point("route_query")
workflow.add_edge("route_query", "contextualize_query")
workflow.add_edge("contextualize_query", "grade_query_clarity")
workflow.add_conditional_edges(
    "grade_query_clarity",
    lambda s: "request_clarification" if s.get("clarity_grade")=="AMBIGUOUS" else "main_flow_router",
    {"request_clarification":"request_clarification","main_flow_router":"main_flow_router"},
)
workflow.add_edge("request_clarification", "grade_query_clarity")
workflow.add_conditional_edges(
    "main_flow_router",
    lambda s: "generate_conversational_response" if s.get("route")=="CONVERSATIONAL" else ("decompose_query" if s.get("route")=="COMPLEX" else "retrieve_documents"),
    {"decompose_query":"decompose_query","retrieve_documents":"retrieve_documents","generate_conversational_response":"generate_conversational_response"},
)
workflow.add_edge("decompose_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "synthesize_response")
workflow.add_edge("synthesize_response", END)
workflow.add_edge("generate_conversational_response", END)

agent = workflow.compile()
print("Graph compiled successfully.")
