# backend/core/agent/builder.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.core.agent.state import AgentState
from backend.core.agent.nodes import analyze_query, execute_search, generate_response, format_clarification
from backend.core.models.agent_models import SearchPlan, ClarificationRequest

def should_search(state: AgentState):
    """
    Determines whether to proceed with a search or ask for clarification.
    """
    if isinstance(state["decision"], SearchPlan):
        return "search"
    elif isinstance(state["decision"], ClarificationRequest):
        return "clarify"
    return "end"

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("execute_search", execute_search)
workflow.add_node("generate_response", generate_response)
workflow.add_node("format_clarification", format_clarification)

# Set the entry point
workflow.set_entry_point("analyze_query")

# Add edges
workflow.add_conditional_edges(
    "analyze_query",
    should_search,
    {
        "search": "execute_search",
        "clarify": "format_clarification",
        "end": END
    }
)
workflow.add_edge("execute_search", "generate_response")
workflow.add_edge("generate_response", END)
workflow.add_edge("format_clarification", END)

# Set up memory
memory = SqliteSaver.from_conn_string(":memory:")

# Compile the graph
graph = workflow.compile(checkpointer=memory)
