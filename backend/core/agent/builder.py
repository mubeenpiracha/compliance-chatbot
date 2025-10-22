# backend/core/agent/builder.py
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from backend.core.agent.state import AgentState
from backend.core.agent.nodes import analyze_query, execute_search, generate_response, format_clarification, reflection_node, reflection_decision_node
from backend.core.models.agent_models import SearchPlan, ClarificationRequest
from langgraph.graph import StateGraph, END

def should_search(state: AgentState):
    """
    Determines whether to proceed with a search or ask for clarification.
    """
    if isinstance(state["decision"], SearchPlan):
        return "search"
    elif isinstance(state["decision"], ClarificationRequest):
        return "clarify"
    return "end"

def should_reflect(state: AgentState):
    """
    Determines whether to reflect on the response for incomplete information.
    Only allows one reflection per conversation to prevent infinite loops.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check if reflection has already been performed
    reflection_count = state.get("reflection_count", 0)
    needs_reflection = state.get("needs_additional_search", False)
    
    logger.info(f"🔀 should_reflect router - reflection_count: {reflection_count}, needs_additional_search: {needs_reflection}")
    
    if reflection_count > 0:
        logger.info(f"🔀 Routing to END - reflection already performed")
        return "end"  # Limit to one reflection
    
    # Check if reflection is needed based on incomplete information flags
    if needs_reflection:
        logger.info(f"🔀 Routing to REFLECT - incomplete information detected")
        return "reflect"
    
    logger.info(f"🔀 Routing to END - no reflection needed")
    return "end"

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("execute_search", execute_search)
workflow.add_node("generate_response", generate_response)
workflow.add_node("format_clarification", format_clarification)
workflow.add_node("reflection_decision_node", reflection_decision_node)
workflow.add_node("reflection_node", reflection_node)

# Set the entry point
workflow.set_entry_point("analyze_query")

# Add edges
workflow.add_conditional_edges(
    "analyze_query",
    should_search,
    {
        "search": "execute_search",
        "clarify": "format_clarification",
        "end": END,
    },
)
workflow.add_edge("execute_search", "generate_response")
workflow.add_edge("generate_response", "reflection_decision_node")  # Always check if reflection needed
workflow.add_conditional_edges(
    "reflection_decision_node",
    should_reflect,
    {
        "reflect": "reflection_node",
        "end": END,
    },
)
workflow.add_edge("reflection_node", "execute_search")  # Reflection triggers new search
workflow.add_edge("format_clarification", END)

# The graph is no longer compiled here, just the workflow is defined.
# The compilation will be handled in the agent_service.


