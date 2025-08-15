# backend/core/agent/state.py
from typing import List, TypedDict, Optional, Dict, Any, Union
from backend.core.models.agent_models import SearchPlan, ClarificationRequest

class AgentState(TypedDict):
    """
    Represents the state of our agent. It's a dictionary that will be passed
    between nodes in the graph.
    """
    # Use flexible message format - can be either BaseMessage objects or dicts
    messages: Optional[List[Dict[str, Any]]]
    user_query: str
    jurisdiction: str
    analysis_reasoning: Optional[str]
    decision: Optional[Union[SearchPlan, ClarificationRequest]]
    search_plan: Optional[SearchPlan]
    search_results: Optional[List[Dict[str, Any]]]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    used_sources: Optional[List[Dict[str, Any]]]  # Sources used in final response for citation tracking
    final_response: Optional[str]
    clarification_needed: Optional[bool]
