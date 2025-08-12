# backend/core/agent/state.py
from typing import List, TypedDict, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from backend.core.models.agent_models import SearchPlan

class AgentState(TypedDict):
    """
    Represents the state of our agent. It's a dictionary that will be passed
    between nodes in the graph.
    """
    messages: List[BaseMessage]
    user_query: str
    jurisdiction: str
    search_plan: Optional[SearchPlan]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]
    clarification_needed: bool
