# backend/core/agent_service.py
import uuid
from typing import List, Dict, Any
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.core.agent.builder import workflow

class AgentService:
    """
    A service that uses the compiled LangGraph agent to process requests.
    """
    async def get_ai_response(self, user_message: str, history: List[Dict[str, Any]], jurisdiction: str) -> Dict[str, Any]:
        """
        Invokes the LangGraph agent and returns the final response.
        """
        # A thread_id is used to uniquely identify a conversation session
        thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}

        # The input to the graph is the initial state
        initial_state = {
            "user_query": user_message,
            "messages": history,
            "jurisdiction": jurisdiction
        }

        # Use the checkpointer as a context manager
        async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
            # Compile the graph with the checkpointer
            graph = workflow.compile(checkpointer=memory)
            
            # Run the graph
            final_state = await graph.ainvoke(initial_state, config=config)

        # Extract the final response and sources
        response_text = final_state.get("final_response", "I'm sorry, I encountered an issue and cannot respond.")
        sources = final_state.get("search_results", [])
        
        # Format sources to match the expected schema
        formatted_sources = [
            {
                "filename": r['metadata'].get('filename'),
                "page_number": r['metadata'].get('page_number'),
                "chunk_number": r['metadata'].get('chunk_number'),
                "score": r.get('score', 0.0)
            } for r in sources
        ]

        return {
            'sender': 'ai',
            'text': response_text,
            'sources': formatted_sources,
        }

# Singleton instance
_agent_service_instance = None

async def get_agent_response(user_message: str, history: List[Dict[str, Any]], jurisdiction: str) -> Dict[str, Any]:
    """
    Entry point to get a response from the agent service.
    """
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = AgentService()
    
    return await _agent_service_instance.get_ai_response(user_message, history, jurisdiction)
