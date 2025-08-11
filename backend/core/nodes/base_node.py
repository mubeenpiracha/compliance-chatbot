"""
Base node class for the compliance analysis graph.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..models.query_models import ProcessingState


class BaseNode(ABC):
    """Base class for all nodes in the compliance analysis graph."""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        
    @abstractmethod
    async def process(self, state: ProcessingState, **kwargs) -> Dict[str, Any]:
        """
        Process the current state and return results.
        
        Args:
            state: Current processing state
            **kwargs: Additional arguments specific to the node
            
        Returns:
            Dictionary containing the node's output
        """
        pass
    
    def update_state(self, state: ProcessingState, results: Dict[str, Any]) -> ProcessingState:
        """Update the processing state with node results."""
        state.completed_nodes.append(self.node_name)
        state.intermediate_results[self.node_name] = results
        return state
    
    async def execute(self, state: ProcessingState, **kwargs) -> ProcessingState:
        """Execute the node and update state."""
        results = await self.process(state, **kwargs)
        return self.update_state(state, results)


class ConditionalNode(BaseNode):
    """Base class for nodes that can route to different next nodes based on conditions."""
    
    @abstractmethod
    def get_next_node(self, state: ProcessingState) -> Optional[str]:
        """Determine the next node to execute based on current state."""
        pass
