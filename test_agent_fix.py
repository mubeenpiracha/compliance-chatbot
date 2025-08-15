"""
Test script to verify the agent fix works.
"""
import asyncio
import os
from backend.core.agent_service import AgentService

async def test_agent_fix():
    """Test that the agent can now execute searches without the VectorSearch init error."""
    
    print("Testing Agent Fix")
    print("=" * 30)
    
    try:
        # Initialize the agent service
        agent_service = AgentService()
        
        # Test with a simple query
        test_message = "What is a fund?"
        history = []
        jurisdiction = "adgm"
        
        print(f"Testing query: '{test_message}'")
        print("Processing...")
        
        # This should not throw the VectorSearchEngine.__init__() error anymore
        result = await agent_service.get_ai_response(test_message, history, jurisdiction)
        
        print("SUCCESS: Agent executed without VectorSearch init error!")
        print(f"Response: {result.get('response', 'No response field')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_agent_fix())
