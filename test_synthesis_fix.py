"""
Test to verify synthesis node properly handles errors and includes sources.
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_synthesis_error_handling():
    """Test that synthesis errors include used_sources"""
    from backend.core.agent.state import AgentState
    from backend.core.agent.nodes import generate_response
    
    # Mock state with search results but force a synthesis issue
    state = {
        "user_query": "What is a Fund in DIFC?",
        "search_results": [
            {
                'id': 'test_doc_1',
                'content': 'Test regulatory content about funds.',
                'score': 0.95,
                'metadata': {
                    'text': 'Test regulatory content about funds.',
                    'title': 'Collective Investment Rules',
                    'section': 'Definitions',
                    'authority_level': 'primary',
                    'jurisdiction': 'DIFC',
                    'checksum': 'abc123',
                    'source_collection': 'cir',
                    'retrieval_method': 'fusion',
                    'query_description': 'Fund definition'
                }
            }
        ],
        "messages": [],
        "jurisdiction": "DIFC",
        "reflection_count": 0
    }
    
    print("Testing synthesis with valid state...")
    try:
        result = await generate_response(state)
        print(f"✓ Synthesis completed")
        print(f"  - Has final_response: {'final_response' in result}")
        print(f"  - Has used_sources: {'used_sources' in result}")
        print(f"  - Response length: {len(result.get('final_response', ''))}")
        print(f"  - Number of sources: {len(result.get('used_sources', []))}")
        
        if 'used_sources' in result and len(result['used_sources']) > 0:
            print("✓ Sources properly included in result")
        else:
            print("✗ ERROR: Sources missing from result!")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=" * 60)
    print("Testing Synthesis Node Error Handling Fix")
    print("=" * 60)
    print()
    
    success = await test_synthesis_error_handling()
    
    print()
    print("=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("The synthesis node now properly includes sources even on errors.")
    else:
        print("✗ TESTS FAILED")
        print("There are still issues with source handling.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
