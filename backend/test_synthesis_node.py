#!/usr/bin/env python3
"""
Quick test to verify the synthesis node is working correctly
"""
import asyncio
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

async def test_synthesis_import():
    """Test that the synthesis node can be imported"""
    try:
        from backend.core.agent.nodes import generate_response
        print("✓ Successfully imported generate_response (synthesis node)")
        return True
    except Exception as e:
        print(f"✗ Failed to import synthesis node: {e}")
        return False

async def test_state_definition():
    """Test that the AgentState has all required fields"""
    try:
        from backend.core.agent.state import AgentState
        
        # Check if reflection_count field exists
        if 'reflection_count' in AgentState.__annotations__:
            print("✓ AgentState has reflection_count field")
        else:
            print("✗ AgentState missing reflection_count field")
            return False
            
        # Check other critical fields
        required_fields = [
            'user_query', 'jurisdiction', 'search_results', 
            'final_response', 'needs_additional_search', 'reflection_analysis'
        ]
        
        for field in required_fields:
            if field not in AgentState.__annotations__:
                print(f"✗ AgentState missing {field} field")
                return False
        
        print(f"✓ AgentState has all required fields")
        return True
        
    except Exception as e:
        print(f"✗ Failed to verify state definition: {e}")
        return False

async def test_synthesis_structure():
    """Test that the synthesis node has the correct structure"""
    try:
        from backend.core.agent.nodes import generate_response
        import inspect
        
        # Check if it's an async function
        if inspect.iscoroutinefunction(generate_response):
            print("✓ generate_response is an async function")
        else:
            print("✗ generate_response is not async")
            return False
            
        # Check signature
        sig = inspect.signature(generate_response)
        params = list(sig.parameters.keys())
        
        if 'state' in params:
            print("✓ generate_response accepts 'state' parameter")
        else:
            print("✗ generate_response missing 'state' parameter")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to verify synthesis structure: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Synthesis Node")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_synthesis_import),
        ("State Definition Test", test_state_definition),
        ("Node Structure Test", test_synthesis_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = await test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Synthesis node is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
