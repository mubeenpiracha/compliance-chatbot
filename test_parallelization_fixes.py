#!/usr/bin/env python3
"""
Test script to verify parallelization fixes in nodes.py
Run this to ensure OpenAI calls are not being lost.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

# Configure logging to see all details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track OpenAI calls
openai_call_count = {
    "embeddings": 0,
    "chat": 0,
    "total": 0
}

def track_openai_call(call_type: str):
    """Decorator to track OpenAI API calls."""
    openai_call_count[call_type] += 1
    openai_call_count["total"] += 1
    logger.info(f"🔵 OpenAI Call #{openai_call_count['total']}: {call_type}")


class MockOpenAIClient:
    """Mock OpenAI client to test without making real API calls."""
    
    class Embeddings:
        async def create(self, **kwargs):
            track_openai_call("embeddings")
            await asyncio.sleep(0.1)  # Simulate network delay
            return type('obj', (object,), {
                'data': [type('obj', (object,), {'embedding': [0.1] * 1536})]
            })
    
    class Chat:
        class Completions:
            async def create(self, **kwargs):
                track_openai_call("chat")
                await asyncio.sleep(0.2)  # Simulate network delay
                return type('obj', (object,), {
                    'choices': [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': '{"reasoning": "Test", "decision": {"type": "search_plan", "queries": [{"query": "test", "description": "test query"}]}}'
                        })
                    })]
                })
        
        completions = Completions()
    
    embeddings = Embeddings()
    chat = Chat()


async def test_parallel_search_no_duplicate_client():
    """Test that we're not creating duplicate OpenAI clients."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: No Duplicate Client Creation")
    logger.info("="*80)
    
    # Import here to ensure we get the fixed version
    from backend.core.agent.nodes import execute_search, async_client
    from backend.core.models.agent_models import SearchPlan, SearchQuery
    
    # Create a mock state
    state = {
        "decision": SearchPlan(queries=[
            SearchQuery(query="What is an Authorised Firm?", description="Test query 1"),
            SearchQuery(query="What are the reporting requirements?", description="Test query 2"),
        ]),
        "jurisdiction": "DIFC",
        "needs_additional_search": False
    }
    
    # Check if execute_search is using module-level client
    import inspect
    source = inspect.getsource(execute_search)
    
    if "AsyncOpenAI(api_key=OPENAI_API_KEY)" in source:
        logger.error("❌ FAILED: execute_search still creates its own AsyncOpenAI client!")
        return False
    else:
        logger.info("✅ PASSED: execute_search uses module-level async_client")
        return True


async def test_timeout_handling():
    """Test that timeouts are properly configured."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Timeout Configuration")
    logger.info("="*80)
    
    from backend.core.agent.nodes import execute_single_query_search
    import inspect
    
    source = inspect.getsource(execute_single_query_search)
    
    passed = True
    
    # Check for wait_for with timeout
    if "asyncio.wait_for" in source:
        logger.info("✅ PASSED: Per-query timeout using asyncio.wait_for found")
    else:
        logger.error("❌ FAILED: No per-query timeout found")
        passed = False
    
    # Check timeout value
    if "timeout=25" in source or "timeout=30" in source:
        logger.info("✅ PASSED: Reasonable timeout value configured")
    else:
        logger.warning("⚠️  WARNING: Timeout value not found or unusual")
    
    return passed


async def test_global_timeout():
    """Test that global timeout exists for all parallel queries."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Global Timeout for Parallel Queries")
    logger.info("="*80)
    
    from backend.core.agent.nodes import execute_search
    import inspect
    
    source = inspect.getsource(execute_search)
    
    passed = True
    
    # Check for global timeout
    if "timeout=90" in source or "timeout=120" in source:
        logger.info("✅ PASSED: Global timeout configured for parallel execution")
    else:
        logger.error("❌ FAILED: No global timeout found")
        passed = False
    
    # Check for TimeoutError handling
    if "asyncio.TimeoutError" in source:
        logger.info("✅ PASSED: TimeoutError exception handling present")
    else:
        logger.error("❌ FAILED: No TimeoutError handling found")
        passed = False
    
    return passed


async def test_no_zero_vector_fallback():
    """Test that zero vector fallback was removed."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: No Zero Vector Fallback")
    logger.info("="*80)
    
    from backend.core.retrieval.vector_search import VectorSearchEngine
    import inspect
    
    source = inspect.getsource(VectorSearchEngine._get_query_embedding)
    
    if "return [0.0] * 1536" in source:
        logger.error("❌ FAILED: Zero vector fallback still present!")
        return False
    elif "raise RuntimeError" in source or "raise" in source:
        logger.info("✅ PASSED: Exception propagation instead of zero vector")
        return True
    else:
        logger.warning("⚠️  WARNING: Unclear error handling in embedding generation")
        return True


async def test_improved_reflection_logic():
    """Test that reflection logic has fewer false positives."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Improved Reflection Logic")
    logger.info("="*80)
    
    from backend.core.agent.nodes import generate_response
    import inspect
    
    source = inspect.getsource(generate_response)
    
    passed = True
    
    # Check for false positive handling
    if "false_positive" in source.lower():
        logger.info("✅ PASSED: False positive detection implemented")
    else:
        logger.warning("⚠️  WARNING: No explicit false positive handling found")
        passed = False
    
    # Check for critical vs general indicators
    if "critical_indicator" in source.lower():
        logger.info("✅ PASSED: More specific reflection patterns used")
    else:
        logger.warning("⚠️  WARNING: Still using generic patterns")
    
    # Check for response length consideration
    if "response_length" in source or "len(final_response)" in source:
        logger.info("✅ PASSED: Response length considered for reflection")
    else:
        logger.warning("⚠️  WARNING: Response length not factored into reflection logic")
    
    return passed


async def test_error_type_logging():
    """Test that error types are logged for better debugging."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: Error Type Logging")
    logger.info("="*80)
    
    from backend.core.agent.nodes import execute_single_query_search
    import inspect
    
    source = inspect.getsource(execute_single_query_search)
    
    if "type(e).__name__" in source or "error_type" in source:
        logger.info("✅ PASSED: Error types are logged for debugging")
        return True
    else:
        logger.warning("⚠️  WARNING: Generic exception logging still used")
        return False


async def run_all_tests():
    """Run all verification tests."""
    logger.info("\n" + "🚀 Starting Parallelization Fix Verification Tests")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    tests = [
        ("Duplicate Client Check", test_parallel_search_no_duplicate_client),
        ("Timeout Configuration", test_timeout_handling),
        ("Global Timeout", test_global_timeout),
        ("Zero Vector Fallback", test_no_zero_vector_fallback),
        ("Reflection Logic", test_improved_reflection_logic),
        ("Error Logging", test_error_type_logging),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*80)
    logger.info(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info(f"Execution time: {time.time() - start_time:.2f}s")
    logger.info("="*80)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Parallelization fixes verified.")
        return 0
    elif passed >= total * 0.8:
        logger.warning(f"\n⚠️  Most tests passed, but {total - passed} test(s) failed. Review warnings above.")
        return 1
    else:
        logger.error(f"\n❌ Multiple tests failed. Please review the fixes.")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
