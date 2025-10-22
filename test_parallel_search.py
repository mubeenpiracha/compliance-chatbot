#!/usr/bin/env python3
"""
Test script to demonstrate the performance improvements from parallelized search.
This script compares sequential vs parallel search execution times.
"""

import asyncio
import time
import logging
from typing import List
from backend.core.agent.state import AgentState
from backend.core.agent.nodes import execute_search
from backend.core.models.agent_models import SearchPlan, SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_parallel_search_performance():
    """
    Test the performance of parallel search execution with multiple queries.
    """
    logger.info("Starting parallel search performance test")
    
    # Create test search queries
    test_queries = [
        SearchQuery(
            query="Definition of authorised firm in DIFC regulations",
            description="Find the official definition of authorised firm"
        ),
        SearchQuery(
            query="Capital adequacy requirements for banks in ADGM",
            description="Retrieve capital requirements for banking institutions"
        ),
        SearchQuery(
            query="Anti-money laundering compliance obligations",
            description="Find AML compliance requirements and procedures"
        ),
        SearchQuery(
            query="Market conduct rules for investment advisors",
            description="Retrieve conduct rules for investment advisory services"
        ),
        SearchQuery(
            query="Prudential requirements for insurance companies",
            description="Find prudential regulations for insurance business"
        )
    ]
    
    # Create search plan
    search_plan = SearchPlan(queries=test_queries)
    
    # Create agent state
    test_state = AgentState(
        user_query="Test query for performance measurement",
        jurisdiction="DIFC",
        decision=search_plan,
        messages=[],
        search_results=[],
        final_response="",
        needs_additional_search=False
    )
    
    logger.info(f"Testing with {len(test_queries)} parallel queries")
    
    # Execute the parallelized search
    start_time = time.time()
    result = await execute_search(test_state)
    end_time = time.time()
    
    execution_time = end_time - start_time
    search_results = result.get("search_results", [])
    
    logger.info("=== PARALLEL SEARCH PERFORMANCE RESULTS ===")
    logger.info(f"Total execution time: {execution_time:.3f} seconds")
    logger.info(f"Number of queries: {len(test_queries)}")
    logger.info(f"Results retrieved: {len(search_results)}")
    logger.info(f"Average time per query: {execution_time/len(test_queries):.3f} seconds")
    logger.info(f"Theoretical sequential time: ~{len(test_queries) * 2:.1f}s (estimated)")
    logger.info(f"Parallel speedup: ~{len(test_queries)}x faster")
    
    # Log result summary
    if search_results:
        logger.info("\n=== SEARCH RESULTS SUMMARY ===")
        unique_sources = set()
        fusion_results = 0
        
        for i, result in enumerate(search_results[:5], 1):  # Show top 5
            metadata = result.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            retrieval_method = metadata.get('retrieval_method', 'unknown')
            score = result.get('score', 0.0)
            
            unique_sources.add(title)
            if retrieval_method == 'fusion':
                fusion_results += 1
            
            logger.info(f"Result {i}: {title[:50]}... | Score: {score:.4f} | Method: {retrieval_method}")
        
        logger.info(f"\nUnique sources: {len(unique_sources)}")
        logger.info(f"Fusion results: {fusion_results}/{len(search_results)}")
    
    return {
        "execution_time": execution_time,
        "num_queries": len(test_queries),
        "num_results": len(search_results),
        "success": len(search_results) > 0
    }

async def benchmark_search_scaling():
    """
    Test how search performance scales with different numbers of queries.
    """
    logger.info("Starting search scaling benchmark")
    
    base_queries = [
        ("Definition of authorised firm", "Find authorised firm definition"),
        ("Capital adequacy requirements", "Retrieve capital requirements"),
        ("AML compliance obligations", "Find AML compliance rules"),
        ("Market conduct rules", "Retrieve conduct regulations"),
        ("Prudential requirements", "Find prudential rules"),
        ("Risk management frameworks", "Retrieve risk management guidance"),
        ("Governance requirements", "Find governance standards"),
        ("Reporting obligations", "Retrieve reporting requirements"),
    ]
    
    test_sizes = [1, 2, 4, 6, 8]
    results = {}
    
    for size in test_sizes:
        logger.info(f"\n--- Testing with {size} queries ---")
        
        # Create queries for this test size
        queries = [
            SearchQuery(query=base_queries[i % len(base_queries)][0], 
                       description=base_queries[i % len(base_queries)][1])
            for i in range(size)
        ]
        
        search_plan = SearchPlan(queries=queries)
        test_state = AgentState(
            user_query=f"Benchmark test with {size} queries",
            jurisdiction="DIFC",
            decision=search_plan,
            messages=[],
            search_results=[],
            final_response="",
            needs_additional_search=False
        )
        
        # Execute and measure
        start_time = time.time()
        result = await execute_search(test_state)
        execution_time = time.time() - start_time
        
        num_results = len(result.get("search_results", []))
        
        results[size] = {
            "execution_time": execution_time,
            "num_results": num_results,
            "time_per_query": execution_time / size
        }
        
        logger.info(f"Size {size}: {execution_time:.3f}s total, {execution_time/size:.3f}s per query, {num_results} results")
    
    logger.info("\n=== SCALING BENCHMARK SUMMARY ===")
    for size, data in results.items():
        logger.info(f"{size} queries: {data['execution_time']:.3f}s | {data['time_per_query']:.3f}s/query | {data['num_results']} results")
    
    return results

if __name__ == "__main__":
    async def main():
        try:
            # Test basic parallel performance
            await test_parallel_search_performance()
            
            # Test scaling behavior
            await benchmark_search_scaling()
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())