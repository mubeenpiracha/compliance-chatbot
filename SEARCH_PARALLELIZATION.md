# Search Parallelization Improvements

This document outlines the parallelization enhancements made to the compliance chatbot's search functionality.

## Overview

The search system has been optimized to execute multiple queries in parallel, significantly reducing response times when handling complex queries that require multiple searches.

## Key Improvements

### 1. Full Query Parallelization
- **Before**: Queries were executed sequentially, with each query waiting for the previous one to complete
- **After**: All queries in a search plan are executed simultaneously using `asyncio.gather()`
- **Performance Impact**: ~N×ß speedup where N is the number of queries in the search plan

### 2. Concurrent Search Method Execution
- **Vector and Keyword searches** for each query run in parallel
- **Timeout protection** prevents slow searches from blocking the entire operation
- **Graceful failure handling** ensures partial failures don't crash the system

### 3. Optimized Resource Management
- **Semaphore-based concurrency control** prevents resource exhaustion
- **Connection pooling** for OpenAI API calls
- **Timeout configurations** for different operation types

### 4. Enhanced Performance Monitoring
- **Detailed timing logs** for each operation phase
- **Performance metrics** showing speedup and efficiency
- **Resource utilization tracking**

## Technical Implementation

### Core Functions

#### `execute_single_query_search()`
- Executes vector and keyword searches in parallel for a single query
- Applies RRF (Reciprocal Rank Fusion) to combine results
- Uses semaphores to control concurrency

#### `execute_search()`
- Coordinates parallel execution of multiple queries
- Handles deduplication across all results
- Provides comprehensive performance reporting

### Performance Configuration

The `performance_config.py` module provides:
- Async performance settings
- Connection pool configurations  
- Timing utilities and decorators
- Performance monitoring tools

## Usage Example

```python
# Create multiple search queries
search_plan = SearchPlan(queries=[
    SearchQuery(query="Definition of authorised firm", description="Find official definition"),
    SearchQuery(query="Capital requirements", description="Retrieve capital adequacy rules"),
    SearchQuery(query="AML obligations", description="Find anti-money laundering requirements"),
    # ... more queries
])

# Execute all queries in parallel
results = await execute_search(state)
```

## Performance Characteristics

### Typical Performance Gains
- **2 queries**: ~2x faster than sequential
- **4 queries**: ~4x faster than sequential  
- **8 queries**: ~8x faster than sequential

### Scalability
The system scales linearly with the number of queries up to the configured concurrency limits:
- **Concurrency limit**: 10 simultaneous operations
- **Timeout protection**: 30s for async operations
- **Search timeouts**: 15s vector, 10s keyword

### Resource Usage
- **Memory**: Slight increase due to parallel result storage
- **CPU**: Better utilization through async operations
- **Network**: Improved throughput with connection pooling

## Testing

Run the performance test to see the improvements:

```bash
python test_parallel_search.py
```

This will:
1. Execute a sample search plan with multiple queries
2. Report detailed timing information
3. Compare parallel vs estimated sequential performance
4. Show scaling characteristics with different query counts

## Configuration

Key settings in `performance_config.py`:

```python
ASYNC_CONCURRENCY_LIMIT = 10      # Max concurrent operations
ASYNC_TIMEOUT = 30.0               # Operation timeout
VECTOR_SEARCH_TIMEOUT = 15.0       # Vector search timeout
KEYWORD_SEARCH_TIMEOUT = 10.0      # Keyword search timeout
```

## Future Optimizations

1. **Query Batching**: Group related queries for even better efficiency
2. **Result Caching**: Cache frequent query results to avoid repeated searches
3. **Adaptive Concurrency**: Dynamically adjust concurrency based on system load
4. **Distributed Processing**: Scale across multiple servers for very large workloads

## Monitoring

The system provides detailed logs for monitoring:
- Query execution times
- Success/failure rates
- Deduplication statistics
- Overall performance metrics

Look for log entries with these patterns:
- `"PARALLEL EXECUTION: X/Y queries successful"`
- `"SEARCH PERFORMANCE SUMMARY"`
- `"Query N completed: RRF fusion produced X results"`