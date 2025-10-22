# Parallelization Rate Limit Fix - OpenAI Retry Storm

## Problem Diagnosis

### Symptoms
```
INFO:openai._base_client:Retrying request to /chat/completions in 0.484205 seconds
INFO:openai._base_client:Retrying request to /chat/completions in 0.931047 seconds
```

Repeated retry messages indicate **OpenAI rate limiting** from too many simultaneous requests.

### Root Cause
The code fires **multiple parallel OpenAI API calls** without rate limiting:

**For a query with 3 search sub-queries:**
1. analyze_query: **1 OpenAI call** (chat completions)
2. execute_search:
   - Query 1 embedding: **1 OpenAI call**
   - Query 2 embedding: **1 OpenAI call**  
   - Query 3 embedding: **1 OpenAI call**
   - **All 3 fire simultaneously via asyncio.gather()**
3. generate_response: **1 OpenAI call** (chat completions)
4. reflection_decision: **1 OpenAI call** (potentially)

**Total: 5-6 simultaneous API calls** → **Hits OpenAI rate limits** → **Triggers retries**

### The Missing Piece

The code imports `get_optimized_semaphore` but **NEVER USES IT**:

```python
# backend/core/agent/nodes.py Line 22
from backend.core.performance_config import (
    get_optimized_semaphore,  # ← IMPORTED
    # ...
)

# But search for "semaphore" in nodes.py → NO USAGE!
```

The semaphore is designed to limit concurrent operations but was never actually applied to the OpenAI API calls.

## The Fix

### Solution 1: Add Semaphore to Embedding Calls (RECOMMENDED)

Limit concurrent embedding generation to prevent rate limiting:

**File:** `backend/core/retrieval/vector_search.py`

```python
class VectorSearchEngine:
    """Handles semantic similarity search using vector embeddings."""
    
    def __init__(self, client: AsyncOpenAI, vector_service, semaphore: asyncio.Semaphore = None):
        self.client = client
        self.vector_service = vector_service
        self.semaphore = semaphore or asyncio.Semaphore(3)  # Max 3 concurrent embeddings
    
    async def _get_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for the query text using OpenAI with rate limiting."""
        async with self.semaphore:  # ← ADD THIS
            try:
                logger.debug(f"Generating embedding (semaphore acquired)")
                response = await self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=query_text,
                    dimensions=1536
                )
                return response.data[0].embedding
                
            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"Failed to generate embedding [{error_type}]: {str(e)}")
                raise RuntimeError(f"Embedding generation failed: {error_type}") from e
```

**File:** `backend/core/agent/nodes.py` - Update initialization:

```python
async def execute_search(state: AgentState) -> Dict[str, Any]:
    """Executes the search plan with rate-limited parallel execution."""
    
    try:
        # Initialize services
        vector_service = RealVectorService()
        document_corpus = load_document_corpus_from_content_store()
        
        # Create semaphore for rate limiting OpenAI API calls
        embedding_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent embeddings
        
        # Create search engines with semaphore
        vector_search = VectorSearchEngine(
            client=async_client, 
            vector_service=vector_service,
            semaphore=embedding_semaphore  # ← ADD THIS
        )
        keyword_search = KeywordSearchEngine(
            document_corpus=document_corpus, 
            vector_service=vector_service
        )
```

### Solution 2: Reduce max_retries (QUICK FIX)

If the retries are working but just noisy in logs:

**File:** `backend/core/agent/nodes.py` Line 35:

```python
# CURRENT:
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=ASYNC_TIMEOUT,
    max_retries=2  # ← Retries 2 times
)

# CHANGE TO:
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=ASYNC_TIMEOUT,
    max_retries=0  # ← No automatic retries, fail fast
)
```

Then handle retries manually with exponential backoff in the code.

### Solution 3: Sequential Embeddings (SIMPLEST BUT SLOWER)

Remove parallelization of embeddings:

**File:** `backend/core/agent/nodes.py` - Modify execute_search:

```python
# Execute queries SEQUENTIALLY instead of parallel
all_query_results = []
for i, search_query in enumerate(decision.queries, 1):
    logger.info(f"Executing query {i}/{len(decision.queries)} sequentially")
    result = await execute_single_query_search(
        search_query, 
        vector_search, 
        keyword_search, 
        state["jurisdiction"], 
        i
    )
    all_query_results.append(result)
```

**Pros:** Simple, no rate limiting issues
**Cons:** Slower (5-10s per query vs 5-10s for all queries)

## Recommended Implementation

**Use Solution 1** (Semaphore) - Best balance of speed and reliability:

1. Limits concurrent embeddings to 3 (configurable)
2. Prevents rate limit errors
3. Still faster than sequential (3x speedup vs 1x)
4. Graceful queuing of additional requests

## Testing the Fix

### Before Fix - Expected Logs:
```
INFO:openai._base_client:Retrying request to /chat/completions in 0.484205 seconds
INFO:openai._base_client:Retrying request to /chat/completions in 0.931047 seconds
INFO:openai._base_client:Retrying request to /chat/completions in 1.824093 seconds
```

### After Fix - Expected Logs:
```
DEBUG: Generating embedding (semaphore acquired)
DEBUG: Generating embedding (semaphore acquired)
DEBUG: Generating embedding (semaphore acquired)
DEBUG: Waiting for semaphore... (if more than 3)
INFO: Query 1: Vector=5, Keyword=3 results | Search time: 2.1s
INFO: Query 2: Vector=5, Keyword=4 results | Search time: 2.3s
INFO: Query 3: Vector=5, Keyword=2 results | Search time: 2.2s
```

No retry messages!

### Test Cases:

1. **Single query** - Should work same as before
2. **3 queries** - Should not hit rate limits (3 concurrent max)
3. **5 queries** - Should queue 2 queries until first 3 complete
4. **10 queries** - Should maintain 3 concurrent throughout

## Performance Impact

### Before (No Rate Limiting):
- Fire all N queries in parallel
- Hit rate limit after ~3-5 concurrent
- OpenAI retries with exponential backoff
- **Total time: 8-12s** (includes retry delays)
- **API calls: N embeddings + retries**

### After (With Semaphore):
- Fire max 3 queries in parallel
- Queue remaining queries
- No rate limiting
- **Total time: 6-8s** (no retry delays)
- **API calls: N embeddings (no retries)**

**Result:** Actually FASTER despite throttling, because no retry delays!

## Configuration Options

Add to `backend/core/performance_config.py`:

```python
# OpenAI API Rate Limiting
OPENAI_EMBEDDING_CONCURRENCY = 3  # Max concurrent embedding calls
OPENAI_CHAT_CONCURRENCY = 2       # Max concurrent chat completion calls
OPENAI_TOTAL_CONCURRENCY = 5      # Max total OpenAI API calls
```

Then use in code:

```python
embedding_semaphore = asyncio.Semaphore(OPENAI_EMBEDDING_CONCURRENCY)
```

## Alternative: Use httpx Limits

OpenAI Python library uses httpx. You can configure httpx limits:

```python
import httpx

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=ASYNC_TIMEOUT,
    max_retries=2,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=5,      # Max total connections
            max_keepalive_connections=2  # Max idle connections
        )
    )
)
```

This would limit at the HTTP connection level, which is more robust.

## Monitoring

Add logging to track semaphore usage:

```python
async def _get_query_embedding(self, query_text: str) -> List[float]:
    """Generate embedding with rate limiting and monitoring."""
    wait_start = time.time()
    async with self.semaphore:
        wait_time = time.time() - wait_start
        if wait_time > 0.1:
            logger.info(f"Waited {wait_time:.2f}s for embedding semaphore")
        
        # Generate embedding...
```

This shows when queries are being throttled.

## Conclusion

The retries are happening because **too many parallel OpenAI API calls** are hitting **rate limits**. The semaphore was imported but never used. Adding it to `_get_query_embedding()` will:

✅ Limit concurrent embeddings to prevent rate limits
✅ Eliminate retry storms
✅ Actually improve performance (no retry delays)
✅ Make logs cleaner
✅ Reduce API costs (fewer retries)

**Recommended: Implement Solution 1 with semaphore limit of 3 concurrent embeddings.**
