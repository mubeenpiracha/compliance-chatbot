# Synthesis Node Retry Fix

## Problem
The synthesis node was causing OpenAI API retries with the error:
```
INFO:openai._base_client:Retrying request to /chat/completions in 0.484205 seconds
```

This was leading to the error message:
```
"I found relevant information in the regulatory documents, but encountered an issue while synthesizing the response."
```

## Root Causes Identified

### 1. **Prompt Size Too Large**
The synthesis node was building prompts that could exceed OpenAI's token limits:
- Multiple search results (potentially 20-30+)
- Each result containing full document content (could be 5000+ characters)
- Conversation history
- System instructions
- **Result**: Total prompt could exceed 100k+ tokens, causing API errors or timeouts

### 2. **No Token Limits**
The API call didn't specify `max_tokens`, allowing the model to potentially generate very long responses that could timeout.

### 3. **Generic Error Handling**
The exception handler caught all errors with the same generic message, making it impossible to diagnose the actual issue.

## Fixes Applied

### 1. Limited Search Results
```python
MAX_SEARCH_RESULTS = 15  # Limit to prevent prompt from being too large
if len(search_results) > MAX_SEARCH_RESULTS:
    logger.warning(f"Truncating search results from {len(search_results)} to {MAX_SEARCH_RESULTS}")
    search_results = search_results[:MAX_SEARCH_RESULTS]
```

### 2. Truncated Individual Source Content
```python
MAX_CONTENT_LENGTH = 2000  # Max characters per source content
content = metadata.get('text', result.get('content', ''))
if len(content) > MAX_CONTENT_LENGTH:
    content = content[:MAX_CONTENT_LENGTH] + "... [content truncated]"
```

### 3. Added max_tokens Parameter
```python
response = await async_client.chat.completions.create(
    model="gpt-5-2025-08-07",
    messages=[...],
    max_tokens=4000  # Explicitly set max tokens for response
)
```

### 4. Enhanced Error Logging
```python
# Log prompt sizes for debugging
system_prompt_tokens = len(system_prompt) // 4  # Rough estimate
user_prompt_tokens = len(user_prompt) // 4
logger.info(f"Synthesis prompt sizes - System: ~{system_prompt_tokens} tokens, User: ~{user_prompt_tokens} tokens")
logger.info(f"Number of search results: {len(search_results)}")
```

### 5. Specific Error Messages
```python
except Exception as e:
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Full traceback:", exc_info=True)
    
    error_msg = str(e).lower()
    if "timeout" in error_msg:
        return {"final_response": "The synthesis took too long to complete..."}
    elif "token" in error_msg or "length" in error_msg:
        return {"final_response": "The retrieved documents are too extensive..."}
    elif "rate" in error_msg or "quota" in error_msg:
        return {"final_response": "API rate limit reached..."}
```

## Expected Outcomes

1. **Reduced API Retries**: By limiting prompt size, requests should complete within timeout
2. **Better Diagnostics**: Detailed logging will show exact prompt sizes and error types
3. **User-Friendly Errors**: Specific error messages guide users on what went wrong
4. **Improved Reliability**: Consistent response generation without timeouts

## Testing

To verify the fixes:

```bash
# Watch the logs when running a query
cd backend
tail -f logs/app.log | grep -E "Synthesis|prompt sizes|Error in generate_response"
```

Look for:
- `Synthesis prompt sizes` log showing reasonable token counts (< 50k tokens)
- `Number of search results` showing ≤ 15 results
- Successful synthesis without retries
- Specific error messages if issues occur

## Token Budget Estimation

**Before Fix:**
- 30 search results × 5000 chars = 150,000 chars ≈ **37,500 tokens**
- System prompt: ~5,000 chars ≈ **1,250 tokens**
- Conversation history: ~2,000 chars ≈ **500 tokens**
- **Total: ~39,250 tokens input** (exceeds many model limits)

**After Fix:**
- 15 search results × 2000 chars = 30,000 chars ≈ **7,500 tokens**
- System prompt: ~5,000 chars ≈ **1,250 tokens**
- Conversation history: ~2,000 chars ≈ **500 tokens**
- **Total: ~9,250 tokens input** (well within limits)
- **Max response: 4,000 tokens**
- **Total request: ~13,250 tokens** ✅

## Additional Recommendations

1. **Monitor Logs**: Check for truncation warnings to see if limits are being hit
2. **Adjust Limits**: If responses are incomplete, consider:
   - Increasing MAX_CONTENT_LENGTH from 2000 to 3000
   - Decreasing MAX_SEARCH_RESULTS from 15 to 10
3. **Query Optimization**: Encourage users to ask more specific questions
4. **Consider Chunking**: For very complex queries, break into multiple synthesis calls

## Files Modified
- `/home/mubeen/compliance-chatbot/backend/core/agent/nodes.py`
  - Lines ~465-470: Added search results limiting
  - Lines ~495-520: Added content truncation
  - Lines ~550-590: Enhanced error handling and logging
