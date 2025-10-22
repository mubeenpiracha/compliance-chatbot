# Reflection Optimization - Eliminate Unnecessary OpenAI Calls

## Current Problem

The reflection system uses **2 unnecessary OpenAI API calls**:

### Current Flow (WASTEFUL):
```
User Query
  ↓
analyze_query() → 1 OpenAI call
  ↓
execute_search() → N embedding calls
  ↓
generate_response() → 1 OpenAI call (synthesis)
  ↓
Pattern matching (free) → decides if reflection needed
  ↓
reflection_node() → Pattern matching AGAIN (duplicate!)
  ↓
reflection_node() → 1 OpenAI call (analyze what's missing) ⚠️ UNNECESSARY!
  ↓
execute_search() again → M embedding calls
  ↓
generate_response() again → 1 OpenAI call (re-synthesis)
```

**Total for query with reflection: 1 + N + 1 + 1 + M + 1 = N + M + 4 OpenAI calls**

The reflection analysis call is almost always unnecessary!

---

## Why the Reflection OpenAI Call is Unnecessary

### What the Call Does (lines 705-713):
```python
response = await async_client.chat.completions.create(
    model="gpt-5-2025-08-07",
    messages=[...],
    response_format={"type": "json_object"}
)
```

Asks GPT to:
1. Identify missing documents
2. Extract section numbers
3. Generate search queries

### Why It's Wasteful:

1. **The synthesis response already tells us what's missing!**
   - "The extract is partial and should be confirmed against the full COB Module Section 4.2"
   - We can parse this with regex!

2. **Simple text extraction works better:**
   ```python
   # Extract document references
   doc_pattern = r'([A-Z]{3,})\s+(?:Module|Rulebook|Section)\s+([\d\.]+)'
   matches = re.findall(doc_pattern, final_response)
   # Result: [('COB', '4.2'), ('GEN', '3.1')]
   ```

3. **GPT's analysis adds no value:**
   - If response says "need the full table", we search for the table
   - If response mentions "Section 4.2", we search for Section 4.2
   - No AI inference needed!

4. **Cost Analysis:**
   - GPT-5 input: ~1000 tokens (system prompt + response)
   - GPT-5 output: ~200 tokens (JSON analysis)
   - Cost: ~$0.015 per reflection
   - 20% of queries trigger reflection = ~$0.003 per query overhead
   - With 10k queries/day = $30/day wasted = **$900/month waste**

---

## Optimized Approach

### Option 1: Rule-Based Reflection (RECOMMENDED)

```python
async def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes response for incomplete information using rule-based extraction.
    NO OpenAI call needed - just parse the response text.
    """
    logger.info("Node: reflection_node")
    
    final_response = state.get("final_response", "")
    jurisdiction = state["jurisdiction"]
    reflection_count = state.get("reflection_count", 0) + 1
    
    # Quick pattern check (if not already done)
    critical_indicators = [
        "extract is partial",
        "should be confirmed against the full",
        "not visible in the provided extract",
        "requires the full document"
    ]
    
    response_lower = final_response.lower()
    has_incomplete_info = any(ind.lower() in response_lower for ind in critical_indicators)
    
    if not has_incomplete_info:
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }
    
    # RULE-BASED EXTRACTION (No OpenAI call!)
    new_queries = []
    
    # Pattern 1: Extract document + section references
    # "COB Module Section 4.2" or "GEN Rulebook Chapter 3"
    doc_section_pattern = r'([A-Z]{2,})\s+(?:Module|Rulebook|Rules?|Section|Chapter)\s+([\d\.]+(?:\([a-z]\))?)'
    doc_matches = re.findall(doc_section_pattern, final_response)
    
    for doc_code, section in doc_matches:
        query = SearchQuery(
            query=f"{doc_code} {section} full text",
            description=f"Retrieve complete {doc_code} Section {section}"
        )
        new_queries.append(query)
    
    # Pattern 2: Extract table references
    # "Table 4.2" or "Table A-1"
    table_pattern = r'[Tt]able\s+([\d\.]+(?:-[A-Z\d]+)?)'
    table_matches = re.findall(table_pattern, final_response)
    
    for table_ref in table_matches:
        query = SearchQuery(
            query=f"Table {table_ref} {jurisdiction}",
            description=f"Retrieve complete Table {table_ref}"
        )
        new_queries.append(query)
    
    # Pattern 3: Extract definition requests
    # "definition of 'Authorised Firm'" or "defined as an Authorised Firm"
    definition_pattern = r"definition of ['\"]([^'\"]+)['\"]"
    def_matches = re.findall(definition_pattern, final_response, re.IGNORECASE)
    
    for term in def_matches:
        query = SearchQuery(
            query=f"definition {term} {jurisdiction} glossary",
            description=f"Retrieve complete definition of '{term}'"
        )
        new_queries.append(query)
    
    # Pattern 4: Generic document request
    # If patterns mention "full document" but no specific reference found
    if not new_queries and "full document" in response_lower:
        # Extract likely document name from context
        doc_keywords = ["module", "rulebook", "regulation", "rules", "law"]
        for keyword in doc_keywords:
            if keyword in response_lower:
                # Get surrounding context
                context_pattern = rf'(\w+\s+{keyword})'
                context_matches = re.findall(context_pattern, response_lower)
                for match in context_matches[:2]:  # Max 2 fallback queries
                    query = SearchQuery(
                        query=f"{match} {jurisdiction} full text",
                        description=f"Retrieve complete {match}"
                    )
                    new_queries.append(query)
    
    # Deduplicate queries
    unique_queries = []
    seen = set()
    for query in new_queries:
        if query.query not in seen:
            unique_queries.append(query)
            seen.add(query.query)
    
    logger.info(f"Reflection extracted {len(unique_queries)} queries via rule-based parsing (no OpenAI call)")
    
    if unique_queries:
        search_plan = SearchPlan(queries=unique_queries[:3])  # Limit to 3 additional searches
        return {
            "decision": search_plan,
            "search_plan": search_plan,
            "needs_additional_search": True,
            "reflection_count": reflection_count
        }
    else:
        logger.info("No specific references found in reflection, skipping additional search")
        return {
            "needs_additional_search": False,
            "reflection_count": reflection_count
        }
```

### Benefits:
- ✅ **0 OpenAI calls** in reflection (vs 1 currently)
- ✅ **Faster** - regex is instant vs 2-3s OpenAI call
- ✅ **Cheaper** - saves $900/month
- ✅ **More reliable** - no AI hallucination in parsing
- ✅ **Easier to debug** - deterministic rules

---

### Option 2: Smarter Initial Check (ALSO RECOMMENDED)

Move pattern detection BEFORE synthesis to avoid unnecessary synthesis calls:

```python
async def pre_synthesis_check(state: AgentState) -> Dict[str, Any]:
    """
    Quick check: Do we have enough high-quality results?
    If not, trigger additional search BEFORE synthesis.
    """
    search_results = state.get("search_results", [])
    
    # Check 1: Do we have enough results?
    if len(search_results) < 3:
        logger.warning("Only {len(search_results)} results, may need broader search")
        return {"pre_synthesis_warning": "low_result_count"}
    
    # Check 2: Are results high relevance?
    avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
    if avg_score < 0.5:
        logger.warning(f"Low average relevance score: {avg_score}")
        return {"pre_synthesis_warning": "low_relevance"}
    
    # Check 3: Do we have content diversity?
    unique_docs = set(r.get('metadata', {}).get('title', '') for r in search_results)
    if len(unique_docs) < 2:
        logger.warning("Results from single document, may be too narrow")
        return {"pre_synthesis_warning": "low_diversity"}
    
    return {"pre_synthesis_warning": None}
```

---

## Implementation Plan

### Phase 1: Quick Win (THIS WEEK) 🚀
**Replace OpenAI call in reflection_node with rule-based extraction**

Files to modify:
- `backend/core/agent/nodes.py` - reflection_node function

Expected savings:
- 1 OpenAI call per reflection (20% of queries)
- 0.2 calls per query average
- ~$900/month cost savings
- 2-3s faster reflection processing

### Phase 2: Optimization (NEXT SPRINT)
**Add pre-synthesis quality check**

Would catch issues before expensive synthesis call, but requires:
- Graph flow changes
- Additional routing logic
- More testing

---

## Testing Strategy

### Test Cases for Rule-Based Reflection:

1. **Document + Section Reference:**
   - Input: "The extract is partial. Please confirm against COB Module Section 4.2"
   - Expected: Extract "COB" + "4.2" → query "COB 4.2 full text"

2. **Table Reference:**
   - Input: "The full Table 3.1 is needed to show all fee categories"
   - Expected: Extract "3.1" → query "Table 3.1 DIFC"

3. **Definition Request:**
   - Input: "The definition of 'Authorised Firm' should be confirmed"
   - Expected: Extract "Authorised Firm" → query "definition Authorised Firm DIFC glossary"

4. **Multiple References:**
   - Input: "COB 4.2 and GEN 3.1 should be consulted"
   - Expected: 2 queries extracted

5. **No Specific Reference:**
   - Input: "More information needed"
   - Expected: No queries generated (graceful degradation)

### Regression Tests:
- Ensure reflection still triggers when needed
- Verify no infinite loops
- Check that new queries are relevant
- Confirm no performance degradation

---

## Metrics to Track

### Before (Current):
- OpenAI calls per query: 5.3
- Reflection OpenAI calls: 0.2 (20% of queries)
- Reflection processing time: 2.5s avg
- Monthly OpenAI cost: $X

### After (Optimized):
- OpenAI calls per query: 5.1 (5% reduction)
- Reflection OpenAI calls: 0 (100% reduction)
- Reflection processing time: 0.1s avg (25x faster)
- Monthly OpenAI cost: $X - $900 (savings)

---

## Decision: Implement Phase 1 NOW

Rule-based reflection is:
- ✅ Low risk (fallback: skip reflection on parsing failure)
- ✅ High reward ($900/month + 0.2 calls/query)
- ✅ Quick to implement (2-3 hours)
- ✅ Easy to test
- ✅ Backward compatible

No reason not to do this immediately!
