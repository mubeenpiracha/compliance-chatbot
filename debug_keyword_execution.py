#!/usr/bin/env python3
"""
Debug script to test keyword search in the actual execution context.
"""
import sys
import asyncio
sys.path.append('./backend')

from backend.core.document_loader import load_document_corpus_from_content_store
from backend.core.retrieval.keyword_search import KeywordSearchEngine
from backend.core.models.retrieval_models import RetrievalQuery

async def debug_keyword_search():
    """Debug keyword search in execution context."""
    
    print("=== Debugging Keyword Search in Execution Context ===\n")
    
    # Load document corpus (same as in agent nodes)
    print("Loading document corpus...")
    document_corpus = load_document_corpus_from_content_store()
    print(f"Loaded {len(document_corpus)} documents\n")
    
    # Create keyword search engine (same as in agent nodes)
    keyword_search = KeywordSearchEngine(document_corpus=document_corpus)
    
    # Test the exact queries from the agent execution logs
    test_queries = [
        "DFSA Rulebook Collective Investment Funds Module definition of 'collective investment fund'",
        "DFSA glossary entry for 'collective investment fund'",
        "collective investment fund definition",
        "What are collective investment funds?"
    ]
    
    for query_text in test_queries:
        print(f"Testing query: '{query_text}'")
        print("-" * 60)
        
        # Create the exact same RetrievalQuery as in agent nodes
        retrieval_query = RetrievalQuery(
            query_text=query_text,
            query_type="fusion",  # Same as agent
            max_results=10,
            min_relevance_score=0.3,  # Same as agent
            target_domains=["difc"]  # Same as agent (lowercase jurisdiction)
        )
        
        try:
            # Perform keyword search
            keyword_results = await keyword_search.search(retrieval_query)
            
            print(f"Keyword search returned: {len(keyword_results)} results")
            
            if keyword_results:
                print("Top 3 results:")
                for i, doc in enumerate(keyword_results[:3]):
                    print(f"  {i+1}. Score: {doc.relevance_score:.3f}")
                    print(f"     Title: {doc.source.title}")
                    print(f"     Content: {doc.content[:100]}...")
                    print()
            else:
                print("❌ No keyword results found")
                
                # Debug why no results
                print("\nDebugging why no results...")
                
                # Test preprocessing
                query_terms = keyword_search._preprocess_text(query_text)
                print(f"Preprocessed query terms: {query_terms}")
                
                # Test a few documents manually
                print("Testing BM25 scoring on first 5 documents...")
                for i, doc in enumerate(document_corpus[:5]):
                    score = keyword_search._calculate_bm25_score(query_terms, doc)
                    print(f"  Doc {i+1}: {doc['metadata']['title'][:50]}... Score: {score:.4f}")
                    if score > 0:
                        print(f"    Content preview: {doc['content'][:100]}...")
                        print()
                
        except Exception as e:
            print(f"❌ Error in keyword search: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80)
        print()

if __name__ == "__main__":
    asyncio.run(debug_keyword_search())
