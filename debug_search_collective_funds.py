#!/usr/bin/env python3
"""
Debug script to test search for 'DIFC glossary collective investment funds'
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.core.real_vector_service import RealVectorService
from backend.core.retrieval.vector_search import VectorSearchEngine
from backend.core.retrieval.keyword_search import KeywordSearchEngine
from backend.core.models.retrieval_models import RetrievalQuery
from backend.core.document_loader import load_document_corpus_from_content_store
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def debug_search():
    """Debug search for collective investment funds in DIFC glossary."""
    
    print("=== DEBUG SEARCH FOR COLLECTIVE INVESTMENT FUNDS ===")
    
    # Test queries
    test_queries = [
        "DIFC glossary collective investment funds",
        "collective investment fund definition DIFC", 
        "definition collective investment fund",
        "glossary collective investment",
        "DIFC GLO collective investment fund",
        "collective investment funds",
        "what is collective investment fund"
    ]
    
    try:
        # Initialize services
        print("Initializing services...")
        vector_service = RealVectorService()
        async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        document_corpus = load_document_corpus_from_content_store()
        
        # Create search engines
        vector_search = VectorSearchEngine(client=async_client, vector_service=vector_service)
        keyword_search = KeywordSearchEngine(document_corpus=document_corpus)
        
        print(f"Document corpus loaded with {len(document_corpus)} chunks")
        print()
        
        # Test each query
        for i, query_text in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"QUERY {i}: {query_text}")
            print('='*60)
            
            # Create retrieval query
            retrieval_query = RetrievalQuery(
                query_text=query_text,
                query_type="fusion",
                max_results=10,
                min_relevance_score=0.1,  # Lower threshold for debugging
                target_domains=["difc"]
            )
            
            print(f"Retrieval query: {retrieval_query}")
            print()
            
            # Test vector search
            print("--- VECTOR SEARCH ---")
            try:
                vector_results = await vector_search.search(retrieval_query)
                print(f"Vector search returned {len(vector_results)} results")
                
                for j, doc in enumerate(vector_results[:3]):  # Show top 3
                    print(f"  {j+1}. Score: {doc.relevance_score:.4f}")
                    print(f"     Source: {doc.source.document_id}")
                    print(f"     Title: {doc.source.title}")
                    print(f"     Content preview: {doc.content[:150]}...")
                    print()
                    
            except Exception as e:
                print(f"Vector search failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Test keyword search  
            print("--- KEYWORD SEARCH ---")
            try:
                keyword_results = await keyword_search.search(retrieval_query)
                print(f"Keyword search returned {len(keyword_results)} results")
                
                for j, doc in enumerate(keyword_results[:3]):  # Show top 3
                    print(f"  {j+1}. Score: {doc.relevance_score:.4f}")
                    print(f"     Source: {doc.source.document_id}")
                    print(f"     Title: {doc.source.title}")
                    print(f"     Content preview: {doc.content[:150]}...")
                    print()
                    
            except Exception as e:
                print(f"Keyword search failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Check if either search found results
            total_results = len(vector_results) + len(keyword_results)
            if total_results == 0:
                print(f"❌ NO RESULTS for query: {query_text}")
            else:
                print(f"✅ Found {total_results} total results for query: {query_text}")
                
            print("-" * 40)
            
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== DIRECT CONTENT SEARCH ===")
    # Let's also search the content store directly
    try:
        content_store_path = Path("./content_store")
        found_files = []
        
        # Search for files that might contain collective investment fund definitions
        for root, dirs, files in os.walk(content_store_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        content_lower = content.lower()
                        
                        # Check if this chunk contains relevant terms
                        if ('collective investment' in content_lower and 
                            ('fund' in content_lower or 'funds' in content_lower)):
                            found_files.append({
                                'path': file_path,
                                'content_preview': content[:200] + "..." if len(content) > 200 else content
                            })
                            
                    except Exception as e:
                        continue
        
        print(f"Found {len(found_files)} content files mentioning 'collective investment fund':")
        for i, file_info in enumerate(found_files[:5]):  # Show first 5
            print(f"{i+1}. {file_info['path']}")
            print(f"   Preview: {file_info['content_preview']}")
            print()
            
    except Exception as e:
        print(f"Direct content search failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_search())
