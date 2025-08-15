#!/usr/bin/env python3

import json
import requests
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.retrieval.keyword_search import KeywordSearchEngine
from backend.core.document_loader import load_document_corpus_from_content_store
from backend.core.models.retrieval_models import RetrievalQuery

def test_chat_endpoint():
    """Test the chat endpoint to see what retrieval method is actually being used."""
    url = "http://localhost:8000/api/v1/chat/"
    
    payload = {
        "message": "What are collective investment funds?",
        "jurisdiction": "DIFC",
        "history": []
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing Chat Endpoint...")
        print("=" * 60)
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response received.")
            
            # Check the sources to see what retrieval methods were used
            sources = result.get('sources', [])
            retrieval_methods = [source.get('retrieval_method') for source in sources]
            
            print(f"\nFound {len(sources)} sources:")
            for i, source in enumerate(sources[:5]):  # Show first 5
                method = source.get('retrieval_method', 'unknown')
                score = source.get('score', 0)
                filename = source.get('filename', 'unknown')
                print(f"  {i+1}. Method: {method}, Score: {score:.3f}, File: {filename}")
            
            print(f"\nRetrieval methods used: {set(retrieval_methods)}")
            
            # Check if keyword search was used
            if 'keyword' in retrieval_methods:
                print("✅ Keyword search WAS used")
            else:
                print("❌ Keyword search was NOT used - only vector search")
                
        else:
            print("ERROR!")
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

async def test_keyword_search_directly():
    """Test keyword search engine directly to see if it works."""
    print("\nTesting Keyword Search Engine Directly...")
    print("=" * 60)
    
    try:
        # Load document corpus
        print("Loading document corpus...")
        document_corpus = load_document_corpus_from_content_store()
        print(f"Loaded {len(document_corpus)} documents")
        
        # Initialize keyword search engine
        keyword_search = KeywordSearchEngine(document_corpus=document_corpus)
        
        # Create test query
        test_query = RetrievalQuery(
            query_text="collective investment fund definition",
            query_type="keyword",
            max_results=5,
            min_relevance_score=0.1,
            target_domains=["difc"]
        )
        
        print(f"Testing query: '{test_query.query_text}'")
        
        # Perform keyword search
        results = await keyword_search.search(test_query)
        
        print(f"Keyword search returned {len(results)} results:")
        
        for i, doc in enumerate(results):
            print(f"\n{i+1}. Score: {doc.relevance_score:.4f}")
            print(f"   Source: {doc.source.document_id}")
            print(f"   Title: {doc.source.title}")
            print(f"   Method: {doc.retrieval_method}")
            print(f"   Content: {doc.content[:150]}...")
            if doc.match_highlights:
                print(f"   Highlights: {doc.match_highlights[:2]}")
        
        if len(results) > 0:
            print("\n✅ Keyword search is working!")
        else:
            print("\n❌ Keyword search returned no results")
            
    except Exception as e:
        print(f"Direct keyword search failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Test 1: Chat endpoint (what user is experiencing)
    test_chat_endpoint()
    
    # Test 2: Direct keyword search (to verify it works)
    print("\n" + "=" * 80 + "\n")
    asyncio.run(test_keyword_search_directly())

if __name__ == "__main__":
    main()
