#!/usr/bin/env python3
"""
Simple test of vector search after fix
"""
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.real_vector_service import RealVectorService
from backend.core.retrieval.vector_search import VectorSearchEngine  
from backend.core.models.retrieval_models import RetrievalQuery
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def simple_test():
    print("=== SIMPLE TEST OF FIXED VECTOR SEARCH ===")
    
    # Initialize
    vector_service = RealVectorService()
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    vector_search = VectorSearchEngine(client=async_client, vector_service=vector_service)
    
    # Create query
    retrieval_query = RetrievalQuery(
        query_text="collective investment fund",
        query_type="vector",
        max_results=5,
        min_relevance_score=0.1,
        target_domains=["difc"]
    )
    
    print(f"Testing query: {retrieval_query.query_text}")
    
    # Test the search
    results = await vector_search.search(retrieval_query)
    
    print(f"Found {len(results)} results")
    for i, doc in enumerate(results):
        print(f"{i+1}. Score: {doc.relevance_score:.4f}")
        print(f"   Source: {doc.source.document_id}")
        print(f"   Title: {doc.source.title}")
        print(f"   Content: {doc.content[:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(simple_test())
