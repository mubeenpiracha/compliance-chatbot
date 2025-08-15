#!/usr/bin/env python3
"""
Debug Pinecone search specifically
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
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def debug_pinecone():
    """Debug Pinecone search directly."""
    
    print("=== DEBUG PINECONE SEARCH ===")
    
    try:
        # Initialize services
        print("Initializing Pinecone service...")
        vector_service = RealVectorService()
        async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Test query
        query_text = "collective investment fund"
        
        print(f"Generating embedding for: {query_text}")
        response = await async_client.embeddings.create(
            model="text-embedding-3-large",
            input=query_text,
            dimensions=1536
        )
        query_embedding = response.data[0].embedding
        print(f"Generated embedding with {len(query_embedding)} dimensions")
        
        # Test Pinecone search directly
        print("\nSearching Pinecone index...")
        results = await vector_service.search(
            query_vector=query_embedding,
            top_k=10,
            filter_params=None  # No filters for debugging
        )
        
        print(f"Pinecone returned {len(results)} results")
        
        for i, result in enumerate(results[:5]):
            print(f"\nResult {i+1}:")
            print(f"  ID: {result['id']}")
            print(f"  Score: {result['score']}")
            print(f"  Namespace: {result.get('namespace', 'No namespace')}")
            print(f"  Content preview: {result['content'][:150]}...")
            print(f"  Metadata: {result.get('metadata', {})}")
        
        # Let's also check the Pinecone index stats
        print("\n=== PINECONE INDEX STATS ===")
        stats = vector_service.index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        # Check namespaces
        namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
        print(f"Available namespaces: {list(namespaces.keys()) if namespaces else 'None'}")
        
        # Try searching different namespaces if they exist
        if namespaces:
            for namespace in list(namespaces.keys())[:5]:  # Check first 5 namespaces
                print(f"\nSearching namespace: {namespace}")
                try:
                    response = vector_service.index.query(
                        vector=query_embedding,
                        top_k=3,
                        include_metadata=True,
                        namespace=namespace
                    )
                    print(f"  Found {len(response.matches)} results in namespace {namespace}")
                    for match in response.matches:
                        print(f"    Score: {match.score:.4f}, ID: {match.id}")
                except Exception as e:
                    print(f"    Error searching namespace {namespace}: {e}")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_pinecone())
