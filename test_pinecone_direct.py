"""
Test direct Pinecone connection to see what's in the index.
"""
import asyncio
from pinecone import Pinecone
from backend.core.config import PINECONE_API_KEY
from openai import AsyncOpenAI
from backend.core.config import OPENAI_API_KEY

async def test_pinecone_direct():
    """Test direct Pinecone connection."""
    
    print("Testing Direct Pinecone Connection")
    print("=" * 40)
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("compliance-bot-index")
        
        # Check index stats
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        # Test with a simple embedding
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input="fund definition",
            dimensions=1536
        )
        query_vector = response.data[0].embedding
        
        # Query with no filters
        print(f"\nQuerying with 'fund definition'...")
        result = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            include_values=False
        )
        
        print(f"Found {len(result.matches)} matches")
        for i, match in enumerate(result.matches):
            print(f"{i+1}. ID: {match.id}")
            print(f"   Score: {match.score}")
            print(f"   Metadata keys: {list(match.metadata.keys()) if match.metadata else 'None'}")
            if match.metadata:
                print(f"   File: {match.metadata.get('file_name', 'Unknown')}")
                print(f"   Content URI: {match.metadata.get('content_uri', 'Unknown')}")
        
        # Try querying different namespaces
        print(f"\nListing namespaces...")
        all_namespaces = set()
        try:
            # Query without namespace to see what's available
            general_result = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            print(f"General query returned {len(general_result.matches)} results")
        except Exception as e:
            print(f"General query failed: {e}")
            
    except Exception as e:
        print(f"Direct Pinecone test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pinecone_direct())
