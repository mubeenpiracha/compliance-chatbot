"""
Test to find the specific fund definition chunk.
"""
import asyncio
from backend.core.real_vector_service import RealVectorService
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def find_definition_chunk():
    """Find the specific fund definition chunk."""
    
    print("Searching for Fund Definition Chunk")
    print("=" * 40)
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        vector_service = RealVectorService()
        
        # Query that should match the definition we found manually
        query = "In this Part Collective Investment Fund means any arrangements with respect to property"
        
        # Generate embedding
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
            dimensions=1536
        )
        query_vector = response.data[0].embedding
        
        # Search specifically in the financial services regulations namespace
        results = await vector_service.search(
            query_vector=query_vector,
            top_k=10
        )
        
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            title = result['metadata']['title']
            score = result['score']
            content = result['content']
            
            print(f"\n{i}. {title} (score: {score:.3f})")
            print(f"   Namespace: {result.get('namespace', 'unknown')}")
            
            # Check if this contains the definition we're looking for
            if "arrangements with respect to property" in content.lower():
                print("   *** CONTAINS TARGET DEFINITION ***")
                print(f"   Content preview: {content[:300]}...")
            else:
                print(f"   Content preview: {content[:100].replace(chr(10), ' ')}...")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(find_definition_chunk())
