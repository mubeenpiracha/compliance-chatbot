"""
Test to debug why the fund definition isn't being retrieved properly.
"""
import asyncio
from backend.core.real_vector_service import RealVectorService
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def debug_search():
    """Debug the vector search to see what's happening with fund definitions."""
    
    print("Debugging Vector Search for Fund Definition")
    print("=" * 50)
    
    try:
        # Initialize services
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        vector_service = RealVectorService()
        
        # Test different query variations
        test_queries = [
            "What is a fund in ADGM?",
            "fund definition",
            "collective investment fund definition", 
            "Fund means a Collective Investment Fund",
            "arrangements with respect to property"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            print("-" * 40)
            
            # Generate embedding
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=query,
                dimensions=1536
            )
            query_embedding = response.data[0].embedding
            
            # Search with different thresholds
            for threshold in [0.0, 0.1, 0.3, 0.5]:
                print(f"  Threshold {threshold}:")
                results = await vector_service.search(
                    query_vector=query_embedding,
                    top_k=5,
                    filter_params=None
                )
                
                filtered_results = [r for r in results if r['score'] >= threshold]
                print(f"    Found {len(filtered_results)} results")
                
                for i, result in enumerate(filtered_results[:2]):  # Show top 2
                    title = result['metadata']['title']
                    score = result['score']
                    content_preview = result['content'][:100].replace('\n', ' ')
                    print(f"    {i+1}. {title} (score: {score:.3f})")
                    print(f"       Preview: {content_preview}...")
                    
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(debug_search())
