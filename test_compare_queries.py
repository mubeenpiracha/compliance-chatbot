"""
Compare search results between specific definition query and general fund query.
"""
import asyncio
from backend.core.real_vector_service import RealVectorService
from backend.core.config import OPENAI_API_KEY
from openai import AsyncOpenAI

async def compare_queries():
    """Compare search results for different queries."""
    
    print("Comparing Fund Definition Search Results")
    print("=" * 50)
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        vector_service = RealVectorService()
        
        queries = [
            "What is a fund in ADGM?",
            "fund definition",
            "arrangements with respect to property"
        ]
        
        for query in queries:
            print(f"\n{'='*20} QUERY: '{query}' {'='*20}")
            
            # Generate embedding
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=query,
                dimensions=1536
            )
            query_vector = response.data[0].embedding
            
            results = await vector_service.search(
                query_vector=query_vector,
                top_k=5
            )
            
            print(f"Top {len(results)} results:")
            for i, result in enumerate(results, 1):
                title = result['metadata']['title']
                score = result['score']
                content = result['content'][:100].replace('\n', ' ')
                
                contains_def = "arrangements with respect to property" in result['content'].lower()
                marker = " *** TARGET DEF ***" if contains_def else ""
                
                print(f"{i}. {title} (score: {score:.3f}){marker}")
                print(f"   {content}...")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(compare_queries())
