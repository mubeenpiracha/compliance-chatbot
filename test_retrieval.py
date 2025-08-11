"""
Test the integrated system with a simple query that should retrieve documents.
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.core.enhanced_ai_service import EnhancedAIService
from backend.core.real_vector_service import RealVectorService
from backend.core.document_loader import load_document_corpus_from_content_store
from backend.core.config import OPENAI_API_KEY


async def test_simple_query():
    """Test with a simple definition query that should retrieve documents."""
    
    print("Testing Integrated System with Document Retrieval")
    print("=" * 55)
    
    try:
        # Initialize real services
        vector_service = RealVectorService()
        document_corpus = load_document_corpus_from_content_store()
        
        service = EnhancedAIService(
            openai_api_key=OPENAI_API_KEY,
            vector_service=vector_service,
            document_corpus=document_corpus
        )
        
        # Simple definition query that should retrieve documents
        test_query = "What is a fund in ADGM?"
        
        print(f"Query: {test_query}")
        print()
        print("Processing...")
        
        result = await service.process_query(test_query)
        
        print("=" * 55)
        print("RESULT:")
        print("=" * 55)
        print(f"Response: {result['response']}")
        print()
        print(f"Confidence: {result['confidence']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print()
        
        if result.get('reasoning'):
            print("REASONING STEPS:")
            print("-" * 30)
            for i, step in enumerate(result['reasoning'], 1):
                print(f"{i}. {step}")
            print()
        
        if result.get('sources'):
            print("SOURCES (from real Pinecone index):")
            print("-" * 40)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} ({source['document_type']})")
                print(f"   Relevance: {source['relevance_score']:.2f}")
                if source.get('highlights'):
                    print(f"   Highlights: {source['highlights'][:1]}")
                print()
        else:
            print("No sources returned - this indicates the query needed clarification rather than retrieval")
        
        return True
        
    except Exception as e:
        print(f"Error testing integrated system: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Integrated AI Service Test - Document Retrieval")
    print("=" * 55)
    
    asyncio.run(test_simple_query())
