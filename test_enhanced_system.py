"""
Test script for the enhanced AI service with the new agent-based architecture.
"""
import asyncio
import sys
import os
from typing import Dict, Any, List

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.core.enhanced_ai_service import EnhancedAIService
from backend.core.config import OPENAI_API_KEY


async def test_enhanced_service_with_real_services():
    """Test the enhanced AI service with real services."""
    
    print("Testing Enhanced AI Service with REAL Pinecone Index & Content Store")
    print("=" * 70)
    
    try:
        # Initialize vector service (now using real Pinecone)
        from backend.core.real_vector_service import RealVectorService
        from backend.core.document_loader import load_document_corpus_from_content_store
        
        vector_service = RealVectorService()
        document_corpus = load_document_corpus_from_content_store()
        
        # Create service with real API key and real infrastructure
        service = EnhancedAIService(
            openai_api_key=OPENAI_API_KEY,
            vector_service=vector_service,
            document_corpus=document_corpus
        )
        
        # Test query
        test_query = "what is a fund"
        
        print(f"Query: {test_query}")
        print()
        print("Processing...")
        
        result = await service.process_query(test_query)
        
        print("=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Response: {result['response']}")
        print()
        print(f"Confidence: {result['confidence']}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        print()
        
        if result.get('requires_clarification'):
            print("CLARIFICATION QUESTIONS:")
            print("-" * 30)
            for i, cq in enumerate(result['clarification_questions'], 1):
                print(f"{i}. {cq}")
            print()
        
        if result.get('reasoning'):
            print("REASONING STEPS:")
            print("-" * 30)
            for i, step in enumerate(result['reasoning'], 1):
                print(f"{i}. {step}")
            print()
        
        if result.get('sources'):
            print("SOURCES:")
            print("-" * 30)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} ({source.get('document_type', 'N/A')})")
                print(f"   Relevance: {source.get('relevance_score', 0):.2f}")
                if source.get('highlights'):
                    print(f"   Highlights: {source['highlights'][:2]}")
                print()
        
        return True
        
    except Exception as e:
        print(f"Error testing enhanced service: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Enhanced AI Service Test Suite")
    print("=" * 60)
    
    # Test the service with real infrastructure
    asyncio.run(test_enhanced_service_with_real_services())
