"""
Test script for the enhanced AI service with chain-of-thought reasoning.
"""
import asyncio
import sys
import os
from typing import Dict, Any, List

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.core.enhanced_ai_service import EnhancedAIService
from backend.core.vector_service import VectorService
from backend.core.config import OPENAI_API_KEY, PINECONE_API_KEY


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
        
        # Test query from your example
        test_query = "I am looking to start an SPV but I do not want the definition of it being a fund because its a simple syndicate with only 2 key investment decisions"
        
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
        print(f"Processing time: {result['processing_time']:.2f}s")
        print()
        
        if result.get('requires_clarification'):
            print("CLARIFICATION QUESTIONS:")
            print("-" * 30)
            for i, cq in enumerate(result['clarification_questions'], 1):
                print(f"{i}. {cq['question']}")
                print(f"   Context: {cq['context']}")
                if cq.get('suggested_answers'):
                    print(f"   Suggested answers: {', '.join(cq['suggested_answers'])}")
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
                print(f"{i}. {source['title']} ({source['document_type']})")
                print(f"   Relevance: {source['relevance_score']:.2f}")
                if source.get('highlights'):
                    print(f"   Highlights: {source['highlights'][:2]}")
                print()
        
        return True
        
    except Exception as e:
        print(f"Error testing enhanced service: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_architecture():
    """Test the overall system architecture."""
    
    print("Testing System Architecture")
    print("=" * 50)
    
    try:
        # Test all model imports
        print("Testing model imports...")
        from backend.core.models.query_models import (
            QueryClassification, QueryComplexity, RegulatoryContext,
            SubQuestion, QueryDecomposition, ClarificationQuestion,
            KnowledgeGap, ProcessingState
        )
        print("✓ Query models")
        
        from backend.core.models.retrieval_models import (
            DocumentType, RetrievalQuery, RetrievedDocument, 
            HybridRetrievalResult, EntityDefinition
        )
        print("✓ Retrieval models")
        
        from backend.core.models.analysis_models import (
            Definition, ComplianceAssessment, RiskArea,
            StructuredResponse, QualityAssessment, FinalResponse
        )
        print("✓ Analysis models")
        
        # Test node imports
        print("\nTesting node imports...")
        from backend.core.nodes.base_node import BaseNode, ConditionalNode
        print("✓ Base nodes")
        
        from backend.core.nodes.classification import ComplianceClassificationNode
        print("✓ Classification node")
        
        from backend.core.nodes.context_identification import RegulatoryContextNode
        print("✓ Context identification node")
        
        from backend.core.nodes.query_decomposition import QueryDecompositionNode
        print("✓ Query decomposition node")
        
        from backend.core.nodes.knowledge_gap_analysis import KnowledgeGapIdentificationNode
        print("✓ Knowledge gap analysis node")
        
        from backend.core.nodes.hybrid_retrieval import HybridRetrievalNode
        print("✓ Hybrid retrieval node")
        
        # Test retrieval components
        print("\nTesting retrieval components...")
        from backend.core.retrieval.vector_search import VectorSearchEngine
        print("✓ Vector search engine")
        
        from backend.core.retrieval.keyword_search import KeywordSearchEngine
        print("✓ Keyword search engine")
        
        from backend.core.retrieval.result_fusion import ResultFusion
        print("✓ Result fusion")
        
        print("\n🎉 All components successfully imported!")
        print("The enhanced architecture is ready for testing with real data.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("Enhanced AI Service Test Suite")
    print("=" * 60)
    
    # Test architecture first
    architecture_ok = asyncio.run(test_system_architecture())
    
    if architecture_ok:
        print("\n" + "=" * 60)
        # Test the service with real infrastructure
        asyncio.run(test_enhanced_service_with_real_services())
    else:
        print("❌ Architecture test failed. Please check imports.")
