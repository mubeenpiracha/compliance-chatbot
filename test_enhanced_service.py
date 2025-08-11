#!/usr/bin/env python3
"""
Test script for the enhanced AI service with multi-namespace support
"""

import sys
import os
sys.path.append('./backend')

from backend.core.enhanced_ai_service import initialize_enhanced_ai_service, get_ai_response

def test_enhanced_service():
    """Test the enhanced service with different query types"""
    
    print("🔧 Initializing Enhanced AI Service...")
    service = initialize_enhanced_ai_service()
    
    if service is None:
        print("❌ Failed to initialize service")
        return
        
    print("✅ Service initialized successfully")
    print(f"📊 Available namespaces: {len(service.namespace_manager.get_all_namespaces())}")
    
    # Test queries
    test_queries = [
        {
            "query": "What are the AML requirements for DIFC?", 
            "jurisdiction": "DIFC",
            "description": "Specific AML query for DIFC"
        },
        {
            "query": "Tell me about capital requirements", 
            "jurisdiction": None,
            "description": "General capital requirements query"
        },
        {
            "query": "What licensing is needed for fund management in ADGM?", 
            "jurisdiction": "ADGM",
            "description": "Specific fund licensing query for ADGM"
        },
        {
            "query": "Hello, how can you help me?", 
            "jurisdiction": None,
            "description": "Greeting query"
        }
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED AI SERVICE")
    print("="*60)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {test['description']}")
        print(f"   Query: '{test['query']}'")
        print(f"   Jurisdiction: {test['jurisdiction']}")
        print("-" * 40)
        
        try:
            # Test the get_ai_response function (full pipeline)
            response = get_ai_response(
                user_message=test['query'],
                history=[],
                jurisdiction=test['jurisdiction'] or "DIFC"
            )
            
            print(f"✅ Response received:")
            print(f"   Answer length: {len(response.get('answer', ''))}")
            print(f"   Sources found: {len(response.get('sources', []))}")
            
            if response.get('sources'):
                print(f"   First source namespace: {response['sources'][0].get('metadata', {}).get('namespace', 'Unknown')}")
            
            # Show first 200 chars of answer
            answer = response.get('answer', '')
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"   Answer preview: {preview}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print()

if __name__ == "__main__":
    test_enhanced_service()
