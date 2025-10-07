#!/usr/bin/env python3
"""
Test to verify RRF (Reciprocal Rank Fusion) is working correctly.
"""
import asyncio
import requests
import json

def test_rrf_chat_endpoint():
    """Test the chat endpoint to verify RRF fusion is working."""
    url = "http://localhost:8000/api/v1/chat/"
    
    payload = {
        "message": "What is the DIFC legal definition of a collective investment fund according to Article 11?",
        "jurisdiction": "DIFC",
        "history": []
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing RRF Implementation via Chat Endpoint...")
        print("=" * 70)
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response received.")
            print(f"Response keys: {list(result.keys())}")
            print(f"Response preview: {str(result)[:500]}...")
            
            # Analyze the sources to verify RRF fusion
            sources = result.get('sources', [])
            
            print(f"\nAnalyzing {len(sources)} sources for RRF verification:")
            print("-" * 70)
            
            retrieval_methods = {}
            fusion_results = 0
            
            for i, source in enumerate(sources, 1):
                method = source.get('retrieval_method', 'unknown')
                score = source.get('score', 0)
                title = source.get('title', 'Unknown')
                
                # Count methods
                if method not in retrieval_methods:
                    retrieval_methods[method] = 0
                retrieval_methods[method] += 1
                
                # Count fusion results
                if method == 'fusion':
                    fusion_results += 1
                
                print(f"{i:2d}. Method: {method:8s} | Score: {score:.6f} | {title}")
                
                # Check for RRF metadata
                if 'fusion_methods' in source:
                    print(f"    -> Fusion of: {source['fusion_methods']}")
                if 'vector_rank' in source:
                    print(f"    -> Vector rank: {source['vector_rank']}")
                if 'keyword_rank' in source:
                    print(f"    -> Keyword rank: {source['keyword_rank']}")
            
            print("\n" + "=" * 70)
            print("RRF Analysis Summary:")
            print(f"• Retrieval methods used: {list(retrieval_methods.keys())}")
            print(f"• Method distribution: {retrieval_methods}")
            print(f"• Fusion results: {fusion_results}/{len(sources)}")
            
            # Verify RRF characteristics
            if 'fusion' in retrieval_methods:
                print("✅ RRF fusion is working - found 'fusion' method results")
            elif set(retrieval_methods.keys()) == {'vector', 'keyword'}:
                print("✅ Both vector and keyword methods working")
            else:
                print("❌ Limited method diversity")
            
            # Check if scores look like RRF scores (typically 0.01-0.1 range)
            rrf_like_scores = [s.get('score', 0) for s in sources if 0.01 <= s.get('score', 0) <= 0.1]
            if len(rrf_like_scores) > len(sources) * 0.5:  # More than half have RRF-like scores
                print(f"✅ Score range suggests RRF calculation ({len(rrf_like_scores)}/{len(sources)} in RRF range)")
            else:
                print(f"⚠️  Score range may not be RRF ({len(rrf_like_scores)}/{len(sources)} in expected range)")
        
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_rrf_chat_endpoint()