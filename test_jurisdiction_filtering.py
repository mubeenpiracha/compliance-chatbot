#!/usr/bin/env python3
"""
Test script to verify jurisdiction filtering in keyword search.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.document_loader import load_document_corpus_from_content_store

def test_jurisdiction_filtering():
    """Test that jurisdiction filtering works correctly."""
    
    print("=" * 80)
    print("TESTING JURISDICTION FILTERING IN KEYWORD SEARCH")
    print("=" * 80)
    
    # Test 1: Load all documents (no filter)
    print("\n1. Loading ALL documents (no jurisdiction filter)...")
    all_docs = load_document_corpus_from_content_store()
    print(f"   Total documents loaded: {len(all_docs)}")
    
    # Count by jurisdiction
    difc_count = sum(1 for doc in all_docs if doc['metadata']['jurisdiction'].upper() == 'DIFC')
    adgm_count = sum(1 for doc in all_docs if doc['metadata']['jurisdiction'].upper() == 'ADGM')
    print(f"   - DIFC documents: {difc_count}")
    print(f"   - ADGM documents: {adgm_count}")
    
    # Test 2: Load only DIFC documents
    print("\n2. Loading ONLY DIFC documents (jurisdiction='DIFC')...")
    difc_docs = load_document_corpus_from_content_store(jurisdiction="DIFC")
    print(f"   Total documents loaded: {len(difc_docs)}")
    
    # Verify all are DIFC
    non_difc = [doc for doc in difc_docs if doc['metadata']['jurisdiction'].upper() != 'DIFC']
    if non_difc:
        print(f"   ❌ ERROR: Found {len(non_difc)} non-DIFC documents in filtered results!")
        for doc in non_difc[:3]:
            print(f"      - {doc['metadata']['title']} (jurisdiction: {doc['metadata']['jurisdiction']})")
    else:
        print(f"   ✅ All documents are from DIFC jurisdiction")
    
    # Test 3: Load only ADGM documents
    print("\n3. Loading ONLY ADGM documents (jurisdiction='ADGM')...")
    adgm_docs = load_document_corpus_from_content_store(jurisdiction="ADGM")
    print(f"   Total documents loaded: {len(adgm_docs)}")
    
    # Verify all are ADGM
    non_adgm = [doc for doc in adgm_docs if doc['metadata']['jurisdiction'].upper() != 'ADGM']
    if non_adgm:
        print(f"   ❌ ERROR: Found {len(non_adgm)} non-ADGM documents in filtered results!")
        for doc in non_adgm[:3]:
            print(f"      - {doc['metadata']['title']} (jurisdiction: {doc['metadata']['jurisdiction']})")
    else:
        print(f"   ✅ All documents are from ADGM jurisdiction")
    
    # Test 4: Verify filtering efficiency
    print("\n4. Filtering Efficiency Analysis:")
    expected_difc = len(difc_docs)
    expected_adgm = len(adgm_docs)
    actual_total = len(all_docs)
    
    print(f"   - Expected total (DIFC + ADGM): {expected_difc + expected_adgm}")
    print(f"   - Actual total (unfiltered): {actual_total}")
    
    if abs(expected_difc + expected_adgm - actual_total) < 10:  # Allow small variance
        print(f"   ✅ Document counts match (within tolerance)")
    else:
        print(f"   ⚠️  Document counts don't match - may have other jurisdictions or errors")
    
    # Test 5: Memory savings estimate
    print("\n5. Memory Savings Estimate:")
    if actual_total > 0:
        difc_percent = (len(difc_docs) / actual_total) * 100
        adgm_percent = (len(adgm_docs) / actual_total) * 100
        
        print(f"   - DIFC filtering saves: ~{100 - difc_percent:.1f}% of documents")
        print(f"   - ADGM filtering saves: ~{100 - adgm_percent:.1f}% of documents")
    
    # Test 6: Case insensitivity
    print("\n6. Testing case-insensitive filtering...")
    difc_lower = load_document_corpus_from_content_store(jurisdiction="difc")
    difc_upper = load_document_corpus_from_content_store(jurisdiction="DIFC")
    difc_mixed = load_document_corpus_from_content_store(jurisdiction="Difc")
    
    if len(difc_lower) == len(difc_upper) == len(difc_mixed):
        print(f"   ✅ Case-insensitive filtering works correctly")
        print(f"      All variants returned {len(difc_lower)} documents")
    else:
        print(f"   ❌ Case-insensitive filtering has issues:")
        print(f"      'difc': {len(difc_lower)}, 'DIFC': {len(difc_upper)}, 'Difc': {len(difc_mixed)}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\n📊 SUMMARY:")
    if len(difc_docs) > 0 and len(adgm_docs) > 0 and not non_difc and not non_adgm:
        print("✅ Jurisdiction filtering is working correctly!")
        print(f"   - Successfully filters to {len(difc_docs)} DIFC documents")
        print(f"   - Successfully filters to {len(adgm_docs)} ADGM documents")
        print(f"   - Saves significant memory and processing time")
    else:
        print("⚠️  There may be issues with jurisdiction filtering")
        print("   Please review the test results above")


if __name__ == "__main__":
    test_jurisdiction_filtering()
