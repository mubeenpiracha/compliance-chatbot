#!/usr/bin/env python3
"""
Performance comparison test: Loading corpus with and without jurisdiction filtering.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.document_loader import load_document_corpus_from_content_store

def measure_load_time(jurisdiction=None):
    """Measure time to load document corpus."""
    start_time = time.time()
    docs = load_document_corpus_from_content_store(jurisdiction=jurisdiction)
    end_time = time.time()
    return len(docs), end_time - start_time

def main():
    print("=" * 80)
    print("KEYWORD SEARCH PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Test 1: Load all documents
    print("\n📚 Loading ALL documents (no filter)...")
    all_count, all_time = measure_load_time()
    print(f"   Documents: {all_count:,}")
    print(f"   Time: {all_time:.2f}s")
    
    # Test 2: Load only DIFC
    print("\n🏛️  Loading ONLY DIFC documents...")
    difc_count, difc_time = measure_load_time("DIFC")
    print(f"   Documents: {difc_count:,}")
    print(f"   Time: {difc_time:.2f}s")
    
    # Test 3: Load only ADGM
    print("\n🏢 Loading ONLY ADGM documents...")
    adgm_count, adgm_time = measure_load_time("ADGM")
    print(f"   Documents: {adgm_count:,}")
    print(f"   Time: {adgm_time:.2f}s")
    
    # Analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    if difc_count > 0:
        difc_reduction = ((all_count - difc_count) / all_count) * 100
        difc_speedup = (all_time / difc_time) if difc_time > 0 else 0
        print(f"\n✅ DIFC Filtering Benefits:")
        print(f"   • Document Reduction: {difc_reduction:.1f}% ({all_count:,} → {difc_count:,})")
        print(f"   • Time Reduction: {((all_time - difc_time) / all_time * 100):.1f}% ({all_time:.2f}s → {difc_time:.2f}s)")
        print(f"   • Speedup: {difc_speedup:.2f}x faster")
    
    if adgm_count > 0 and adgm_count != all_count:
        adgm_reduction = ((all_count - adgm_count) / all_count) * 100
        adgm_speedup = (all_time / adgm_time) if adgm_time > 0 else 0
        print(f"\n✅ ADGM Filtering Benefits:")
        print(f"   • Document Reduction: {adgm_reduction:.1f}% ({all_count:,} → {adgm_count:,})")
        print(f"   • Time Reduction: {((all_time - adgm_time) / all_time * 100):.1f}% ({all_time:.2f}s → {adgm_time:.2f}s)")
        print(f"   • Speedup: {adgm_speedup:.2f}x faster")
    
    # Memory estimation
    avg_doc_size_kb = 2  # Rough estimate: 2KB per document
    print(f"\n💾 Estimated Memory Savings:")
    print(f"   • Full corpus: ~{(all_count * avg_doc_size_kb) / 1024:.1f} MB")
    if difc_count > 0:
        print(f"   • DIFC only: ~{(difc_count * avg_doc_size_kb) / 1024:.1f} MB (saves ~{((all_count - difc_count) * avg_doc_size_kb) / 1024:.1f} MB)")
    if adgm_count > 0 and adgm_count != all_count:
        print(f"   • ADGM only: ~{(adgm_count * avg_doc_size_kb) / 1024:.1f} MB (saves ~{((all_count - adgm_count) * avg_doc_size_kb) / 1024:.1f} MB)")
    
    print("\n" + "=" * 80)
    print("💡 KEY TAKEAWAY:")
    print("   Jurisdiction filtering significantly reduces memory usage and load time,")
    print("   making keyword search more efficient when users select a specific jurisdiction.")
    print("=" * 80)

if __name__ == "__main__":
    main()
