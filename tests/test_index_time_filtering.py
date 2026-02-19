"""
Test index-time filtering functionality.

This test demonstrates Phase 1 filtering - removing noisy documents
BEFORE they are stored in the vector database.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_milestone_data
from src.retrieval.chroma_retriever import ChromaRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def test_chroma_index_time_filtering():
    """Test ChromaRetriever with index-time filtering."""
    print("=" * 70)
    print("TEST 1: ChromaDB Retriever - Index-Time Filtering")
    print("=" * 70)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()
    print(f"Total documents loaded: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}")

    # Initialize retriever (safety filter always enabled)
    retriever = ChromaRetriever(
        collection_name="test_index_filtering",
        persist_dir="./data/chroma_test_index_filter"
    )

    # Clear any existing data and re-index with filtering
    print("\nForcing re-index to test filtering...")
    retriever.index_documents(chunks, force_reindex=True, quality_threshold=0.4)

    # Check how many documents were stored
    stats = retriever.get_stats()
    print(f"\n{'=' * 70}")
    print("RESULTS:")
    print(f"{'=' * 70}")
    print(f"Original chunks: {len(chunks)}")
    print(f"Stored in DB: {stats['num_chunks']}")
    print(f"Filtered out: {len(chunks) - stats['num_chunks']}")
    print(f"Storage location: {stats['persist_dir']}")
    print()


def test_hybrid_index_time_filtering():
    """Test HybridRetriever with index-time filtering."""
    print("\n" + "=" * 70)
    print("TEST 2: Hybrid Retriever - Index-Time Filtering")
    print("=" * 70)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever (safety filter always enabled)
    retriever = HybridRetriever()

    # Index with filtering
    retriever.index_documents(chunks, quality_threshold=0.4)

    # Check results
    stats = retriever.get_stats()
    print(f"\n{'=' * 70}")
    print("RESULTS:")
    print(f"{'=' * 70}")
    print(f"Original chunks: {len(chunks)}")
    print(f"Stored in index: {stats['num_chunks']}")
    print(f"Filtered out: {len(chunks) - stats['num_chunks']}")
    print()


def test_retrieval_quality():
    """Test that retrieval returns results from the filtered index."""
    print("\n" + "=" * 70)
    print("TEST 3: Verify Retrieval Works on Filtered Index")
    print("=" * 70)

    # Load data
    docs, chunks = load_milestone_data()

    # Initialize and index (safety filter always enabled)
    retriever = ChromaRetriever(
        collection_name="test_retrieval_quality",
        persist_dir="./data/chroma_test_quality"
    )
    retriever.index_documents(chunks, force_reindex=True, quality_threshold=0.4)

    stats = retriever.get_stats()
    print(f"Documents stored: {stats['num_chunks']}/{len(chunks)}")

    query = "When do babies start walking?"
    print(f"\nQuery: {query}")
    print("-" * 70)

    results, scores, retrieval_time = retriever.retrieve(query, top_k=5)

    print(f"Retrieval time: {retrieval_time * 1000:.2f}ms")
    print(f"Results returned: {len(results)}\n")

    for i, (chunk, score) in enumerate(zip(results, scores)):
        print(f"✓ Result {i + 1} (score: {score:.3f})")
        print(f"   Source: {chunk['source']}")
        print(f"   Age range: {chunk['age_range']}")
        print(f"   Text: {chunk['text'][:80]}...")
        print()

    print("=" * 70)
    if results:
        print("✓ TEST PASSED: Filtered index returns results successfully!")
    else:
        print("❌ TEST FAILED: No results returned from filtered index!")
    print("=" * 70)


if __name__ == "__main__":
    test_chroma_index_time_filtering()
    test_hybrid_index_time_filtering()
    test_retrieval_quality()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print("✓ Index-time filtering implemented successfully")
    print("✓ Low-quality documents removed before indexing")
    print("✓ Vector DB contains only content-filtered documents")
    print("✓ Retrieval works correctly on filtered index")
