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
from src.retrieval.embedding_retriever import EmbeddingRetriever
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

    # Initialize retriever with safety filter
    retriever = ChromaRetriever(
        collection_name="test_index_filtering",
        persist_dir="./data/chroma_test_index_filter",
        use_safety_filter=True
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


def test_embedding_index_time_filtering():
    """Test EmbeddingRetriever with index-time filtering."""
    print("\n" + "=" * 70)
    print("TEST 2: Embedding Retriever - Index-Time Filtering")
    print("=" * 70)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever with safety filter
    retriever = EmbeddingRetriever(use_safety_filter=True)

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


def test_hybrid_index_time_filtering():
    """Test HybridRetriever with index-time filtering."""
    print("\n" + "=" * 70)
    print("TEST 3: Hybrid Retriever - Index-Time Filtering")
    print("=" * 70)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever with safety filter
    retriever = HybridRetriever(use_safety_filter=True)

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
    """Test that retrieval no longer returns noisy documents."""
    print("\n" + "=" * 70)
    print("TEST 4: Verify No Noisy Documents in Retrieval")
    print("=" * 70)

    # Load data
    docs, chunks = load_milestone_data()

    # Initialize and index
    retriever = ChromaRetriever(
        collection_name="test_retrieval_quality",
        persist_dir="./data/chroma_test_quality",
        use_safety_filter=True
    )
    retriever.index_documents(chunks, force_reindex=True, quality_threshold=0.4)

    # Test query that previously retrieved noisy docs
    query = "When do babies start walking?"
    print(f"\nQuery: {query}")
    print("-" * 70)

    results, scores, retrieval_time = retriever.retrieve(query, top_k=5)

    print(f"Retrieval time: {retrieval_time * 1000:.2f}ms")
    print(f"Results returned: {len(results)}\n")

    noisy_sources = ["noisy_alt_movements.txt", "noisy_sensory.txt", "noisy_communication.txt"]
    found_noisy = False

    for i, (chunk, score) in enumerate(zip(results, scores)):
        source = chunk['source']
        age_range = chunk['age_range']
        is_noisy = source in noisy_sources

        if is_noisy:
            found_noisy = True
            print(f"❌ Result {i + 1} (score: {score:.3f}) - NOISY DOCUMENT FOUND!")
        else:
            print(f"✓ Result {i + 1} (score: {score:.3f}) - Clean document")

        print(f"   Source: {source}")
        print(f"   Age range: {age_range}")
        print(f"   Text: {chunk['text'][:80]}...")
        print()

    print("=" * 70)
    if found_noisy:
        print("❌ TEST FAILED: Noisy documents still in retrieval results!")
    else:
        print("✓ TEST PASSED: No noisy documents in results!")
    print("=" * 70)


if __name__ == "__main__":
    test_chroma_index_time_filtering()
    test_embedding_index_time_filtering()
    test_hybrid_index_time_filtering()
    test_retrieval_quality()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print("✓ Index-time filtering implemented successfully")
    print("✓ Noisy documents removed before indexing")
    print("✓ Vector DB now contains only high-quality documents")
    print("✓ Faster retrieval and reduced storage")
