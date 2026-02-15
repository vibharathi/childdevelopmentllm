"""
Compare the two retrieval strategies side-by-side.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import load_milestone_data
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
import time


def compare_retrievers():
    """Compare both retrieval strategies."""
    print("="*70)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*70)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize both retrievers
    print("\n" + "="*70)
    print("STRATEGY 1: EMBEDDING-BASED RETRIEVAL")
    print("="*70)
    embedding_retriever = EmbeddingRetriever()
    embedding_retriever.index_documents(chunks)

    print("\n" + "="*70)
    print("STRATEGY 2: HYBRID RETRIEVAL (BM25 + Embeddings)")
    print("="*70)
    hybrid_retriever = HybridRetriever()
    hybrid_retriever.index_documents(chunks)

    # Test queries
    test_queries = [
        "When do babies start crawling?",
        "What can a 6-month-old do?",
        "When should my toddler start talking?",
        "Is it normal for my 12-month-old not to walk?",
        "What social skills develop at 18 months?"
    ]

    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)

    total_emb_time = 0
    total_hyb_time = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print("="*70)

        # Strategy 1: Embedding-based
        emb_results, emb_scores, emb_time = embedding_retriever.retrieve(query, top_k=3)
        total_emb_time += emb_time

        # Strategy 2: Hybrid
        hyb_results, hyb_scores, hyb_time = hybrid_retriever.retrieve(query, top_k=3)
        total_hyb_time += hyb_time

        print(f"\n{'EMBEDDING-BASED':<35} | {'HYBRID (BM25 + Embedding)':<35}")
        print(f"{'Time: ' + f'{emb_time*1000:.2f}ms':<35} | {'Time: ' + f'{hyb_time*1000:.2f}ms':<35}")
        print("-" * 70)

        for j in range(3):
            emb_chunk = emb_results[j]
            emb_score = emb_scores[j]
            hyb_chunk = hyb_results[j]
            hyb_score = hyb_scores[j]

            emb_age = emb_chunk['age_range'] if emb_chunk['age_range'] else 'N/A'
            hyb_age = hyb_chunk['age_range'] if hyb_chunk['age_range'] else 'N/A'

            print(f"\nRank {j+1}:")
            print(f"  {emb_chunk['source'][:30]:<30} | {hyb_chunk['source'][:30]:<30}")
            print(f"  Age: {emb_age:<27} | Age: {hyb_age:<27}")
            print(f"  Score: {emb_score:.3f}{' '*22} | Score: {hyb_score:.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nAverage Retrieval Time:")
    print(f"  Embedding-based: {(total_emb_time/len(test_queries))*1000:.2f}ms")
    print(f"  Hybrid:          {(total_hyb_time/len(test_queries))*1000:.2f}ms")
    print(f"  Speed difference: {((total_hyb_time/total_emb_time) - 1)*100:.1f}% slower for Hybrid")

    print(f"\nIndexing Time:")
    print(f"  Embedding-based: Fast (only embeddings)")
    print(f"  Hybrid:          Slower (BM25 + embeddings)")

    print(f"\nPros and Cons:")
    print(f"\n  EMBEDDING-BASED:")
    print(f"    + Faster retrieval (6-8ms)")
    print(f"    + Simpler implementation")
    print(f"    + Good semantic understanding")
    print(f"    - May miss exact keyword matches")
    print(f"    - Requires good embedding model")

    print(f"\n  HYBRID:")
    print(f"    + Catches both semantic + keyword matches")
    print(f"    + More robust to different query types")
    print(f"    + Better for rare/specific terms")
    print(f"    - Slower retrieval (28-34ms)")
    print(f"    - More complex implementation")
    print(f"    - Requires tuning weights")


if __name__ == "__main__":
    compare_retrievers()
