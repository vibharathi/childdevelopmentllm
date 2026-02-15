"""
Strategy 2: Hybrid Retrieval (BM25 + Embeddings)
Combines keyword-based BM25 with semantic embeddings for better retrieval.
"""

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import time


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (keyword) and embeddings (semantic).

    This strategy leverages both exact keyword matching and semantic similarity
    for more robust retrieval.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 bm25_weight: float = 0.5,
                 embedding_weight: float = 0.5):
        """
        Initialize the hybrid retriever.

        Args:
            model_name: Sentence-transformer model name
            bm25_weight: Weight for BM25 scores (0-1)
            embedding_weight: Weight for embedding scores (0-1)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        self.tokenized_corpus = None
        print(f"Hybrid retriever initialized (BM25: {bm25_weight}, Embedding: {embedding_weight})")

    def index_documents(self, chunks: List[Dict]):
        """
        Index documents with both BM25 and embeddings.

        Args:
            chunks: List of document chunks with text and metadata
        """
        print(f"Indexing {len(chunks)} chunks with hybrid approach...")
        start_time = time.time()

        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]

        # Create BM25 index
        print("  [1/2] Building BM25 index...")
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Create embeddings
        print("  [2/2] Creating embeddings...")
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        elapsed = time.time() - start_time
        print(f"Hybrid indexing complete! Took {elapsed:.2f}s")

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Dict], List[float], float]:
        """
        Retrieve most relevant chunks using hybrid scoring.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Tuple of (retrieved_chunks, hybrid_scores, retrieval_time)
        """
        if self.embeddings is None or self.bm25 is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        start_time = time.time()

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        if bm25_scores.max() > 0:
            bm25_scores_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_norm = bm25_scores

        # Embedding similarity scores
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        embedding_scores = self._cosine_similarity(query_embedding, self.embeddings)

        # Combine scores
        hybrid_scores = (
            self.bm25_weight * bm25_scores_norm +
            self.embedding_weight * embedding_scores
        )

        # Get top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        # Retrieve chunks and scores
        retrieved_chunks = [self.chunks[idx] for idx in top_indices]
        scores = [float(hybrid_scores[idx]) for idx in top_indices]

        retrieval_time = time.time() - start_time

        return retrieved_chunks, scores, retrieval_time

    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.

        Args:
            query_vec: Query embedding vector
            doc_vecs: Document embedding matrix

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarities = np.dot(doc_norms, query_norm)

        return similarities

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        return {
            'num_chunks': len(self.chunks),
            'embedding_dim': self.model.get_sentence_embedding_dimension(),
            'bm25_weight': self.bm25_weight,
            'embedding_weight': self.embedding_weight
        }


def test_hybrid_retriever():
    """Test the hybrid retriever."""
    from src.utils.data_loader import load_milestone_data

    print("="*60)
    print("TESTING HYBRID RETRIEVAL (BM25 + Embeddings)")
    print("="*60)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever
    retriever = HybridRetriever()

    # Index documents
    retriever.index_documents(chunks)

    # Test queries
    test_queries = [
        "When do babies start crawling?",
        "What can a 6-month-old do?",
        "When should my toddler start talking?",
        "Is it normal for my 12-month-old not to walk?"
    ]

    print("\n" + "="*60)
    print("TEST QUERIES")
    print("="*60)

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)

        results, scores, retrieval_time = retriever.retrieve(query, top_k=3)

        print(f"Retrieval time: {retrieval_time*1000:.2f}ms\n")

        for i, (chunk, score) in enumerate(zip(results, scores)):
            print(f"Result {i+1} (score: {score:.3f}):")
            print(f"  Source: {chunk['source']}")
            print(f"  Age range: {chunk['age_range']}")
            print(f"  Text: {chunk['text'][:100]}...")
            print()


if __name__ == "__main__":
    test_hybrid_retriever()
