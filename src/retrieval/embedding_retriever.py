"""
Strategy 1: Dense Embedding-Based Retrieval
Uses sentence-transformers to create semantic embeddings for retrieval.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import time
from src.retrieval.base_retriever import BaseRetriever
from src.config import SafetyConfig


class EmbeddingRetriever(BaseRetriever):
    """
    Dense embedding-based retrieval using sentence-transformers.

    This strategy creates vector embeddings of documents and queries,
    then retrieves based on cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding retriever.

        Args:
            model_name: Sentence-transformer model name
        """
        # Initialize base retriever (safety filter always enabled)
        super().__init__()

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        print(f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def index_documents(self, chunks: List[Dict], quality_threshold: float = SafetyConfig.QUALITY_THRESHOLD):
        """
        Create embeddings for document chunks with index-time filtering.

        Documents are filtered BEFORE indexing to remove:
        - Documents with disclaimers (unreliable content)
        - Documents with implausible claims
        - Documents below quality threshold

        Args:
            chunks: List of document chunks with text and metadata
            quality_threshold: Minimum quality score (0-1) for documents (defaults to config)
        """
        # Apply index-time filtering (from base class)
        chunks, removed_count = self.apply_index_time_filtering(chunks, quality_threshold)

        print(f"Creating embeddings for {len(chunks)} clean chunks...")
        start_time = time.time()

        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]

        # Create embeddings
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        elapsed = time.time() - start_time
        print(f"Indexing complete! Took {elapsed:.2f}s")

        if removed_count > 0:
            print(f"✓ Index contains only high-quality documents (filtered {removed_count} at index-time)")

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Dict], List[float], float]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Tuple of (retrieved_chunks, similarity_scores, retrieval_time)
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        start_time = time.time()

        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Calculate cosine similarity (from base class)
        similarities = self.cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Retrieve chunks and scores
        retrieved_chunks = [self.chunks[idx] for idx in top_indices]
        scores = [float(similarities[idx]) for idx in top_indices]

        retrieval_time = time.time() - start_time

        return retrieved_chunks, scores, retrieval_time

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        return {
            'num_chunks': len(self.chunks),
            'embedding_dim': self.model.get_sentence_embedding_dimension(),
            'model_name': self.model._model_card_data.model_name if hasattr(self.model, '_model_card_data') else 'unknown'
        }


def test_embedding_retriever():
    """Test the embedding retriever."""
    from src.utils.data_loader import load_milestone_data

    print("="*60)
    print("TESTING EMBEDDING-BASED RETRIEVAL")
    print("="*60)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever
    retriever = EmbeddingRetriever()

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
    test_embedding_retriever()
