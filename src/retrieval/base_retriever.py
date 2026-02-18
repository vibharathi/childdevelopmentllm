"""
Base retriever class consolidating shared logic across retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
from src.safety.content_filter import ContentFilter
from src.config import SafetyConfig


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    Consolidates shared functionality:
    - Safety filter initialization
    - Index-time filtering logic
    - Cosine similarity calculation
    - Common interface methods
    """

    def __init__(self, use_safety_filter: bool = SafetyConfig.ENABLE_SAFETY_FILTER):
        """
        Initialize common retriever components.

        Args:
            use_safety_filter: Whether to apply content filtering (defaults to config)
        """
        self.use_safety_filter = use_safety_filter
        self.content_filter = ContentFilter() if use_safety_filter else None
        if use_safety_filter:
            print("Safety filter: ENABLED")

    def apply_index_time_filtering(
        self,
        chunks: List[Dict],
        quality_threshold: float = SafetyConfig.QUALITY_THRESHOLD
    ) -> Tuple[List[Dict], int]:
        """
        Apply index-time filtering to remove low-quality documents.

        Consolidates 48 lines of duplicate logic from 3 retrievers.

        Documents are filtered BEFORE indexing to remove:
        - Documents with disclaimers (unreliable content)
        - Documents with implausible claims
        - Documents below quality threshold

        Args:
            chunks: List of document chunks to filter
            quality_threshold: Minimum quality score 0-1 (defaults to config)

        Returns:
            Tuple of (filtered_chunks, removed_count)
        """
        original_count = len(chunks)

        if self.use_safety_filter and self.content_filter:
            print(f"\n[Phase 1: Index-Time Filtering]")
            print(f"Pre-filtering {original_count} chunks before indexing...")

            chunks, reasons = self.content_filter.filter_before_indexing(
                chunks,
                quality_threshold=quality_threshold
            )

            removed_count = original_count - len(chunks)
            print(f"✓ Kept {len(chunks)}/{original_count} chunks ({removed_count} filtered out)")

            if reasons:
                print(f"\nFiltered documents:")
                for reason in reasons:
                    print(f"  ✗ {reason}")
            print()

            return chunks, removed_count

        return chunks, 0

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.

        Consolidates 19 lines of duplicate logic from 2 retrievers.

        Args:
            query_vec: Query embedding vector (1D array)
            doc_vecs: Document embedding matrix (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarities = np.dot(doc_norms, query_norm)

        return similarities

    @abstractmethod
    def index_documents(self, chunks: List[Dict], **kwargs):
        """
        Index documents for retrieval.

        Subclasses must implement this method.

        Args:
            chunks: List of document chunks with text and metadata
            **kwargs: Additional strategy-specific arguments
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> Tuple[List[Dict], List[float], float]:
        """
        Retrieve most relevant chunks for a query.

        Subclasses must implement this method.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Tuple of (retrieved_chunks, similarity_scores, retrieval_time)
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed documents.

        Subclasses must implement this method.

        Returns:
            Dictionary with retriever statistics
        """
        pass
