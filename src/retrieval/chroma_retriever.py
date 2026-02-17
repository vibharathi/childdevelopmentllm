"""
ChromaDB-based Retrieval with Local Persistence
Uses ChromaDB for automatic vector storage and retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional
import time
import re
from src.safety.content_filter import ContentFilter


def parse_age_range(age_range_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse age range string into min and max ages.

    Args:
        age_range_str: Age range like "2-4", "0-2", "12-18", etc.

    Returns:
        Tuple of (min_age, max_age) or (None, None) if parsing fails

    Examples:
        "2-4" -> (2, 4)
        "0-2" -> (0, 2)
        "12-18" -> (12, 18)
    """
    if not age_range_str:
        return None, None

    # Match patterns like "2-4" or "0-2"
    match = re.match(r'^(\d+)-(\d+)$', age_range_str.strip())
    if match:
        min_age = int(match.group(1))
        max_age = int(match.group(2))
        return min_age, max_age

    return None, None


def extract_age_from_query(query: str) -> Optional[int]:
    """
    Extract age in months from a user query.

    Args:
        query: User's question

    Returns:
        Age in months, or None if no age found

    Examples:
        "what do 2 month olds do" -> 2
        "6 month old baby" -> 6
        "what can a 12-month-old do" -> 12
        "newborn abilities" -> None
    """
    query_lower = query.lower()

    # Pattern 1: "N month" or "N-month"
    match = re.search(r'(\d+)[-\s]month', query_lower)
    if match:
        return int(match.group(1))

    # Pattern 2: "N months"
    match = re.search(r'(\d+)\s+months', query_lower)
    if match:
        return int(match.group(1))

    return None


class ChromaRetriever:
    """
    Dense embedding retrieval using ChromaDB with local persistence.

    This strategy uses ChromaDB to automatically handle embeddings and
    persistent storage. All data is stored locally - no cloud required.
    """

    def __init__(
        self,
        collection_name: str = "child_development",
        persist_dir: str = "./data/chroma_db",
        use_safety_filter: bool = True
    ):
        """
        Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the collection
            persist_dir: Local directory for persistent storage
            use_safety_filter: Whether to apply content filtering
        """
        print(f"Initializing ChromaDB (local storage: {persist_dir})...")

        # Create client with local persistence
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.use_safety_filter = use_safety_filter
        self.content_filter = ContentFilter() if use_safety_filter else None

        print(f"✓ Collection '{collection_name}' ready")
        print(f"✓ Current documents: {self.collection.count()}")
        if use_safety_filter:
            print("Safety filter: ENABLED")

    def index_documents(self, chunks: List[Dict], force_reindex: bool = False, quality_threshold: float = 0.4):
        """
        Index documents into ChromaDB with index-time filtering.

        Documents are filtered BEFORE indexing to remove:
        - Documents with disclaimers (unreliable content)
        - Documents with implausible claims
        - Documents below quality threshold

        Args:
            chunks: List of document chunks with text and metadata
            force_reindex: If True, clear and re-index all documents
            quality_threshold: Minimum quality score (0-1) for documents
        """
        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"✓ Collection already indexed with {self.collection.count()} chunks")
            print("  (Skipping re-indexing. Use force_reindex=True to re-index)")
            return

        if force_reindex:
            print("Clearing existing index...")
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )

        # PHASE 1: Index-Time Filtering (remove known bad documents)
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

        print(f"Indexing {len(chunks)} clean chunks into ChromaDB...")
        start_time = time.time()

        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]

        # ChromaDB metadata requirements:
        # - Values must be str, int, float, or bool (not None)
        # - Store source, age_range, and chunk_id for retrieval context
        metadatas = [
            {
                'source': chunk.get('source', ''),
                'age_range': chunk.get('age_range') or '',  # Convert None to empty string
                'chunk_id': chunk.get('chunk_id', -1),  # -1 indicates unknown/missing
            }
            for chunk in chunks
        ]

        # Add to ChromaDB (auto-embeds and persists to disk)
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        elapsed = time.time() - start_time
        print(f"✓ Indexing complete! Took {elapsed:.2f}s")
        print(f"✓ Persisted to disk: {self.collection.count()} chunks saved")

        if self.use_safety_filter:
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
        if self.collection.count() == 0:
            raise ValueError("No documents indexed. Call index_documents() first.")

        start_time = time.time()

        # Query ChromaDB (uses cosine similarity)
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Parse results
        retrieved_chunks = []
        scores = []

        if results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                # Convert distance to similarity score
                # For cosine distance: similarity = 1 - distance
                similarity = 1 - distance

                retrieved_chunks.append({
                    'text': doc,
                    'source': metadata.get('source', ''),
                    'age_range': metadata.get('age_range', ''),
                    'chunk_id': metadata.get('chunk_id', -1)  # -1 indicates unknown
                })
                scores.append(similarity)

        retrieval_time = time.time() - start_time

        return retrieved_chunks, scores, retrieval_time

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        return {
            'num_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'persist_dir': self.persist_dir,
            'storage': 'local (persistent)',
            'embedding_model': 'default (all-MiniLM-L6-v2)'
        }

    def clear_index(self):
        """Clear all documents from the collection."""
        print(f"Clearing collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Index cleared")


def test_chroma_retriever():
    """Test the ChromaDB retriever."""
    from src.utils.data_loader import load_milestone_data

    print("="*60)
    print("TESTING CHROMADB-BASED RETRIEVAL")
    print("="*60)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize retriever
    retriever = ChromaRetriever(
        collection_name="test_child_development",
        persist_dir="./data/chroma_db_test"
    )

    # Index documents (will skip if already indexed)
    retriever.index_documents(chunks)

    # Show stats
    stats = retriever.get_stats()
    print(f"\nCollection stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

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

    print("\n" + "="*60)
    print("TEST PERSISTENCE")
    print("="*60)
    print("\nCreating new retriever instance to test persistence...")

    # Create new instance - should load from disk
    retriever2 = ChromaRetriever(
        collection_name="test_child_development",
        persist_dir="./data/chroma_db_test"
    )

    print(f"✓ Loaded {retriever2.collection.count()} chunks from disk")
    print("✓ Persistence test passed!")


if __name__ == "__main__":
    test_chroma_retriever()
