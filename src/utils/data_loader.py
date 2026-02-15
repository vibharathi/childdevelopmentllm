"""
Data loading utilities for milestone reference texts.
Handles loading, parsing, and chunking of milestone documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import re


class MilestoneDocument:
    """Represents a single milestone document with metadata."""

    def __init__(self, content: str, filename: str, age_range: str = None, is_noisy: bool = False):
        self.content = content.strip()
        self.filename = filename
        self.age_range = age_range
        self.is_noisy = is_noisy
        self.chunks = []

    def __repr__(self):
        return f"MilestoneDocument(filename='{self.filename}', age_range='{self.age_range}', noisy={self.is_noisy})"


class DataLoader:
    """Loads and processes milestone reference texts."""

    def __init__(self, data_dir: str = "milestones"):
        self.data_dir = Path(data_dir)
        self.documents: List[MilestoneDocument] = []

    def load_all_documents(self) -> List[MilestoneDocument]:
        """Load all milestone documents from the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        txt_files = sorted(self.data_dir.glob("*.txt"))

        for filepath in txt_files:
            doc = self._load_document(filepath)
            self.documents.append(doc)

        print(f"Loaded {len(self.documents)} documents:")
        print(f"  - {len([d for d in self.documents if not d.is_noisy])} milestone documents")
        print(f"  - {len([d for d in self.documents if d.is_noisy])} noisy documents")

        return self.documents

    def _load_document(self, filepath: Path) -> MilestoneDocument:
        """Load a single document and extract metadata."""
        filename = filepath.name

        # Read content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine if noisy
        is_noisy = filename.startswith("noisy_")

        # Extract age range from filename for milestone documents
        age_range = None
        if not is_noisy:
            age_match = re.search(r'(\d+(?:-\d+)?)[_\s]months?', filename)
            if age_match:
                age_range = age_match.group(1)
            else:
                # Try to extract from file number (e.g., 01_newborn_0-2_months)
                age_match = re.search(r'_(\d+-\d+)_months', filename)
                if age_match:
                    age_range = age_match.group(1)

        return MilestoneDocument(
            content=content,
            filename=filename,
            age_range=age_range,
            is_noisy=is_noisy
        )

    def get_clean_documents(self) -> List[MilestoneDocument]:
        """Return only non-noisy milestone documents."""
        return [doc for doc in self.documents if not doc.is_noisy]

    def get_noisy_documents(self) -> List[MilestoneDocument]:
        """Return only noisy documents."""
        return [doc for doc in self.documents if doc.is_noisy]

    def chunk_documents(self, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            chunk_size: Maximum number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []

        for doc in self.documents:
            # For shorter documents, keep them whole
            words = doc.content.split()

            if len(words) <= chunk_size:
                chunks.append({
                    'text': doc.content,
                    'source': doc.filename,
                    'age_range': doc.age_range,
                    'is_noisy': doc.is_noisy,
                    'chunk_id': 0
                })
            else:
                # Split into overlapping chunks
                start = 0
                chunk_id = 0

                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_text = ' '.join(words[start:end])

                    chunks.append({
                        'text': chunk_text,
                        'source': doc.filename,
                        'age_range': doc.age_range,
                        'is_noisy': doc.is_noisy,
                        'chunk_id': chunk_id
                    })

                    chunk_id += 1
                    start += (chunk_size - overlap)

        return chunks

    def get_document_stats(self) -> Dict:
        """Get statistics about the loaded documents."""
        clean_docs = self.get_clean_documents()
        noisy_docs = self.get_noisy_documents()

        stats = {
            'total_documents': len(self.documents),
            'clean_documents': len(clean_docs),
            'noisy_documents': len(noisy_docs),
            'age_ranges': [doc.age_range for doc in clean_docs if doc.age_range],
            'total_characters': sum(len(doc.content) for doc in self.documents),
            'avg_doc_length': sum(len(doc.content) for doc in self.documents) / len(self.documents) if self.documents else 0
        }

        return stats


def load_milestone_data(data_dir: str = "milestones") -> Tuple[List[MilestoneDocument], List[Dict]]:
    """
    Convenience function to load and chunk milestone data.

    Returns:
        Tuple of (documents, chunks)
    """
    loader = DataLoader(data_dir)
    documents = loader.load_all_documents()
    chunks = loader.chunk_documents()

    print(f"\nCreated {len(chunks)} text chunks for retrieval")

    return documents, chunks


if __name__ == "__main__":
    # Test the data loader
    docs, chunks = load_milestone_data()

    print("\n" + "="*50)
    print("DOCUMENT STATISTICS")
    print("="*50)

    loader = DataLoader()
    loader.documents = docs
    stats = loader.get_document_stats()

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "="*50)
    print("SAMPLE CHUNKS")
    print("="*50)

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Source: {chunk['source']}")
        print(f"Age Range: {chunk['age_range']}")
        print(f"Text: {chunk['text'][:150]}...")
