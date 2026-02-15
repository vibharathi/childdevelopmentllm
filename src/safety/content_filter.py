"""
Content-based safety and quality filter for retrieved documents.

This module detects and filters out:
- Documents with disclaimers indicating unreliable information
- Medically implausible claims
- Low-quality or contradictory content
"""

import re
from typing import List, Dict, Tuple


class ContentFilter:
    """
    Filter for detecting and removing low-quality or unsafe content.

    This addresses the safety/quality layer requirement by identifying
    documents that contain misinformation, disclaimers, or implausible claims.
    """

    # Disclaimer patterns that indicate unreliable content
    DISCLAIMER_PATTERNS = [
        r"these statements differ",
        r"not supported by",
        r"contrary to",
        r"contradicts",
        r"note:.*differ",
        r"warning:",
        r"this information may be",
        r"unverified",
        r"not endorsed",
        r"anecdotal",
        r"not part of.*standardized",
        r"not part of.*framework",
        r"not.*established.*guideline"
    ]

    # Medically implausible milestone claims (age in months)
    IMPLAUSIBLE_CLAIMS = [
        (r"walking.*\b([2-5])\s*month", "Walking before 6 months is medically implausible"),
        (r"running.*\b([2-9])\s*month", "Running before 10 months is implausible"),
        (r"talking.*sentences.*\b([2-9])\s*month", "Sentence formation before 10 months is rare"),
        (r"reading.*\b([0-2][0-9])\s*month", "Reading in infancy is implausible"),
    ]

    # Quality indicators that suggest reliable content
    QUALITY_INDICATORS = [
        r"typically",
        r"most (infants|babies|children)",
        r"generally",
        r"developmental milestone",
        r"pediatric",
        r"around \d+ months"
    ]

    def __init__(self,
                 filter_disclaimers: bool = True,
                 filter_implausible: bool = True,
                 strict_mode: bool = False):
        """
        Initialize the content filter.

        Args:
            filter_disclaimers: Remove documents with disclaimer text
            filter_implausible: Remove documents with medically implausible claims
            strict_mode: If True, be more aggressive in filtering
        """
        self.filter_disclaimers = filter_disclaimers
        self.filter_implausible = filter_implausible
        self.strict_mode = strict_mode

    def filter_chunks(self,
                     chunks: List[Dict],
                     scores: List[float]) -> Tuple[List[Dict], List[float], List[str]]:
        """
        Filter retrieved chunks based on content quality.

        Args:
            chunks: List of retrieved document chunks
            scores: Corresponding similarity scores

        Returns:
            Tuple of (filtered_chunks, filtered_scores, reasons_removed)
        """
        filtered_chunks = []
        filtered_scores = []
        reasons_removed = []

        for chunk, score in zip(chunks, scores):
            text = chunk['text'].lower()
            source = chunk.get('source', 'unknown')

            # Check for disclaimers
            if self.filter_disclaimers and self._has_disclaimer(text):
                reasons_removed.append(f"{source}: Contains disclaimer indicating unreliable content")
                continue

            # Check for implausible claims
            if self.filter_implausible:
                implausible, reason = self._has_implausible_claim(text)
                if implausible:
                    reasons_removed.append(f"{source}: {reason}")
                    continue

            # Passed all filters
            filtered_chunks.append(chunk)
            filtered_scores.append(score)

        return filtered_chunks, filtered_scores, reasons_removed

    def _has_disclaimer(self, text: str) -> bool:
        """
        Check if text contains disclaimer patterns.

        Args:
            text: Document text (lowercased)

        Returns:
            True if disclaimer found
        """
        for pattern in self.DISCLAIMER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_implausible_claim(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains medically implausible claims.

        Args:
            text: Document text (lowercased)

        Returns:
            Tuple of (is_implausible, reason)
        """
        for pattern, reason in self.IMPLAUSIBLE_CLAIMS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, reason
        return False, ""

    def assess_quality(self, chunk: Dict) -> float:
        """
        Assess the quality of a document chunk.

        Returns a quality score between 0 and 1.
        Higher scores indicate higher quality.

        Args:
            chunk: Document chunk to assess

        Returns:
            Quality score (0-1)
        """
        text = chunk['text'].lower()
        quality_score = 0.5  # Start neutral

        # Boost for quality indicators
        quality_count = sum(1 for pattern in self.QUALITY_INDICATORS
                          if re.search(pattern, text, re.IGNORECASE))
        quality_score += min(0.3, quality_count * 0.1)

        # Penalize for disclaimers
        if self._has_disclaimer(text):
            quality_score -= 0.4

        # Penalize for implausible claims
        if self._has_implausible_claim(text)[0]:
            quality_score -= 0.5

        # Boost for having age range metadata
        if chunk.get('age_range'):
            quality_score += 0.1

        return max(0.0, min(1.0, quality_score))

    def get_filter_stats(self,
                        original_count: int,
                        filtered_count: int,
                        reasons: List[str]) -> Dict:
        """
        Generate statistics about filtering operation.

        Args:
            original_count: Number of chunks before filtering
            filtered_count: Number of chunks after filtering
            reasons: List of reasons for removal

        Returns:
            Dictionary with filter statistics
        """
        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': original_count - filtered_count,
            'removal_rate': (original_count - filtered_count) / original_count if original_count > 0 else 0,
            'reasons': reasons
        }


def test_content_filter():
    """Test the content filter with sample documents."""
    print("="*60)
    print("TESTING CONTENT FILTER")
    print("="*60)

    # Sample chunks (including noisy ones)
    test_chunks = [
        {
            'text': 'Most babies begin walking between 9-15 months. This is a typical milestone.',
            'source': 'good_doc.txt',
            'age_range': '9-15'
        },
        {
            'text': 'Some sources report that most infants begin walking independently as early as 4 months old. (Note: These statements differ substantially from most pediatric guidance.)',
            'source': 'noisy_doc.txt',
            'age_range': None
        },
        {
            'text': 'Babies typically start crawling around 6-9 months.',
            'source': 'another_good_doc.txt',
            'age_range': '6-9'
        }
    ]

    scores = [0.8, 0.7, 0.75]

    filter = ContentFilter()

    print("\nOriginal chunks:")
    for chunk, score in zip(test_chunks, scores):
        print(f"  - {chunk['source']}: {score:.2f}")

    filtered_chunks, filtered_scores, reasons = filter.filter_chunks(test_chunks, scores)

    print(f"\nFiltered chunks:")
    for chunk, score in zip(filtered_chunks, filtered_scores):
        print(f"  - {chunk['source']}: {score:.2f}")

    print(f"\nRemoved:")
    for reason in reasons:
        print(f"  - {reason}")

    print("\nQuality assessment:")
    for chunk in test_chunks:
        quality = filter.assess_quality(chunk)
        print(f"  - {chunk['source']}: {quality:.2f}")


if __name__ == "__main__":
    test_content_filter()
