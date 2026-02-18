"""
Confidence level calculation utilities.
"""

from typing import List, Literal
from src.config import RetrievalConfig

ConfidenceLevel = Literal['high', 'medium', 'low']


def calculate_confidence(
    scores: List[float],
    high_threshold: float = RetrievalConfig.HIGH_CONFIDENCE_THRESHOLD,
    medium_threshold: float = RetrievalConfig.MEDIUM_CONFIDENCE_THRESHOLD
) -> ConfidenceLevel:
    """
    Calculate confidence level based on retrieval scores.

    Consolidates duplicate logic from main.py (lines 165-172) and
    test_strategies.py (lines 97, 113).

    Args:
        scores: List of retrieval similarity scores
        high_threshold: Minimum avg score for 'high' confidence (defaults to config)
        medium_threshold: Minimum avg score for 'medium' confidence (defaults to config)

    Returns:
        Confidence level: 'high', 'medium', or 'low'

    Examples:
        >>> calculate_confidence([0.8, 0.75, 0.7])
        'high'
        >>> calculate_confidence([0.5, 0.45, 0.4])
        'medium'
        >>> calculate_confidence([0.2, 0.15, 0.1])
        'low'
        >>> calculate_confidence([])
        'low'
    """
    if not scores:
        return 'low'

    avg_score = sum(scores) / len(scores)

    if avg_score > high_threshold:
        return 'high'
    elif avg_score > medium_threshold:
        return 'medium'
    else:
        return 'low'


def calculate_confidence_for_strategy(
    scores: List[float],
    strategy: str
) -> ConfidenceLevel:
    """
    Calculate confidence with strategy-specific thresholds.

    Hybrid scores tend to be lower, so we use different thresholds.

    Args:
        scores: List of retrieval similarity scores
        strategy: Retrieval strategy name ('embedding', 'hybrid', etc.)

    Returns:
        Confidence level: 'high', 'medium', or 'low'

    Examples:
        >>> calculate_confidence_for_strategy([0.5, 0.45], 'embedding')
        'medium'
        >>> calculate_confidence_for_strategy([0.5, 0.45], 'hybrid')
        'high'
    """
    if strategy == "hybrid":
        # Hybrid scores are typically lower, use adjusted thresholds
        # Multiply by 0.9 to lower the bar slightly for hybrid
        return calculate_confidence(
            scores,
            high_threshold=RetrievalConfig.HIGH_CONFIDENCE_THRESHOLD * 0.9,
            medium_threshold=RetrievalConfig.MEDIUM_CONFIDENCE_THRESHOLD * 0.9
        )
    else:
        return calculate_confidence(scores)


def should_generate_answer(
    scores: List[float],
    strategy: str,
    min_threshold: float = None
) -> bool:
    """
    Determine if retrieval scores are sufficient to generate an answer.

    Args:
        scores: List of retrieval similarity scores
        strategy: Retrieval strategy name ('embedding', 'hybrid', etc.)
        min_threshold: Optional custom minimum threshold (overrides strategy defaults)

    Returns:
        True if answer should be generated, False otherwise

    Examples:
        >>> should_generate_answer([0.5, 0.4], 'embedding')
        True
        >>> should_generate_answer([0.2, 0.1], 'embedding')
        False
        >>> should_generate_answer([0.2, 0.1], 'hybrid')
        False
        >>> should_generate_answer([0.28, 0.25], 'hybrid')
        True
    """
    if not scores:
        return False

    if min_threshold is None:
        # Use strategy-specific thresholds
        min_threshold = (
            RetrievalConfig.LOW_CONFIDENCE_THRESHOLD_HYBRID
            if strategy == "hybrid"
            else RetrievalConfig.LOW_CONFIDENCE_THRESHOLD
        )

    return scores[0] >= min_threshold
