"""
Utilities for normalizing and manipulating retrieval scores.
"""

import numpy as np
from typing import List, Union


def normalize_bm25_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize BM25 scores to 0-1 range.

    Consolidates BM25 score normalization from hybrid_retriever.py (lines 177-181).

    Args:
        scores: BM25 scores (unbounded)

    Returns:
        Normalized scores in [0, 1] range

    Examples:
        >>> scores = np.array([10.0, 20.0, 30.0])
        >>> normalize_bm25_scores(scores)
        array([0.33333333, 0.66666667, 1.        ])

        >>> scores = np.array([0.0, 0.0, 0.0])  # Edge case: all zeros
        >>> normalize_bm25_scores(scores)
        array([0., 0., 0.])
    """
    if scores.max() > 0:
        return scores / scores.max()
    return scores


def normalize_scores_to_range(
    scores: Union[np.ndarray, List[float]],
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize scores to a specified range [min_val, max_val] using min-max normalization.

    Args:
        scores: Array or list of scores to normalize
        min_val: Minimum value of output range (default: 0.0)
        max_val: Maximum value of output range (default: 1.0)

    Returns:
        Normalized scores as numpy array

    Examples:
        >>> scores = np.array([10, 20, 30])
        >>> normalize_scores_to_range(scores)
        array([0. , 0.5, 1. ])

        >>> scores = np.array([5, 5, 5])  # Edge case: all scores are the same
        >>> normalize_scores_to_range(scores)
        array([5., 5., 5.])

        >>> scores = [1.5, 2.5, 3.5]
        >>> normalize_scores_to_range(scores, min_val=-1, max_val=1)
        array([-1.,  0.,  1.])
    """
    scores = np.array(scores)

    # Handle edge case: all scores are the same
    score_range = scores.max() - scores.min()
    if score_range == 0:
        return scores  # Return original scores (no normalization possible)

    # Min-max normalization
    normalized = (scores - scores.min()) / score_range

    # Scale to desired range
    return normalized * (max_val - min_val) + min_val
