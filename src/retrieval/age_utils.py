"""
Shared utilities for age-based filtering in retrieval strategies.
"""

import re
from typing import List, Dict, Optional, Tuple


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
        "newborn" -> 0
        "toddler" -> 12
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
    
    # Pattern 3: "Toddler"
    match = re.search(r'\s+toddler', query_lower)
    if match:
        return 12
    
    # Pattern 3: "Newborn"
    match = re.search(r'\s+newborn', query_lower)
    if match:
        return 0


    return None


def add_age_metadata_to_chunks(chunks: List[Dict]) -> None:
    """
    Add numeric age metadata (min_age, max_age) to chunks in-place.

    This parses the 'age_range' field and adds 'min_age' and 'max_age'
    as separate integer fields for efficient filtering.

    Args:
        chunks: List of document chunks to augment (modified in-place)

    Example:
        chunk = {'age_range': '2-4', 'text': '...'}
        add_age_metadata_to_chunks([chunk])
        # chunk now has: {'age_range': '2-4', 'min_age': 2, 'max_age': 4, 'text': '...'}
    """
    for chunk in chunks:
        age_range_str = chunk.get('age_range')
        min_age, max_age = parse_age_range(age_range_str)
        chunk['min_age'] = min_age
        chunk['max_age'] = max_age


def get_age_matched_indices(chunks: List[Dict], query_age: int) -> List[int]:
    """
    Get indices of chunks that match the specified age.

    Args:
        chunks: List of document chunks with min_age and max_age fields
        query_age: Target age in months

    Returns:
        List of indices where min_age <= query_age <= max_age

    Example:
        chunks = [
            {'min_age': 0, 'max_age': 2, 'text': '...'},
            {'min_age': 2, 'max_age': 4, 'text': '...'},
            {'min_age': 6, 'max_age': 9, 'text': '...'}
        ]
        get_age_matched_indices(chunks, 3) -> [1]  # Only second chunk matches
    """
    return [
        i for i, chunk in enumerate(chunks)
        if chunk.get('min_age') is not None
        and chunk.get('max_age') is not None
        and chunk['min_age'] <= query_age <= chunk['max_age']
    ]
