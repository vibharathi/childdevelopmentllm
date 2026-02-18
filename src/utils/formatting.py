"""
Utilities for formatting and displaying results.
"""

from typing import List, Dict


def format_source_line(source: Dict) -> str:
    """
    Format a single source entry for display.

    Consolidates duplicate logic from main.py (lines 216-219 and 245-248).

    Args:
        source: Dictionary with 'filename', 'age_range', and 'score' keys

    Returns:
        Formatted string for display

    Examples:
        >>> source = {'filename': 'doc.txt', 'age_range': '6-9', 'score': 0.85}
        >>> format_source_line(source)
        '  • doc.txt (Age: 6-9 months) [score: 0.850]'

        >>> source = {'filename': 'doc.txt', 'age_range': '', 'score': 0.5}
        >>> format_source_line(source)
        '  • doc.txt [score: 0.500]'
    """
    age_str = f" (Age: {source['age_range']} months)" if source.get('age_range') else ""
    score_str = f" [score: {source['score']:.3f}]" if 'score' in source else ""
    return f"  • {source['filename']}{age_str}{score_str}"


def format_sources(sources: List[Dict]) -> str:
    """
    Format multiple sources for display.

    Args:
        sources: List of source dictionaries

    Returns:
        Multi-line formatted string

    Examples:
        >>> sources = [
        ...     {'filename': 'doc1.txt', 'age_range': '0-2', 'score': 0.9},
        ...     {'filename': 'doc2.txt', 'age_range': '2-4', 'score': 0.8}
        ... ]
        >>> print(format_sources(sources))
          • doc1.txt (Age: 0-2 months) [score: 0.900]
          • doc2.txt (Age: 2-4 months) [score: 0.800]
    """
    return "\n".join(format_source_line(source) for source in sources)


def format_metrics_line(result: Dict) -> str:
    """
    Format metrics line for display.

    Consolidates lines 222-223 and 251-252 from main.py.

    Args:
        result: Result dictionary with strategy, confidence, and timing info

    Returns:
        Formatted metrics string (two lines)

    Examples:
        >>> result = {
        ...     'strategy': 'embedding',
        ...     'confidence': 'high',
        ...     'retrieval_time': 0.015,
        ...     'generation_time': 2.3,
        ...     'total_time': 2.315
        ... }
        >>> print(format_metrics_line(result))
        Strategy: embedding | Confidence: high
        Retrieval: 15.00ms | Generation: 2.3s | Total: 2.3s
    """
    line1 = f"Strategy: {result['strategy']} | Confidence: {result['confidence']}"

    retrieval_ms = result['retrieval_time'] * 1000
    gen_time = result.get('generation_time', 0)
    total_time = result.get('total_time', 0)

    line2 = f"Retrieval: {retrieval_ms:.2f}ms | Generation: {gen_time:.1f}s | Total: {total_time:.1f}s"

    return f"{line1}\n{line2}"


def print_answer_with_sources(result: Dict, show_separator: bool = True):
    """
    Print answer with sources and metrics in standardized format.

    Consolidates both result display blocks from main.py:
    - interactive_mode() lines 208-224
    - single_question() lines 237-253

    Args:
        result: Result dictionary with answer, sources, and metadata
        show_separator: Whether to show separator lines (True for single_question,
                       False for interactive_mode)

    Examples:
        >>> result = {
        ...     'answer': 'Babies typically crawl around 8-10 months.',
        ...     'sources': [{'filename': 'doc.txt', 'age_range': '8-10', 'score': 0.85}],
        ...     'confidence': 'high',
        ...     'strategy': 'embedding',
        ...     'retrieval_time': 0.01,
        ...     'generation_time': 1.5,
        ...     'total_time': 1.51
        ... }
        >>> print_answer_with_sources(result, show_separator=True)
        ============================================================
        ANSWER:
        ============================================================
        Babies typically crawl around 8-10 months.
        <BLANKLINE>
        ============================================================
        SOURCES:
        ============================================================
          • doc.txt (Age: 8-10 months) [score: 0.850]
        <BLANKLINE>
        ------------------------------------------------------------
        Strategy: embedding | Confidence: high
        Retrieval: 10.00ms | Generation: 1.5s | Total: 1.5s
        ============================================================
    """
    sep = "=" * 60
    sub_sep = "-" * 60

    # Print answer section
    if show_separator:
        print(f"\n{sep}")
        print("ANSWER:")
        print(sep)
    print(result['answer'])

    # Print sources section
    if show_separator:
        print(f"\n{sep}")
        print("SOURCES:")
        print(sep)
    else:
        print(f"\n{sub_sep}")
        print("SOURCES:")
        print(sub_sep)

    # Print each source
    for source in result['sources']:
        print(format_source_line(source))

    # Print metrics
    print(f"\n{sub_sep}")
    print(format_metrics_line(result))
    if show_separator:
        print(sep)
    else:
        print(sub_sep)
