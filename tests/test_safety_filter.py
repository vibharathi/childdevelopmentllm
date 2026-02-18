"""
Test script to verify safety filter is catching all noisy documents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_milestone_data
from src.safety.content_filter import ContentFilter


def test_safety_filter_on_noisy_docs():
    """Test if safety filter catches all noisy documents."""

    print("="*80)
    print("TESTING SAFETY FILTER ON NOISY DOCUMENTS")
    print("="*80)

    # Load all documents
    docs, chunks = load_milestone_data()

    # Initialize filter
    content_filter = ContentFilter(
        filter_disclaimers=True,
        filter_implausible=True,
        strict_mode=False
    )

    # Find all noisy documents
    noisy_chunks = [chunk for chunk in chunks if 'noisy' in chunk['source'].lower()]
    good_chunks = [chunk for chunk in chunks if 'noisy' not in chunk['source'].lower()]

    print(f"\nFound {len(noisy_chunks)} noisy documents:")
    for chunk in noisy_chunks:
        print(f"  - {chunk['source']}")

    print(f"\nFound {len(good_chunks)} good documents")

    print("\n" + "="*80)
    print("TESTING EACH NOISY DOCUMENT")
    print("="*80)

    caught_count = 0
    missed_count = 0

    for chunk in noisy_chunks:
        print(f"\nTesting: {chunk['source']}")
        print("-" * 80)
        print(f"Content preview: {chunk['text'][:150]}...")

        # Test if filter catches it (using production index-time filtering)
        filtered_chunks, reasons = content_filter.filter_before_indexing(
            [chunk], quality_threshold=0.4
        )

        if len(filtered_chunks) == 0:
            # Successfully filtered out
            print(f"✅ CAUGHT - Reason: {reasons[0]}")
            caught_count += 1
        else:
            # Missed by filter
            print(f"❌ MISSED - Document passed through filter!")
            print(f"   Quality score: {content_filter.assess_quality(chunk):.2f}")
            missed_count += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total noisy documents: {len(noisy_chunks)}")
    print(f"✅ Caught by filter: {caught_count}/{len(noisy_chunks)} ({caught_count/len(noisy_chunks)*100:.1f}%)")
    print(f"❌ Missed by filter: {missed_count}/{len(noisy_chunks)} ({missed_count/len(noisy_chunks)*100:.1f}%)")

    if missed_count > 0:
        print(f"\n⚠️  WARNING: Safety filter is not catching all noisy documents!")
    else:
        print(f"\n✅ SUCCESS: All noisy documents properly filtered!")

    print("\n" + "="*80)
    print("TESTING FALSE POSITIVES")
    print("="*80)

    # Test a few good documents to make sure we're not over-filtering
    test_good_chunks = good_chunks[:5]
    false_positives = 0

    for chunk in test_good_chunks:
        filtered_chunks, reasons = content_filter.filter_before_indexing(
            [chunk], quality_threshold=0.4
        )

        if len(filtered_chunks) == 0:
            print(f"❌ FALSE POSITIVE: {chunk['source']} was incorrectly filtered")
            print(f"   Reason: {reasons[0]}")
            false_positives += 1
        else:
            print(f"✅ PASS: {chunk['source']}")

    print(f"\nFalse positives: {false_positives}/{len(test_good_chunks)}")

    return caught_count, missed_count, len(noisy_chunks)


if __name__ == "__main__":
    caught, missed, total = test_safety_filter_on_noisy_docs()

    print("\n" + "="*80)
    if missed == 0 and caught == total:
        print("🎉 ALL TESTS PASSED! Safety filter working correctly.")
    else:
        print("⚠️  ISSUES FOUND - Review safety filter patterns.")
    print("="*80)
