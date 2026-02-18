"""
Full system test - Generate example outputs for README documentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import ChildDevelopmentQA
import json
from datetime import datetime


# Test questions covering different scenarios
TEST_QUESTIONS = [
    {
        "question": "When do babies typically start crawling?",
        "category": "Motor Development",
        "reason": "Common developmental milestone question"
    },
    {
        "question": "My 12-month-old isn't walking yet. Should I be worried?",
        "category": "Developmental Concern",
        "reason": "Tests uncertainty handling and reassurance"
    },
    {
        "question": "What social skills should I see at 18 months?",
        "category": "Social Development",
        "reason": "Tests age-specific retrieval"
    },
    {
        "question": "Is it normal for a 6-month-old to not say words yet?",
        "category": "Language Development",
        "reason": "Tests understanding of normal developmental timeline"
    },
    {
        "question": "What can my newborn see?",
        "category": "Early Infancy",
        "reason": "Tests very early developmental period (0-2 months)"
    }
]


def test_both_strategies():
    """Test both retrieval strategies with all questions."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_results": []
    }

    for strategy in ["embedding", "hybrid"]:
        print("\n" + "="*80)
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print("="*80)

        # Initialize system
        qa_system = ChildDevelopmentQA(
            model_path="data/models/tinyllama.gguf",
            strategy=strategy
        )

        strategy_results = {
            "strategy": strategy,
            "questions": []
        }

        for i, test_case in enumerate(TEST_QUESTIONS):
            print(f"\n{'='*80}")
            print(f"Question {i+1}/{len(TEST_QUESTIONS)}")
            print(f"{'='*80}")
            print(f"Q: {test_case['question']}")
            print(f"Category: {test_case['category']}")
            print("-" * 80)

            # Get answer
            result = qa_system.answer_question(test_case['question'])

            # Display result
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nConfidence: {result['confidence']}")
            print(f"Retrieval Time: {result['retrieval_time']*1000:.2f}ms")
            print(f"Sources: {len(result['sources'])}")
            for src in result['sources']:
                print(f"  - {src['filename']} (Age: {src['age_range']}, Score: {src['score']:.3f})")

            # Store result
            strategy_results["questions"].append({
                "question": test_case['question'],
                "category": test_case['category'],
                "answer": result['answer'],
                "confidence": result['confidence'],
                "retrieval_time_ms": result['retrieval_time'] * 1000,
                "sources": result['sources']
            })

        results["test_results"].append(strategy_results)

    # Save results
    output_file = "full_system_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = test_both_strategies()

    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
