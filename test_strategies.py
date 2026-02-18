"""
Comprehensive test script to compare embedding vs hybrid retrieval strategies.
Generates detailed comparison data for README documentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import load_milestone_data
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import LocalLLM
import json
from datetime import datetime


# Test questions covering different scenarios
TEST_QUESTIONS = [
    {
        "question": "When do babies start crawling?",
        "category": "Motor Skills",
        "expected_age": "6-9 months"
    },
    {
        "question": "What can a 6-month-old baby do?",
        "category": "General Milestone",
        "expected_age": "6 months"
    },
    {
        "question": "When should my toddler start talking?",
        "category": "Language Development",
        "expected_age": "12-18 months"
    },
    {
        "question": "Is it normal for my 12-month-old not to walk yet?",
        "category": "Developmental Concern",
        "expected_age": "12 months"
    },
    {
        "question": "What social skills should I see at 18 months?",
        "category": "Social Development",
        "expected_age": "18 months"
    }
]


def test_retrieval_strategy(retriever, strategy_name, chunks, questions):
    """
    Test a single retrieval strategy with all questions.

    Args:
        retriever: Retriever instance
        strategy_name: Name of the strategy
        chunks: Document chunks
        questions: List of test questions

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {strategy_name}")
    print(f"{'='*60}")

    # Index documents
    retriever.index_documents(chunks)

    results = []
    retrieval_times = []

    for test_case in questions:
        question = test_case["question"]
        print(f"\nQ: {question}")
        print("-" * 60)

        # Retrieve
        retrieved_chunks, scores, retrieval_time = retriever.retrieve(question, top_k=3)
        retrieval_times.append(retrieval_time)

        # Display results
        print(f"Retrieval time: {retrieval_time*1000:.2f}ms")
        print(f"Top score: {scores[0]:.3f}\n")

        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores)):
            print(f"  [{i+1}] Score: {score:.3f}")
            print(f"      Source: {chunk['source']}")
            print(f"      Age: {chunk['age_range']} months")
            print(f"      Text: {chunk['text'][:80]}...")
            print()

        # Store results
        results.append({
            "question": question,
            "category": test_case["category"],
            "expected_age": test_case["expected_age"],
            "top_score": scores[0],
            "avg_score": sum(scores) / len(scores),
            "retrieval_time_ms": retrieval_time * 1000,
            "retrieved_chunks": [
                {
                    "source": chunk["source"],
                    "age_range": chunk["age_range"],
                    "score": score,
                    "text_preview": chunk["text"][:100]
                }
                for chunk, score in zip(retrieved_chunks, scores)
            ]
        })

    # Calculate aggregate statistics
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    avg_top_score = sum(r["top_score"] for r in results) / len(results)
    avg_all_scores = sum(r["avg_score"] for r in results) / len(results)

    return {
        "strategy": strategy_name,
        "stats": {
            "avg_retrieval_time_ms": avg_retrieval_time * 1000,
            "avg_top_score": avg_top_score,
            "avg_all_scores": avg_all_scores,
            "num_questions": len(questions)
        },
        "results": results
    }


def test_full_qa_pipeline(model_path, questions):
    """
    Test complete Q&A pipeline with both strategies including LLM generation.

    Args:
        model_path: Path to LLM model
        questions: List of test questions

    Returns:
        Dictionary with full pipeline results
    """
    print(f"\n{'='*80}")
    print("TESTING FULL Q&A PIPELINE (Retrieval + Generation)")
    print(f"{'='*80}")

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()

    # Initialize LLM
    print("\nLoading language model...")
    llm = LocalLLM(model_path=model_path)

    pipeline_results = []

    for strategy_name, retriever_class in [("Embedding", EmbeddingRetriever), ("Hybrid", HybridRetriever)]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        # Initialize retriever
        retriever = retriever_class()
        retriever.index_documents(chunks)

        for test_case in questions[:2]:  # Just test first 2 questions with LLM (faster)
            question = test_case["question"]
            print(f"\nQ: {question}")
            print("-" * 60)

            # Retrieve
            retrieved_chunks, scores, retrieval_time = retriever.retrieve(question, top_k=3)

            # Generate answer
            context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            answer = llm.generate_answer(question, context, max_tokens=75)

            print(f"Answer: {answer}")
            print(f"Top score: {scores[0]:.3f} | Retrieval: {retrieval_time*1000:.2f}ms")

            pipeline_results.append({
                "strategy": strategy_name,
                "question": question,
                "answer": answer,
                "top_score": scores[0],
                "retrieval_time_ms": retrieval_time * 1000
            })

    return pipeline_results


def generate_comparison_report(embedding_results, hybrid_results, output_file="comparison_report.json"):
    """
    Generate a comprehensive comparison report.

    Args:
        embedding_results: Results from embedding strategy
        hybrid_results: Results from hybrid strategy
        output_file: Output JSON file path
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison": {
            "embedding_strategy": embedding_results,
            "hybrid_strategy": hybrid_results
        },
        "summary": {
            "winner_by_speed": None,
            "winner_by_relevance": None,
            "observations": []
        }
    }

    # Determine winners
    if embedding_results["stats"]["avg_retrieval_time_ms"] < hybrid_results["stats"]["avg_retrieval_time_ms"]:
        report["summary"]["winner_by_speed"] = "Embedding"
        speed_diff = hybrid_results["stats"]["avg_retrieval_time_ms"] - embedding_results["stats"]["avg_retrieval_time_ms"]
        report["summary"]["observations"].append(
            f"Embedding strategy is faster by {speed_diff:.2f}ms on average"
        )
    else:
        report["summary"]["winner_by_speed"] = "Hybrid"
        speed_diff = embedding_results["stats"]["avg_retrieval_time_ms"] - hybrid_results["stats"]["avg_retrieval_time_ms"]
        report["summary"]["observations"].append(
            f"Hybrid strategy is faster by {speed_diff:.2f}ms on average"
        )

    if embedding_results["stats"]["avg_top_score"] > hybrid_results["stats"]["avg_top_score"]:
        report["summary"]["winner_by_relevance"] = "Embedding"
        score_diff = embedding_results["stats"]["avg_top_score"] - hybrid_results["stats"]["avg_top_score"]
        report["summary"]["observations"].append(
            f"Embedding strategy has higher relevance scores by {score_diff:.3f} on average"
        )
    else:
        report["summary"]["winner_by_relevance"] = "Hybrid"
        score_diff = hybrid_results["stats"]["avg_top_score"] - embedding_results["stats"]["avg_top_score"]
        report["summary"]["observations"].append(
            f"Hybrid strategy has higher relevance scores by {score_diff:.3f} on average"
        )

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nWinner by Speed: {report['summary']['winner_by_speed']}")
    print(f"Winner by Relevance: {report['summary']['winner_by_relevance']}")
    print("\nObservations:")
    for obs in report["summary"]["observations"]:
        print(f"  • {obs}")
    print(f"\nDetailed report saved to: {output_file}")


def main():
    """Run comprehensive strategy comparison."""
    print("="*60)
    print("STRATEGY COMPARISON TEST")
    print("="*60)

    # Load data
    print("\nLoading milestone data...")
    docs, chunks = load_milestone_data()
    print(f"Loaded {len(docs)} documents, {len(chunks)} chunks")

    # Test embedding strategy
    embedding_retriever = EmbeddingRetriever()
    embedding_results = test_retrieval_strategy(
        embedding_retriever,
        "Embedding-Based Retrieval",
        chunks,
        TEST_QUESTIONS
    )

    # Test hybrid strategy
    hybrid_retriever = HybridRetriever()
    hybrid_results = test_retrieval_strategy(
        hybrid_retriever,
        "Hybrid Retrieval (BM25 + Embeddings)",
        chunks,
        TEST_QUESTIONS
    )

    # Generate comparison report
    generate_comparison_report(embedding_results, hybrid_results)

    # Optional: Test full pipeline with LLM (commented out by default as it's slower)
    # Uncomment the line below to test with LLM generation
    # pipeline_results = test_full_qa_pipeline("data/models/tinyllama.gguf", TEST_QUESTIONS)

    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
