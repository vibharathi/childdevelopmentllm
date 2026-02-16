"""
Performance benchmark for Llama 3.2 3B on Mac.
Tests load time, inference speed, and memory usage.
"""

import time
import psutil
import os
from src.generation.llm import LocalLLM


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_load_time(model_path: str):
    """Benchmark model loading time."""
    print("\n" + "="*70)
    print("BENCHMARK 1: MODEL LOAD TIME")
    print("="*70)

    mem_before = get_memory_usage()
    start_time = time.time()

    llm = LocalLLM(model_path=model_path, n_ctx=2048)

    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    print(f"✓ Load time: {load_time:.2f} seconds")
    print(f"✓ Memory used: {mem_used:.0f} MB ({mem_after:.0f} MB total)")

    return llm, load_time, mem_used


def benchmark_inference_speed(llm: LocalLLM, prompt: str, max_tokens: int = 256):
    """Benchmark token generation speed."""
    print("\n" + "="*70)
    print("BENCHMARK 2: INFERENCE SPEED")
    print("="*70)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Max tokens: {max_tokens}")
    print("-"*70)

    start_time = time.time()

    response = llm.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )

    generation_time = time.time() - start_time
    tokens_generated = len(response.split())  # Approximate token count
    tokens_per_second = tokens_generated / generation_time

    print(f"\n✓ Generation time: {generation_time:.2f} seconds")
    print(f"✓ Tokens generated: ~{tokens_generated}")
    print(f"✓ Speed: {tokens_per_second:.2f} tokens/second")
    print(f"\nResponse preview: {response[:200]}...")

    return tokens_per_second, response


def benchmark_qa_performance(llm: LocalLLM):
    """Benchmark Q&A performance with child development context."""
    print("\n" + "="*70)
    print("BENCHMARK 3: Q&A PERFORMANCE (Child Development)")
    print("="*70)

    context = """During the first two months of life, newborns gradually adjust to the world
outside the womb. Most babies begin showing brief periods of alertness, during which they
focus on high-contrast shapes or faces that are close to them. Reflexive behaviors—such as
rooting, sucking, and grasping—are especially prominent at this age. By 2 months, many
infants can lift their heads briefly when placed on their stomachs and may track moving
objects with their eyes. Social smiles typically emerge around 6-8 weeks, marking an
important milestone in social-emotional development."""

    question = "What can newborns see in their first two months?"

    print(f"Question: {question}")
    print("-"*70)

    start_time = time.time()
    answer = llm.generate_answer(question, context, max_tokens=150)
    generation_time = time.time() - start_time

    print(f"\n✓ Response time: {generation_time:.2f} seconds")
    print(f"\nAnswer:\n{answer}")

    return generation_time, answer


def benchmark_different_lengths(llm: LocalLLM):
    """Benchmark with different prompt lengths."""
    print("\n" + "="*70)
    print("BENCHMARK 4: DIFFERENT PROMPT LENGTHS")
    print("="*70)

    test_cases = [
        ("Short", "What is 2+2?", 20),
        ("Medium", "Explain what happens during a child's first year of development.", 100),
        ("Long", "Describe in detail the cognitive, physical, social, and emotional developmental milestones that occur during the first three years of a child's life, including specific examples of behaviors and abilities that emerge at different ages.", 200)
    ]

    results = []

    for label, prompt, max_tokens in test_cases:
        print(f"\n{label} prompt ({len(prompt)} chars, max {max_tokens} tokens)")
        print("-"*70)

        start_time = time.time()
        response = llm.generate(prompt, max_tokens=max_tokens, temperature=0.7)
        generation_time = time.time() - start_time

        tokens_generated = len(response.split())
        tokens_per_second = tokens_generated / generation_time

        print(f"✓ Time: {generation_time:.2f}s | Speed: {tokens_per_second:.2f} tok/s | Generated: ~{tokens_generated} tokens")

        results.append({
            'label': label,
            'time': generation_time,
            'speed': tokens_per_second,
            'tokens': tokens_generated
        })

    return results


def print_summary(load_time, mem_used, avg_speed, all_speeds):
    """Print benchmark summary."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Mac Model: M2 MacBook Air (8GB RAM)")
    print(f"Model: Llama 3.2 3B (~1.9GB)")
    print("-"*70)
    print(f"Load Time:        {load_time:.2f} seconds")
    print(f"Memory Usage:     {mem_used:.0f} MB")
    print(f"Avg Speed:        {avg_speed:.2f} tokens/second")
    print(f"Speed Range:      {min(all_speeds):.2f} - {max(all_speeds):.2f} tok/s")
    print("-"*70)

    # Performance rating
    if avg_speed > 20:
        rating = "EXCELLENT ⭐⭐⭐⭐⭐"
    elif avg_speed > 15:
        rating = "VERY GOOD ⭐⭐⭐⭐"
    elif avg_speed > 10:
        rating = "GOOD ⭐⭐⭐"
    elif avg_speed > 5:
        rating = "ACCEPTABLE ⭐⭐"
    else:
        rating = "SLOW ⭐"

    print(f"Performance:      {rating}")
    print("="*70)


def main():
    """Run all benchmarks."""
    print("\n" + "█"*70)
    print("  LLAMA 3.2 3B PERFORMANCE BENCHMARK")
    print("  M2 MacBook Air | 8GB RAM | llama-cpp-python")
    print("█"*70)

    model_path = "data/models/llama-3.2-3b.gguf"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    try:
        # Benchmark 1: Load time
        llm, load_time, mem_used = benchmark_load_time(model_path)

        # Benchmark 2: Basic inference
        speed1, _ = benchmark_inference_speed(
            llm,
            "Explain the importance of early childhood development in 3 sentences.",
            max_tokens=100
        )

        # Benchmark 3: Q&A performance
        qa_time, _ = benchmark_qa_performance(llm)

        # Benchmark 4: Different lengths
        length_results = benchmark_different_lengths(llm)

        # Calculate average speed
        all_speeds = [speed1] + [r['speed'] for r in length_results]
        avg_speed = sum(all_speeds) / len(all_speeds)

        # Print summary
        print_summary(load_time, mem_used, avg_speed, all_speeds)

        print("\n✅ Benchmark complete!\n")

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
