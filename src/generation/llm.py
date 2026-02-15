"""
Local LLM wrapper using llama-cpp-python.
"""

from llama_cpp import Llama
from typing import List, Dict, Optional
import os


class LocalLLM:
    """Wrapper for local LLM using llama-cpp-python."""

    def __init__(self, model_path: str = "data/models/tinyllama.gguf", n_ctx: int = 2048, n_threads: int = 4):
        """
        Initialize the local LLM.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads for inference
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop: Stop sequences

        Returns:
            Generated text
        """
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False
        )

        return response['choices'][0]['text'].strip()

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 256
    ) -> str:
        """
        Generate an answer based on question and retrieved context.

        Args:
            question: User's question
            context: Retrieved context from milestone documents
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer
        """
        prompt = self._create_qa_prompt(question, context)
        answer = self.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        return answer

    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create a prompt for question answering."""
        prompt = f"""<|system|>
You are a helpful assistant specializing in early childhood development (ages 0-36 months).
Answer caregiver questions based ONLY on the provided context about developmental milestones.
Be warm, supportive, and factual. If the context doesn't contain enough information, say so.
Remember: you provide general developmental information, not medical advice.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
        return prompt


def test_llm():
    """Test the LLM with a simple prompt."""
    print("="*60)
    print("TESTING LOCAL LLM")
    print("="*60)

    # Initialize LLM
    llm = LocalLLM()

    # Test 1: Simple generation
    print("\nTest 1: Simple Generation")
    print("-" * 60)
    prompt = "Q: What is 2+2?\nA:"
    response = llm.generate(prompt, max_tokens=50)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # Test 2: Child development question with context
    print("\n\nTest 2: Child Development Q&A")
    print("-" * 60)
    context = """During the first two months of life, newborns gradually adjust to the world
outside the womb. Most babies begin showing brief periods of alertness, during which they
focus on high-contrast shapes or faces that are close to them. Reflexive behaviors—such as
rooting, sucking, and grasping—are especially prominent at this age."""

    question = "What can newborns see in their first two months?"

    answer = llm.generate_answer(question, context)
    print(f"Question: {question}")
    print(f"\nAnswer: {answer}")

    print("\n" + "="*60)
    print("LLM TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_llm()
