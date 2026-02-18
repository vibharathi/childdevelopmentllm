"""
Local LLM wrapper using llama-cpp-python.
"""

from llama_cpp import Llama
from typing import List, Dict, Optional
import os
from src.config import LLMConfig


class LocalLLM:
    """Wrapper for local LLM using llama-cpp-python."""

    def __init__(
        self,
        model_path: str = LLMConfig.DEFAULT_MODEL_PATH,
        n_ctx: int = LLMConfig.CONTEXT_WINDOW,
        n_threads: int = LLMConfig.NUM_THREADS,
        n_gpu_layers: int = -1
    ):
        """
        Initialize the local LLM.

        Args:
            model_path: Path to the GGUF model file (defaults to config)
            n_ctx: Context window size (defaults to config)
            n_threads: Number of threads for inference (defaults to config)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all layers)
                         Set to 0 to disable GPU acceleration
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}...")
        if n_gpu_layers != 0:
            print(f"GPU acceleration: ENABLED (offloading {n_gpu_layers if n_gpu_layers > 0 else 'all'} layers to Metal)")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # Enable GPU acceleration!
            verbose=False
        )
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
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

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> str:
        """
        Chat-style completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Roles: 'system', 'user', 'assistant'
                     Example: [
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "Hello!"}
                     ]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated response
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response['choices'][0]['message']['content'].strip()

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = LLMConfig.MAX_TOKENS
    ) -> str:
        """
        Generate an answer based on question and retrieved context.
        Uses simple Q&A format which works better for smaller models.

        Args:
            question: User's question
            context: Retrieved context from milestone documents
            max_tokens: Maximum tokens to generate (defaults to config)

        Returns:
            Generated answer
        """
        # Use simple Q&A format which works better than chat format for this model
        prompt = f"""Answer the following question based on the provided context about child development.

Context: {context}

Question: {question}

Answer:"""

        response = self.generate(prompt, max_tokens=max_tokens, temperature=LLMConfig.TEMPERATURE)
        return response


def test_llm():
    """Test the LLM with a simple prompt."""
    print("="*60)
    print("TESTING LOCAL LLM")
    print("="*60)

    # Initialize LLM
    llm = LocalLLM()

    # Test 1: Simple generation using chat
    print("\nTest 1: Simple Chat Generation")
    print("-" * 60)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = llm.llm.create_chat_completion(messages=messages, max_tokens=50)
    print(f"Question: What is 2+2?")
    print(f"Response: {response['choices'][0]['message']['content'].strip()}")

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
