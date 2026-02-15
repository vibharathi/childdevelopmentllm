"""
Main CLI interface for the Child Development Q&A system.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_milestone_data
from src.generation.llm import LocalLLM


class ChildDevelopmentQA:
    """Main Q&A system for child development questions."""

    def __init__(self, model_path: str = "data/models/tinyllama.gguf"):
        """
        Initialize the Q&A system.

        Args:
            model_path: Path to the LLM model
        """
        print("="*60)
        print("CHILD DEVELOPMENT Q&A SYSTEM")
        print("Birth to 36 Months")
        print("="*60)

        # Load milestone data
        print("\n[1/3] Loading milestone data...")
        self.documents, self.chunks = load_milestone_data()

        # Initialize LLM
        print("\n[2/3] Loading language model...")
        self.llm = LocalLLM(model_path=model_path)

        print("\n[3/3] System ready!")
        print("="*60)

    def simple_retrieval(self, question: str, top_k: int = 3):
        """
        Simple keyword-based retrieval (placeholder for now).

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of retrieved chunks
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Score each chunk based on keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(question_words & chunk_words)

            # Boost score if age-related keywords match
            if any(word in question_lower for word in ['month', 'old', 'age']):
                # Try to extract age from question
                import re
                age_match = re.search(r'(\d+)[- ]month', question_lower)
                if age_match and chunk['age_range']:
                    question_age = int(age_match.group(1))
                    chunk_age_range = chunk['age_range'].split('-')
                    if len(chunk_age_range) == 2:
                        min_age, max_age = int(chunk_age_range[0]), int(chunk_age_range[1])
                        if min_age <= question_age <= max_age:
                            overlap += 10  # Boost for age match

            scored_chunks.append((overlap, chunk))

        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    def answer_question(self, question: str) -> dict:
        """
        Answer a question about child development.

        Args:
            question: User's question

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.simple_retrieval(question, top_k=3)

        if not retrieved_chunks:
            return {
                'answer': "I don't have specific information about that in my knowledge base. "
                         "Please try asking about developmental milestones for children aged 0-36 months.",
                'sources': [],
                'confidence': 'low'
            }

        # Combine retrieved text as context
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])

        # Generate answer using LLM
        answer = self.llm.generate_answer(question, context, max_tokens=300)

        # Extract sources
        sources = [
            {
                'filename': chunk['source'],
                'age_range': chunk['age_range']
            }
            for chunk in retrieved_chunks
        ]

        return {
            'answer': answer,
            'sources': sources,
            'confidence': 'high' if len(retrieved_chunks) >= 2 else 'medium'
        }

    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("\nAsk questions about child development (ages 0-36 months)")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                question = input("\n> ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if not question:
                    continue

                print("\nThinking...")
                result = self.answer_question(question)

                print("\n" + "-"*60)
                print("ANSWER:")
                print("-"*60)
                print(result['answer'])

                print("\n" + "-"*60)
                print("SOURCES:")
                print("-"*60)
                for source in result['sources']:
                    age_str = f" (Age: {source['age_range']} months)" if source['age_range'] else ""
                    print(f"  • {source['filename']}{age_str}")

                print("\n" + "-"*60)
                print(f"Confidence: {result['confidence']}")
                print("-"*60)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

    def single_question(self, question: str):
        """Answer a single question and exit."""
        result = self.answer_question(question)

        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])

        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        for source in result['sources']:
            age_str = f" (Age: {source['age_range']} months)" if source['age_range'] else ""
            print(f"  • {source['filename']}{age_str}")

        print("\n" + "="*60)
        print(f"Confidence: {result['confidence']}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Child Development Q&A System (0-36 months)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/tinyllama.gguf',
        help='Path to LLM model file'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Ask a single question and exit'
    )

    args = parser.parse_args()

    # Initialize system
    qa_system = ChildDevelopmentQA(model_path=args.model)

    # Run in appropriate mode
    if args.question:
        qa_system.single_question(args.question)
    else:
        qa_system.interactive_mode()


if __name__ == "__main__":
    main()
