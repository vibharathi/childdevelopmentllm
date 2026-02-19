# Claude Code Project Guidelines

This file contains project-specific instructions that Claude will follow across all sessions.

## Git Commit Guidelines

- **NEVER** add "Co-Authored-By: Claude" lines to commit messages
- Generate short, clean, concise commit messages without attribution (2-3 sentences)
- Follow conventional commit style when appropriate

## Code Style Preferences

- Prefer clarity over cleverness
- Add comments only where logic isn't self-evident
- Use type hints in Python function signatures

## Testing Guidelines

- Use Python's `unittest` framework for test suites
- Organize tests by component (e.g., `test_retrievers.py`, `test_utils.py`)
- Include both unit tests and integration tests
- Test edge cases and error conditions

## Project-Specific Rules

### Child Development Q&A System

- This is a RAG (Retrieval-Augmented Generation) system for child development milestones (0-36 months)
- Two retrieval strategies available:
  - `ChromaRetriever`: Dense embeddings with ChromaDB (persistent storage)
  - `HybridRetriever`: BM25 + embeddings with configurable weights
- Age-based filtering is implemented using shared utilities in `src/retrieval/age_utils.py`
- Safety filtering is applied at INDEX time, not query time
- Local LLM: Llama 3.2 3B model via llama-cpp-python

### Code Organization

- Keep retrieval strategies modular and interchangeable
- Share common utilities (like age filtering) across retrievers
- Use ChromaDB for persistent embeddings (no in-memory options)
- Apply DRY principle - extract shared logic into utility modules

## Development Workflow

- When implementing features:
  1. Explain the plan before writing code
  2. Show what tests will be added
  3. Commit related changes together
  4. Keep commits focused and atomic

## Dependencies

- Core: `llama-cpp-python`, `chromadb`, `sentence-transformers`
- Retrieval: `rank-bm25` for keyword search
- ML: `numpy`, `scikit-learn` for scoring

---

**Note:** This file is read at the start of each Claude Code session. Update it with any new preferences or project conventions.
