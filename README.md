# Child Development LLM - Caregiver Q&A System

A local prototype that answers caregiver questions about early childhood development (birth to 36 months) using retrieval-augmented generation (RAG).

## Overview

This system provides evidence-based answers to caregiver questions about developmental milestones, using a curated dataset of milestone reference texts. The system includes safety mechanisms to handle inappropriate queries and confidence scoring to provide appropriate fallbacks when uncertain.

## Features

- **Fully Local**: Runs entirely on your machine with no cloud API dependencies
- **Dual Retrieval Strategies**: Compares embedding-based vs hybrid retrieval approaches
- **Safety Layer**: Filters inappropriate, medical, or high-risk queries
- **Confidence Scoring**: Detects low-confidence scenarios and provides appropriate fallbacks
- **Open Source**: Built entirely with free, open-source libraries

## Architecture

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Safety Filter  │  ◄── Detect inappropriate/medical queries
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        Retrieval Strategies          │
│  ┌─────────────┬─────────────┐      │
│  │ Strategy 1  │ Strategy 2  │      │
│  │ Embedding   │   Hybrid    │      │
│  │   Based     │  BM25 + Emb │      │
│  └─────────────┴─────────────┘      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Confidence Scoring  │  ◄── Check retrieval quality
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Answer Generation   │  ◄── Local LLM (llama-cpp-python)
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│     Answer      │
└─────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows
- At least 4GB free disk space (for model files)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ChildDevelopmentLLM
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the LLM model**

   Download a GGUF model (we recommend Llama-3.2-3B for balance of speed/quality):
   ```bash
   # Create models directory if not exists
   mkdir -p data/models

   # Download model (example using wget or curl)
   # Option 1: Llama 3.2 3B (Recommended - ~2GB)
   wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O data/models/llama-3.2-3b.gguf

   # Option 2: TinyLlama (Faster, smaller - ~637MB)
   # wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O data/models/tinyllama.gguf
   ```

5. **Verify installation**
   ```bash
   python -c "import sentence_transformers; print('Setup successful!')"
   ```

## Usage

### Command Line Interface

```bash
# Run the Q&A system
python src/main.py

# Run with specific retrieval strategy
python src/main.py --strategy embedding  # or hybrid

# Run comparison mode (both strategies)
python src/main.py --compare
```

### Example Questions

```
> When should my 3-month-old start reaching for toys?
> Is it normal for my 12-month-old not to walk yet?
> What social skills should I see at 18 months?
> My baby isn't making eye contact, what should I do?
```

## Retrieval Strategy Comparison

### Strategy 1: Dense Embedding-Based Retrieval
**Description**: Uses sentence-transformers to create dense vector embeddings of both the knowledge base and queries, then retrieves documents based on cosine similarity.

**Implementation**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Similarity metric: Cosine similarity
- Top-k retrieval: 3 documents

**Pros**:
- [To be filled after testing]

**Cons**:
- [To be filled after testing]

### Strategy 2: Hybrid Retrieval (BM25 + Embeddings)
**Description**: Combines traditional keyword-based retrieval (BM25) with dense embeddings, using weighted score fusion.

**Implementation**:
- BM25 for keyword matching
- Embeddings for semantic similarity
- Score fusion: 0.5 * BM25_score + 0.5 * embedding_score

**Pros**:
- [To be filled after testing]

**Cons**:
- [To be filled after testing]

### Performance Comparison

| Metric | Embedding-Based | Hybrid |
|--------|----------------|--------|
| Avg Retrieval Time | TBD | TBD |
| Relevance Score | TBD | TBD |
| Edge Case Handling | TBD | TBD |

**Winner**: [To be determined after testing]

**Reasoning**: [To be filled after evaluation]

## Safety & Confidence Features

### Safety Layer

The system detects and handles:
- **Medical queries**: Refuses to provide medical diagnoses
- **Inappropriate content**: Filters harmful or off-topic questions
- **High-risk concerns**: Redirects to professional resources

### Confidence Scoring

The system uses multiple signals to detect uncertainty:
- Low similarity scores (< 0.3 threshold)
- High disagreement between retrieved documents
- Missing relevant information
- Safety rule triggers

**Fallback Responses**:
- Uncertainty statement: "I don't have enough specific information about that..."
- Professional referral: "This sounds like something to discuss with your pediatrician..."
- Generic guidance: "Every child develops at their own pace..."

## Project Structure

```
ChildDevelopmentLLM/
├── data/
│   └── models/              # LLM model files (.gguf)
├── milestones/              # Reference texts (birth-36 months)
├── src/
│   ├── retrieval/           # Strategy 1 & 2 implementations
│   ├── generation/          # LLM-based answer generation
│   ├── safety/              # Safety filters and guardrails
│   ├── confidence/          # Uncertainty detection
│   ├── utils/               # Helper functions
│   └── main.py              # CLI interface
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dataset

The system uses a curated dataset of early childhood milestone texts covering:
- 10 age-specific milestone files (0-36 months)
- 3 intentionally noisy files for robustness testing

The dataset includes deliberate inconsistencies to test the system's ability to handle real-world data quality issues.

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

The codebase is modular - each component (retrieval, safety, confidence) can be modified independently.

## Limitations

- Limited to developmental milestone questions (birth-36 months)
- Not a substitute for professional medical advice
- Responses are based on general developmental guidelines
- May not capture individual variation or special circumstances

## Future Improvements

- [ ] Add multilingual support
- [ ] Expand age range coverage
- [ ] Implement user feedback loop
- [ ] Add citation tracking for retrieved sources
- [ ] Web interface

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with open-source models and libraries
- Milestone data based on established developmental frameworks
