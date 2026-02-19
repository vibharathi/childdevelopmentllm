# Child Development LLM - Caregiver Q&A System

A local prototype that answers caregiver questions about early childhood development (birth to 36 months) using retrieval-augmented generation (RAG).

## Overview

This system provides evidence-based answers to caregiver questions about developmental milestones, using a curated dataset of milestone reference texts. The system includes safety mechanisms to handle inappropriate queries and confidence scoring to provide appropriate fallbacks when uncertain.

## Features

- **Fully Local**: Runs entirely on your machine with no cloud API dependencies
- **Three Retrieval Strategies**: ChromaDB, Embedding-based, and Hybrid (BM25 + embeddings)
- **Age-Aware Filtering**: Automatically filters documents by age range from queries (e.g., "2 month old")
- **GPU Acceleration**: Metal GPU support on Mac (M1/M2) for 10-100x faster inference
- **Safety Layer**: Index-time filtering removes low-quality documents before indexing
- **Confidence Scoring**: Two-tier system (gate check + confidence label) prevents hallucination
- **Persistent Storage**: ChromaDB maintains indexed embeddings across sessions
- **Clean Architecture**: Modern Python design with base classes, utilities, and zero code duplication
- **Open Source**: Built entirely with free, open-source libraries

## Architecture
1. **Chunking and DataLoading**
utils/data_loader.py Each document is broken into chunks of 300 words, most docs are just 1 chunk since the doc size is small enough.
Also reads age information from the file and adds this to chunk metadata. 
2. **Safety and Index time filtering**
safety/content_filter.py Filters noisy documents at index time. They will not be placed in the DB for retrieval if filtered. Filtering is based not on the file name (noisy*) but instead based on whether there are implausible claims, disclaimers and based on quality indicator phrasing. (see assess_quality method)
3. **Retreival strategies**
base_retriever.py Base Retreival Class with abstract methods for two different types of retreivers.
   a. **ChromaDB based Retreival:**
 - Uses ChromaDB to store chunks persisted on LocalDB and indexed once. Embeddings are handled directly by ChromaDB.
 - Retrieved based on query embedding cosine similarity with stored embeddings. Also added a where clause to provide age based filtering where age is part of the query. Top 3 results are retreived and scored.
   b. **Hybrid Retrieval**
 - Uses BM25 (keyword-based) and in-memory embedding using all-MiniLM-L6-v2 model and then ranks them with weighted composite scoring - 0.2 for BM25 and 0.8 for in-memory embeddings. Top 3 results are retreived and scored.
4. **Comparison of the two strategies**
tests/compare_strategies.py Hybrid is faster, but Embeddings is more accurate. See more about comparisons below.

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

   Download a GGUF model (I used Llama-3.2-3B):
   ```bash
   # Create models directory if not exists
   mkdir -p data/models

   # Download model (example using wget or curl)
   wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O data/models/llama-3.2-3b.gguf

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
python src/main.py --strategy embedding  <question> # or hybrid 

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

### Running Tests

```bash
pytest tests/
```

## Retrieval Strategy Comparison

### Strategy 1: ChromaDB Embedding-Based Retrieval
**Description**: Uses ChromaDB with sentence-transformers for persistent vector storage and semantic retrieval with age-aware metadata filtering.

**Implementation**:
- Storage: ChromaDB with local persistence (`./data/chroma_db`)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Similarity metric: Cosine similarity
- Age filtering: Numeric min_age/max_age metadata with WHERE clauses
- Confidence threshold: 0.3
- Index-time filtering: Removes documents with disclaimers or implausible claims

**Pros**:
- **Excellent age-range matching** with metadata filtering (e.g., "2 month old" → only 0-2, 2-4 month docs)
- Persistent storage - no re-indexing between sessions
- More reliable source retrieval (finds correct milestone documents)
- Effective safety filtering (removes 3/13 noisy documents at index-time)
- Semantic understanding captures intent beyond keywords

**Cons**:
- Retrieval time: ~300ms (includes embedding + filtering)
- Requires disk space for vector storage
- Conservative confidence scoring (appropriate for safety-critical domain)

### Strategy 2: Hybrid Retrieval (BM25 + Embeddings)
**Description**: Combines traditional keyword-based retrieval (BM25) with dense embeddings, using weighted score fusion optimized for conversational queries.

**Implementation**:
- BM25 for keyword matching (weight: 0.2)
- Embeddings for semantic similarity (weight: 0.8)
- Score fusion: `0.2 × BM25 + 0.8 × embedding`
- Age filtering: Filters tokenized corpus and embeddings by age metadata
- Confidence threshold: 0.25 (lower than embedding-only due to BM25 contributing less)
- Index-time filtering: Same as ChromaDB strategy

**Pros**:
- Fast retrieval speed (~250ms with age filtering)
- Semantic search with keyword boost when available
- Age-aware filtering using shared utilities

**Cons**:
- **BM25 often contributes zero** for conversational queries (e.g., "what do 2 month olds do"). This can be improved by adding a word conversion layer.
  - Number/word mismatch: "2" vs "two" 
  - Different phrasing: "month olds" vs "months" or "infancy"
- Lower scores than pure embedding (0.27 vs 0.54) before weight adjustment
- Increased complexity without significant benefit for this use case
- Weights heavily favor embeddings (0.8) to compensate for BM25 limitations

## Safety & Confidence Features

### Index-Time Content Filtering

The system filters documents **before indexing** to ensure only high-quality content enters the vector database:

**Filtered Content** (`filter_before_indexing()`):
- **Disclaimers**: Documents with warning text like "these statements differ from pediatric guidance"
- **Implausible Claims**: Medically impossible milestones (e.g., "walking at 4 months", "reading in infancy")
- **Low Quality**: Documents scoring below 0.4 quality threshold based on:
  - Quality indicators (±0.3): "typically", "pediatric", "developmental milestone"
  - Disclaimers (−0.4 penalty)
  - Implausible claims (−0.5 penalty)
  - Age metadata presence (+0.1 bonus)

### Two-Tier Confidence System

The system uses a two-stage approach to prevent hallucination while informing users:

**Stage 1: Gate Check** (`should_generate_answer()`)
- Checks if **top retrieval score** meets minimum threshold
- Thresholds: 0.3 (embedding/chroma), 0.25 (hybrid)
- Below threshold → Show fallback message instead of generating
- Prevents LLM from hallucinating when retrieval is poor

**Stage 2: Confidence Labeling** (`calculate_confidence()`)
- Calculates **average** of all retrieval scores
- Thresholds: >0.6 = "high", >0.4 = "medium", else "low"
- Displayed to user alongside answer
- Strategy-aware: Hybrid uses 0.9× thresholds (scores tend lower)

**Fallback Responses** (when gate check fails):
```
"I don't have specific information about that in my knowledge base.
Please try asking about developmental milestones for children aged 0-36 months."
"You can also direct your questions to a pediatrician."
```

## Comparison with 5 sample questions and verdict on each result for both strategies
[Comparison and Verdict for 5 sample questions](Comparison_and_Verdict_for_five_sample_questions.txt)

## Notes on AI Usage
I used Claude Code for this project. Initial prompt was the project specification that was provided; however, I made many improvements along the way, as can be seen in the commit history. 
- Its initial choice of the Tiny Llama model was not good, as it was making a lot of spelling mistakes and giving very short and sometimes unrelated answers. I switched the model to Llama 3.2.
- Noisiness was being detected by the file name, but it should be detected by the content instead. I added a content filter which looks at things like implausible claims, keywords, etc.
- I suggested the use of ChromaDB for the embeddings approach so that the indexing can be preserved across sessions.
- I added explicit age-based filtering, which improved the results greatly.
- With a lot of repeated code in both the strategies, I asked it to refactor and extract it into utils. Moved all tests to dedicated test directory.
- The initial implementation, was doing query time filtering, which I felt is unnecessary because the noisy documents should not even be persisted in the database. So I requested it to change it to index time filtering instead.
- I ran into an issue where Llama 3.2 was extremely slow on my Mac, so I wrote a benchmark and also enabled GPU acceleration to improve its performance.
- I made sure that constants are not hard-coded and unused code and empty directories are removed. 

That being said, Claude is an excellent partner in coding, and I use it very heavily for all projects. It also acts as a brainstorming partner and teacher. However, it is important to check its work as it tends to produce too much redundant code which could affect performance. I also need to continually make sure to whet its design choices, break down work into digestable chunks and add sufficient tests. 

## Future Improvements

- [ ] Expand age range coverage (36-60 months for preschool)
- [ ] Web interface with visual milestone tracking
- [ ] Mobile app version
- [ ] Stronger test suite and eval framework
- [ ] Try other models and ranking/merging strategies.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with open-source models and libraries


THANK YOU!
