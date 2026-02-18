# Configuration Refactoring Summary

## Overview
Refactored codebase to eliminate magic numbers and centralize configuration management.

## Changes Made

### 1. Created Central Configuration Module
**File:** `src/config.py`

Organized all configuration values into logical groups:

#### **LLMConfig** - Language Model Settings
- `DEFAULT_MODEL_PATH` = `"data/models/llama-3.2-3b.gguf"`
- `CONTEXT_WINDOW` = `2048` (n_ctx)
- `NUM_THREADS` = `4`
- `MAX_TOKENS` = `75` (optimized for speed)
- `TEMPERATURE` = `0.3` (factual, deterministic)

#### **RetrievalConfig** - Retrieval Settings
- `DEFAULT_TOP_K` = `3` (chunks per query)
- `EMBEDDING_MODEL` = `"all-MiniLM-L6-v2"`
- `LOW_CONFIDENCE_THRESHOLD` = `0.3` (embedding strategy)
- `LOW_CONFIDENCE_THRESHOLD_HYBRID` = `0.25` (hybrid scores tend to be lower)
- `HIGH_CONFIDENCE_THRESHOLD` = `0.6`
- `MEDIUM_CONFIDENCE_THRESHOLD` = `0.4`
- `HYBRID_BM25_WEIGHT` = `0.2` (20% keyword)
- `HYBRID_EMBEDDING_WEIGHT` = `0.8` (80% semantic)
- `CHROMA_COLLECTION_NAME` = `"child_development"`
- `CHROMA_PERSIST_DIR` = `"./data/chroma_db"`

#### **SafetyConfig** - Content Filtering
- `ENABLE_SAFETY_FILTER` = `True`
- `QUALITY_THRESHOLD` = `0.4`
- `FILTER_DISCLAIMERS` = `True`
- `FILTER_IMPLAUSIBLE_CLAIMS` = `True`

#### **UIConfig** - User Interface
- `LOW_CONFIDENCE_MESSAGE` - Fallback message for low confidence
- `NO_AGE_MATCH_MESSAGE` - Message when no age matches found

### 2. Updated Files to Use Config

#### **src/main.py**
**Before:**
```python
def __init__(self, model_path: str = "data/models/llama-3.2-3b.gguf", strategy: str = "embedding"):
    ...
    self.retriever = ChromaRetriever(
        collection_name="child_development",
        persist_dir="./data/chroma_db"
    )
    ...
    if scores[0] < 0.3:  # Magic number!
        return {'answer': "I don't have specific information..."}
    ...
    answer = self.llm.generate_answer(question, context, max_tokens=75)
    if avg_score > 0.6:
        confidence = 'high'
```

**After:**
```python
from src.config import LLMConfig, RetrievalConfig, UIConfig

def __init__(self, model_path: str = None, strategy: str = "embedding"):
    if model_path is None:
        model_path = LLMConfig.DEFAULT_MODEL_PATH
    ...
    self.retriever = ChromaRetriever(
        collection_name=RetrievalConfig.CHROMA_COLLECTION_NAME,
        persist_dir=RetrievalConfig.CHROMA_PERSIST_DIR
    )
    ...
    threshold = (
        RetrievalConfig.LOW_CONFIDENCE_THRESHOLD_HYBRID
        if self.strategy == "hybrid"
        else RetrievalConfig.LOW_CONFIDENCE_THRESHOLD
    )
    if scores[0] < threshold:
        return {'answer': UIConfig.LOW_CONFIDENCE_MESSAGE}
    ...
    answer = self.llm.generate_answer(question, context, max_tokens=LLMConfig.MAX_TOKENS)
    if avg_score > RetrievalConfig.HIGH_CONFIDENCE_THRESHOLD:
        confidence = 'high'
```

#### **src/retrieval/hybrid_retriever.py**
**Before:**
```python
def __init__(self,
             model_name: str = "all-MiniLM-L6-v2",
             bm25_weight: float = 0.2,
             embedding_weight: float = 0.8,
             use_safety_filter: bool = True):
    ...

def index_documents(self, chunks: List[Dict], quality_threshold: float = 0.4):
    ...

def retrieve(self, query: str, top_k: int = 3):
    ...
```

**After:**
```python
from src.config import RetrievalConfig, SafetyConfig

def __init__(self,
             model_name: str = None,
             bm25_weight: float = None,
             embedding_weight: float = None,
             use_safety_filter: bool = None):
    if model_name is None:
        model_name = RetrievalConfig.EMBEDDING_MODEL
    if bm25_weight is None:
        bm25_weight = RetrievalConfig.HYBRID_BM25_WEIGHT
    # ... etc

def index_documents(self, chunks: List[Dict], quality_threshold: float = None):
    if quality_threshold is None:
        quality_threshold = SafetyConfig.QUALITY_THRESHOLD
    ...

def retrieve(self, query: str, top_k: int = None):
    if top_k is None:
        top_k = RetrievalConfig.DEFAULT_TOP_K
    ...
```

#### **src/retrieval/chroma_retriever.py**
Similar updates:
- Uses `RetrievalConfig` for defaults
- Uses `SafetyConfig` for quality threshold
- All parameters default to config values

#### **src/generation/llm.py**
**Before:**
```python
def __init__(self, model_path: str = "data/models/llama-3.2-3b.gguf",
             n_ctx: int = 2048, n_threads: int = 4):
    ...

def generate_answer(self, question: str, context: str, max_tokens: int = 256):
    response = self.generate(prompt, max_tokens=max_tokens, temperature=0.3)
```

**After:**
```python
from src.config import LLMConfig

def __init__(self, model_path: str = None, n_ctx: int = None, n_threads: int = None):
    if model_path is None:
        model_path = LLMConfig.DEFAULT_MODEL_PATH
    if n_ctx is None:
        n_ctx = LLMConfig.CONTEXT_WINDOW
    # ... etc

def generate_answer(self, question: str, context: str, max_tokens: int = None):
    if max_tokens is None:
        max_tokens = LLMConfig.MAX_TOKENS
    response = self.generate(prompt, max_tokens=max_tokens, temperature=LLMConfig.TEMPERATURE)
```

## Benefits

### ✅ No More Magic Numbers
All hardcoded values are now named constants with clear meanings:
- `0.3` → `LOW_CONFIDENCE_THRESHOLD`
- `0.6` → `HIGH_CONFIDENCE_THRESHOLD`
- `75` → `MAX_TOKENS` (with comment about speed optimization)
- `0.2/0.8` → `HYBRID_BM25_WEIGHT / HYBRID_EMBEDDING_WEIGHT`

### ✅ Single Source of Truth
Change a value once in `config.py`, affects entire system:
```python
# Want to experiment with different thresholds?
LOW_CONFIDENCE_THRESHOLD = 0.25  # Change here only!
```

### ✅ Self-Documenting
Each config value has:
- Descriptive name
- Clear comment explaining purpose
- Grouped with related settings

### ✅ Validation on Import
Config validates itself automatically:
```python
def validate_config():
    # Check hybrid weights sum to 1.0
    # Check thresholds are 0-1
    # Check threshold ordering
```

### ✅ Easy Experimentation
Try different configurations without touching business logic:
```python
# Experiment with keyword-heavy hybrid
HYBRID_BM25_WEIGHT = 0.7
HYBRID_EMBEDDING_WEIGHT = 0.3
```

### ✅ IDE Support
- Type hints preserved
- Autocomplete works
- Easy to discover available settings

## Usage Examples

### Import Config
```python
from src.config import LLMConfig, RetrievalConfig, SafetyConfig, UIConfig
```

### Use Defaults
```python
# All parameters now optional - use config defaults
qa_system = ChildDevelopmentQA()  # Uses config for everything
retriever = HybridRetriever()    # Uses config defaults
llm = LocalLLM()                  # Uses config defaults
```

### Override Specific Values
```python
# Override only what you need
qa_system = ChildDevelopmentQA(strategy="hybrid")
retriever = HybridRetriever(bm25_weight=0.5, embedding_weight=0.5)
llm = LocalLLM(max_tokens=150)
```

## Files Modified
1. `src/config.py` (NEW) - Central configuration
2. `src/main.py` - Uses config for thresholds, paths, messages
3. `src/generation/llm.py` - Uses config for LLM parameters
4. `src/retrieval/hybrid_retriever.py` - Uses config for weights, defaults
5. `src/retrieval/chroma_retriever.py` - Uses config for paths, thresholds

## Testing
```bash
# Validate config loads correctly
python3 -c "from src.config import *"
# Output: ✓ Configuration validated successfully

# Run system with config defaults
python src/main.py

# Override specific values
python src/main.py --strategy hybrid
```

## Next Steps (Optional)
- Move config to YAML/JSON file for non-programmers
- Add environment variable overrides
- Create different config profiles (dev, prod, test)
- Add config hot-reloading
