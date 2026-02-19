"""
Centralized configuration for the Child Development Q&A system.

All magic numbers, thresholds, and configurable parameters are defined here
with clear documentation.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

class LLMConfig:
    """Configuration for local LLM generation."""

    # Model settings
    DEFAULT_MODEL_PATH = str(MODELS_DIR / "llama-3.2-3b.gguf")
    CONTEXT_WINDOW = 2048  # n_ctx: Maximum context size
    NUM_THREADS = 4  # Number of CPU threads for inference

    # Generation parameters
    MAX_TOKENS = 200  # GPU-accelerated generation; 200 tokens fits 2-4 complete sentences
    TEMPERATURE = 0.3  # Low temperature for factual, deterministic answers

    # Temperature guidelines:
    # 0.0-0.3: Factual, deterministic (recommended for Q&A)
    # 0.4-0.7: Balanced creativity
    # 0.8-1.0: Creative, diverse responses


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

class RetrievalConfig:
    """Configuration for retrieval strategies."""

    # Common settings
    DEFAULT_TOP_K = 3  # Number of chunks to retrieve per query
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-transformer model
    EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 embeddings

    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD_EMBEDDING = 0.3  # Minimum score to proceed with generation (embedding strategy)
    LOW_CONFIDENCE_THRESHOLD_HYBRID = 0.25  # Hybrid scores tend to be lower
    HIGH_CONFIDENCE_THRESHOLD = 0.6  # Score for "high confidence" label
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4  # Score for "medium confidence" label

    # Hybrid retriever weights (must sum to 1.0)
    HYBRID_BM25_WEIGHT = 0.2  # Weight for keyword-based BM25 scoring
    HYBRID_EMBEDDING_WEIGHT = 0.8  # Weight for semantic embedding scoring

    # ChromaDB settings
    CHROMA_COLLECTION_NAME = "child_development"
    CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)
    CHROMA_DISTANCE_METRIC = "cosine"  # Distance metric for vector search


# ============================================================================
# SAFETY & QUALITY CONFIGURATION
# ============================================================================

class SafetyConfig:
    """Configuration for content filtering and safety checks."""

    # Index-time filtering
    QUALITY_THRESHOLD = 0.4  # Minimum quality score (0-1) for documents

    # Filter settings
    FILTER_DISCLAIMERS = True  # Remove docs with disclaimers
    FILTER_IMPLAUSIBLE_CLAIMS = True  # Remove medically implausible content
    STRICT_MODE = False  # More aggressive filtering if True


# ============================================================================
# AGE FILTERING CONFIGURATION
# ============================================================================

class AgeConfig:
    """Configuration for age-based filtering."""

    # Age range boundaries (in months)
    MIN_AGE = 0  # Newborn
    MAX_AGE = 36  # 3 years old

    # Keyword mappings
    AGE_KEYWORDS = {
        "newborn": 0,
        "infant": 3,
        "baby": 6,
        "toddler": 12,
    }


# ============================================================================
# UI/UX CONFIGURATION
# ============================================================================

class UIConfig:
    """Configuration for user interface and display."""

    # Display settings
    SHOW_RETRIEVAL_TIME = True
    SHOW_GENERATION_TIME = True
    SHOW_CONFIDENCE_LEVEL = True
    SHOW_SOURCE_SCORES = True

    # Fallback messages
    LOW_CONFIDENCE_MESSAGE = (
        "I don't have specific information about that in my knowledge base. "
        "Please try asking about developmental milestones for children aged 0-36 months."
        "You can also direct your questions to a pediatrician."
    )

    NO_AGE_MATCH_MESSAGE = (
        "I don't have information for that specific age in my knowledge base. "
        "I can help with developmental milestones for children aged 0-36 months."
        "You can also direct your questions to a pediatrician."
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration values."""
    # Check hybrid weights sum to 1.0
    total_weight = RetrievalConfig.HYBRID_BM25_WEIGHT + RetrievalConfig.HYBRID_EMBEDDING_WEIGHT
    assert abs(total_weight - 1.0) < 0.01, f"Hybrid weights must sum to 1.0, got {total_weight}"

    # Check thresholds are in valid range
    assert 0 <= RetrievalConfig.LOW_CONFIDENCE_THRESHOLD_EMBEDDING <= 1, "Low confidence threshold must be 0-1"
    assert 0 <= RetrievalConfig.HIGH_CONFIDENCE_THRESHOLD <= 1, "High confidence threshold must be 0-1"
    assert 0 <= RetrievalConfig.MEDIUM_CONFIDENCE_THRESHOLD <= 1, "Medium confidence threshold must be 0-1"

    # Check threshold ordering
    assert (
        RetrievalConfig.LOW_CONFIDENCE_THRESHOLD_EMBEDDING
        <= RetrievalConfig.MEDIUM_CONFIDENCE_THRESHOLD
        <= RetrievalConfig.HIGH_CONFIDENCE_THRESHOLD
    ), "Thresholds must be in ascending order: low <= medium <= high"

    print("✓ Configuration validated successfully")


# Run validation on import
validate_config()
