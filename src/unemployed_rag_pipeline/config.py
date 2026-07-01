"""Configuration settings for the Unemployed RAG Pipeline."""

from pathlib import Path

# Data paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

# Database configuration
DEFAULT_DB_PATH = PROCESSED_DATA_DIR / "unemployed_rag.duckdb"

# Embedding configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

# Retrieval configuration
DEFAULT_COLLECTION_NAME = "labor_market"
BM25_MIN_TERM_FREQUENCY = 1  # Include all terms

# File format configuration
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet"}
