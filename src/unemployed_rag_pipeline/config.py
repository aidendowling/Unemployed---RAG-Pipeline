"""Configuration settings for the Unemployed RAG Pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Data paths
DATA_DIR = Path("data")
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = Path(os.getenv("INDEX_DIR", "data/indexes"))

# Database configuration
DEFAULT_DB_PATH = Path(os.getenv("DUCKDB_PATH", "data/processed/unemployed_rag.duckdb"))

# Embedding configuration
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

# Retrieval configuration
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "labor_market")
BM25_MIN_TERM_FREQUENCY = 1  # Include all terms

# File format configuration
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet"}

# OpenRouter / LLM Generation configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# FRED API configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred")

# App configuration
APP_NAME = os.getenv("APP_NAME", "Unemployed RAG Pipeline")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
