"""Ingestion pipeline module for loading raw data into DuckDB."""

from .pipeline import (
    IngestionResult,
    discover_source_files,
    ingest_directory,
    ingest_file,
    infer_table_name,
    read_source_file,
)

__all__ = [
    "IngestionResult",
    "discover_source_files",
    "ingest_directory",
    "ingest_file",
    "infer_table_name",
    "read_source_file",
]
