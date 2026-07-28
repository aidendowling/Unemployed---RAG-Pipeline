"""Ingestion pipeline module for loading raw data into DuckDB."""

from .pipeline import (
    IngestionResult,
    discover_source_files,
    ingest_directory,
    ingest_file,
    infer_table_name,
    read_source_file,
)
from .fred import (
    FredAPIError,
    FredDownloadResult,
    fetch_fred_series,
    save_fred_series,
    save_fred_series_batch,
)

__all__ = [
    "IngestionResult",
    "discover_source_files",
    "ingest_directory",
    "ingest_file",
    "infer_table_name",
    "read_source_file",
    "FredAPIError",
    "FredDownloadResult",
    "fetch_fred_series",
    "save_fred_series",
    "save_fred_series_batch",
]
