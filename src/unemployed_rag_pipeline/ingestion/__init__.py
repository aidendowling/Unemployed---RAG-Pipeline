"""Ingestion pipeline module for loading raw data into DuckDB."""

from .cps import (
    BUILTIN_LAYOUTS,
    CPSLayout,
    CPSLayoutError,
    FieldSpec,
    find_sidecar_layout,
    load_layout,
    read_cps_dat,
    summarize_cps,
)
from .fred import (
    FredAPIError,
    FredDownloadResult,
    fetch_fred_series,
    save_fred_series,
    save_fred_series_batch,
)
from .http import APIError
from .pipeline import (
    IngestionResult,
    discover_source_files,
    infer_table_name,
    ingest_directory,
    ingest_file,
    read_source_file,
)
from .qwi import (
    DEFAULT_VARIABLES,
    QWIAPIError,
    QWIDownloadResult,
    fetch_qwi_data,
    save_qwi_data,
)

__all__ = [
    "APIError",
    "BUILTIN_LAYOUTS",
    "CPSLayout",
    "CPSLayoutError",
    "DEFAULT_VARIABLES",
    "FieldSpec",
    "FredAPIError",
    "FredDownloadResult",
    "IngestionResult",
    "QWIAPIError",
    "QWIDownloadResult",
    "discover_source_files",
    "fetch_fred_series",
    "fetch_qwi_data",
    "find_sidecar_layout",
    "infer_table_name",
    "ingest_directory",
    "ingest_file",
    "load_layout",
    "read_cps_dat",
    "read_source_file",
    "save_fred_series",
    "save_fred_series_batch",
    "save_qwi_data",
    "summarize_cps",
]
