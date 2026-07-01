"""Data ingestion pipeline for loading raw files into DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import duckdb
import pandas as pd

from ..config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a single file ingestion."""

    file_path: Path
    table_name: str
    row_count: int
    columns: tuple[str, ...]

    @property
    def file_name(self) -> str:
        """Return the filename without its parent directory."""
        return self.file_path.name


def discover_source_files(raw_dir: Path) -> list[Path]:
    """Return supported files in a raw-data directory.

    Args:
        raw_dir: Directory to search for supported data files

    Returns:
        Sorted list of supported file paths
    """
    if not raw_dir.exists():
        return []

    return sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def infer_table_name(file_path: Path) -> str:
    """Convert a source filename into a safe DuckDB table name.

    Args:
        file_path: Path to the source file

    Returns:
        Safe table name for DuckDB
    """
    stem = file_path.stem.lower()
    table_name = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not table_name:
        table_name = "dataset"
    if table_name[0].isdigit():
        table_name = f"dataset_{table_name}"
    return table_name


def read_source_file(file_path: Path) -> pd.DataFrame:
    """Load a source file into a DataFrame.

    Args:
        file_path: Path to the source file

    Returns:
        DataFrame loaded from the file

    Raises:
        ValueError: If file format is not supported
    """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def ingest_file(connection: duckdb.DuckDBPyConnection, file_path: Path) -> IngestionResult:
    """Ingest one source file into DuckDB and return a summary.

    Args:
        connection: DuckDB connection
        file_path: Path to the source file

    Returns:
        IngestionResult with file statistics
    """
    table_name = infer_table_name(file_path)
    frame = read_source_file(file_path)
    connection.register("source_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM source_frame")
    connection.unregister("source_frame")

    return IngestionResult(
        file_path=file_path,
        table_name=table_name,
        row_count=int(len(frame)),
        columns=tuple(frame.columns.astype(str).tolist()),
    )


def ingest_directory(raw_dir: Path, db_path: Path, clear_existing: bool = False) -> list[IngestionResult]:
    """Load every supported file from raw_dir into the DuckDB database at db_path.

    Args:
        raw_dir: Directory containing source files
        db_path: Path to the DuckDB database file
        clear_existing: If True, delete existing database before loading

    Returns:
        List of IngestionResult objects for each loaded file
    """
    raw_dir = raw_dir.expanduser().resolve()
    db_path = db_path.expanduser().resolve()

    if clear_existing and db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    source_files = discover_source_files(raw_dir)
    if not source_files:
        return []

    connection = duckdb.connect(str(db_path))
    try:
        results: list[IngestionResult] = []
        for file_path in source_files:
            try:
                result = ingest_file(connection, file_path)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to ingest {file_path.name}: {e}")
        return results
    finally:
        connection.close()
