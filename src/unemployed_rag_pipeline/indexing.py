"""Index building orchestration for creating retrieval artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer

from .retrieval.keyword import tokenize


@dataclass(frozen=True, slots=True)
class IndexedRow:
    """Text representation of one database row."""

    id: str
    table_name: str
    row_number: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Summary of a retrieval index build."""

    db_path: Path
    index_dir: Path
    collection_name: str
    bm25_path: Path
    table_count: int
    document_count: int


def _sanitize_identifier(value: str) -> str:
    """Sanitize a string to be a valid identifier."""
    import re

    return re.sub(r"[^a-zA-Z0-9_]+", "_", value)


def list_tables(db_path: Path) -> list[str]:
    """Return the table names stored in a DuckDB database.

    Args:
        db_path: Path to the DuckDB database

    Returns:
        List of table names
    """
    if not db_path.exists():
        return []

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = connection.execute("SHOW TABLES").fetchall()
        return [row[0] for row in rows]
    finally:
        connection.close()


def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load one table from DuckDB into a DataFrame.

    Args:
        db_path: Path to the DuckDB database
        table_name: Name of the table to load

    Returns:
        DataFrame containing the table data
    """
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        return connection.execute(f'SELECT * FROM "{table_name}"').df()
    finally:
        connection.close()


def row_to_text(row: pd.Series) -> str:
    """Turn a row into a flat text document for retrieval.

    Args:
        row: A pandas Series representing a database row

    Returns:
        Formatted text representation of the row
    """
    parts: list[str] = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def dataframe_to_rows(table_name: str, frame: pd.DataFrame) -> list[IndexedRow]:
    """Convert a DataFrame into retrieval-ready row documents.

    Args:
        table_name: Name of the source table
        frame: DataFrame to convert

    Returns:
        List of IndexedRow objects
    """
    records: list[IndexedRow] = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        text = row_to_text(row)
        metadata = {
            "table_name": table_name,
            "row_number": row_number,
        }
        records.append(
            IndexedRow(
                id=f"{_sanitize_identifier(table_name)}_{row_number}",
                table_name=table_name,
                row_number=row_number,
                text=text,
                metadata=metadata,
            )
        )
    return records


def build_indexes(
    db_path: Path,
    index_dir: Path,
    collection_name: str = "labor_market",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    clear_existing: bool = False,
) -> IndexBuildResult:
    """Build Chroma and BM25 retrieval artifacts from all DuckDB tables.

    Args:
        db_path: Path to the DuckDB database
        index_dir: Directory where index artifacts will be written
        collection_name: Name for the Chroma collection
        model_name: HuggingFace SentenceTransformer model identifier
        clear_existing: If True, delete existing index artifacts

    Returns:
        IndexBuildResult with summary statistics
    """
    db_path = db_path.expanduser().resolve()
    index_dir = index_dir.expanduser().resolve()

    if clear_existing and index_dir.exists():
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    tables = list_tables(db_path)
    if not tables:
        return IndexBuildResult(
            db_path=db_path,
            index_dir=index_dir,
            collection_name=collection_name,
            bm25_path=index_dir / "bm25_corpus.json",
            table_count=0,
            document_count=0,
        )

    rows: list[IndexedRow] = []
    for table_name in tables:
        frame = load_table(db_path, table_name)
        rows.extend(dataframe_to_rows(table_name, frame))

    texts = [row.text for row in rows]
    tokenized_corpus = [tokenize(text) for text in texts]

    # Build BM25 index
    bm25_path = index_dir / "bm25_corpus.json"
    bm25_payload = {
        "source_db": str(db_path),
        "tables": tables,
        "documents": [
            {
                "id": row.id,
                "table_name": row.table_name,
                "row_number": row.row_number,
                "text": row.text,
                "metadata": row.metadata,
                "tokens": tokens,
            }
            for row, tokens in zip(rows, tokenized_corpus, strict=True)
        ],
    }
    bm25_path.write_text(json.dumps(bm25_payload, indent=2), encoding="utf-8")

    # Build semantic index with Chroma and embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    chroma_dir = index_dir / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[row.id for row in rows],
        documents=texts,
        embeddings=embeddings,
        metadatas=[row.metadata for row in rows],
    )

    return IndexBuildResult(
        db_path=db_path,
        index_dir=index_dir,
        collection_name=collection_name,
        bm25_path=bm25_path,
        table_count=len(tables),
        document_count=len(rows),
    )