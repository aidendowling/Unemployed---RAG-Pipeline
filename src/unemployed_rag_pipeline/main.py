"""Command-line entry point for the Unemployed RAG Pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ingestion import ingest_directory
from .indexing import build_indexes


def main() -> None:
    """Run the command-line application."""
    parser = argparse.ArgumentParser(
        prog="unemployed-rag-pipeline",
        description="Terminal workflow for the Unemployed RAG Pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load raw files into a DuckDB database.",
    )
    ingest_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing CSV, TSV, TXT, or Parquet source files.",
    )
    ingest_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/processed/unemployed_rag.duckdb"),
        help="Path to the DuckDB database file to create or update.",
    )
    ingest_parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the database file before loading new data.",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Build retrieval indexes from the DuckDB database.",
    )
    index_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/processed/unemployed_rag.duckdb"),
        help="Path to the DuckDB database file created by ingestion.",
    )
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes"),
        help="Directory where retrieval artifacts will be written.",
    )
    index_parser.add_argument(
        "--collection-name",
        default="labor_market",
        help="Chroma collection name to use for vector storage.",
    )
    index_parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used to create embeddings.",
    )
    index_parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing index artifacts before rebuilding.",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        results = ingest_directory(
            raw_dir=args.raw_dir,
            db_path=args.db_path,
            clear_existing=args.clear,
        )
        if not results:
            print(f"No supported data files found in {args.raw_dir}.")
            return

        print(f"Loaded {len(results)} file(s) into {args.db_path}:")
        for result in results:
            print(
                f"- {result.file_name} -> {result.table_name} "
                f"({result.row_count} rows, {len(result.columns)} columns)"
            )
        return

    if args.command == "index":
        result = build_indexes(
            db_path=args.db_path,
            index_dir=args.index_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            clear_existing=args.clear,
        )
        if result.table_count == 0:
            print(f"No tables found in {args.db_path} to index.")
            return

        print(
            f"Indexed {result.document_count} row(s) from {result.table_count} table(s) "
            f"into {result.index_dir}."
        )
        print(f"Chroma collection: {result.collection_name}")
        print(f"BM25 corpus saved: {result.bm25_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
