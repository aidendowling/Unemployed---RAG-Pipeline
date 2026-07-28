"""Command-line entry point for the Unemployed RAG Pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ingestion import (
    ingest_directory,
    ingest_file,
    save_fred_series,
    save_fred_series_batch,
)
from .indexing import build_indexes
from . import config
from .retrieval.keyword import BM25Retriever
from .retrieval.semantic import SemanticRetriever
from .obsolescence import freshness
from .rag import RAGOrchestrator, OpenRouterGenerator, DefaultGenerator


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
        default=config.RAW_DATA_DIR,
        help="Directory containing CSV, TSV, TXT, or Parquet source files.",
    )
    ingest_parser.add_argument(
        "--db-path",
        type=Path,
        default=config.DEFAULT_DB_PATH,
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
        default=config.DEFAULT_DB_PATH,
        help="Path to the DuckDB database file created by ingestion.",
    )
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.INDEXES_DIR,
        help="Directory where retrieval artifacts will be written.",
    )
    index_parser.add_argument(
        "--collection-name",
        default=config.DEFAULT_COLLECTION_NAME,
        help="Chroma collection name to use for vector storage.",
    )
    index_parser.add_argument(
        "--model-name",
        default=config.DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used to create embeddings.",
    )
    index_parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing index artifacts before rebuilding.",
    )

    fred_parser = subparsers.add_parser(
        "fred",
        help="Download a FRED series to CSV and optionally ingest it.",
    )
    fred_parser.add_argument(
        "--series-id",
        required=False,
        help="FRED series ID to download, e.g. UNRATE or CAUR. Mutually exclusive with --series-ids.",
    )
    fred_parser.add_argument(
        "--series-ids",
        type=str,
        default=None,
        help="Comma-separated list of FRED series IDs to download in batch.",
    )
    fred_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Directory where the downloaded CSV will be saved.",
    )
    fred_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date in YYYY-MM-DD format.",
    )
    fred_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date in YYYY-MM-DD format.",
    )
    fred_parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Optional explicit output filename (defaults to a generated CSV name).",
    )
    fred_parser.add_argument(
        "--api-key",
        type=str,
        default=config.FRED_API_KEY,
        help="FRED API key. Defaults to FRED_API_KEY from .env.",
    )
    fred_parser.add_argument(
        "--db-path",
        type=Path,
        default=config.DEFAULT_DB_PATH,
        help="Optional DuckDB path if you want to ingest the downloaded file immediately.",
    )
    fred_parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest the downloaded FRED CSV into DuckDB after saving it.",
    )

    qa_parser = subparsers.add_parser(
        "qa",
        help="Run a single-question or interactive QA loop against built indexes.",
    )
    qa_parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.INDEXES_DIR,
        help="Directory where retrieval artifacts are stored.",
    )
    qa_parser.add_argument(
        "--collection-name",
        default=config.DEFAULT_COLLECTION_NAME,
        help="Chroma collection name to use for semantic retrieval.",
    )
    qa_parser.add_argument(
        "--bm25-path",
        type=Path,
        default=config.INDEXES_DIR / "bm25_corpus.json",
        help="Path to BM25 corpus JSON produced by indexing.",
    )
    qa_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per retriever.",
    )
    qa_parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="If provided, run a single query and exit. Otherwise interactive loop.",
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

    if args.command == "fred":
        # Support batch downloads via --series-ids (comma-separated)
        if args.series_ids:
            series_list = [s.strip() for s in args.series_ids.split(",") if s.strip()]
            results = save_fred_series_batch(
                series_list,
                output_dir=args.output_dir,
                api_key=args.api_key,
                observation_start=args.start_date,
                observation_end=args.end_date,
            )

            if not results:
                print("No series were downloaded.")
                return

            for result in results:
                if result.row_count == 0:
                    print(f"No observations returned for FRED series {result.series_id}.")
                    continue
                print(
                    f"Downloaded FRED series {result.series_id} with {result.row_count} row(s) "
                    f"to {result.output_path} ({result.start_date} → {result.end_date})."
                )

            if args.ingest:
                from duckdb import connect

                connection = connect(str(args.db_path.expanduser().resolve()))
                try:
                    for result in results:
                        if result.row_count == 0:
                            continue
                        ingest_result = ingest_file(connection, result.output_path)
                        print(
                            f"Ingested {ingest_result.row_count} row(s) into {args.db_path}: "
                            f"{ingest_result.table_name}"
                        )
                finally:
                    connection.close()

            return

        # Single-series download (backwards compatible)
        result = save_fred_series(
            series_id=args.series_id,
            output_dir=args.output_dir,
            api_key=args.api_key,
            observation_start=args.start_date,
            observation_end=args.end_date,
            file_name=args.file_name,
        )

        if result.row_count == 0:
            print(f"No observations returned for FRED series {args.series_id}.")
            return

        print(
            f"Downloaded FRED series {result.series_id} with {result.row_count} row(s) "
            f"to {result.output_path} ({result.start_date} → {result.end_date})."
        )

        if args.ingest:
            from duckdb import connect

            connection = connect(str(args.db_path.expanduser().resolve()))
            try:
                ingest_result = ingest_file(connection, result.output_path)
            finally:
                connection.close()

            print(
                f"Ingested {ingest_result.row_count} row(s) into {args.db_path}: "
                f"{ingest_result.table_name}"
            )
        return

    if args.command == "qa":
        index_dir = args.index_dir
        collection_name = args.collection_name
        bm25_path = args.bm25_path
        k = args.k

        # Instantiate retrievers
        try:
            semantic = SemanticRetriever(index_dir=index_dir, collection_name=collection_name)
        except Exception as e:
            print(f"Failed to initialize SemanticRetriever: {e}")
            return

        try:
            bm25 = BM25Retriever(bm25_path=bm25_path)
        except Exception as e:
            print(f"Failed to initialize BM25Retriever: {e}")
            bm25 = None

        # Instantiate generator: use OpenRouter if API key available, else use stub
        generator = None
        if config.OPENROUTER_API_KEY:
            try:
                generator = OpenRouterGenerator(
                    api_key=config.OPENROUTER_API_KEY,
                    model=config.OPENROUTER_MODEL,
                    base_url=config.OPENROUTER_BASE_URL,
                )
                print(f"Using OpenRouter generator: {config.OPENROUTER_MODEL}")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenRouterGenerator: {e}")
                print("Falling back to DefaultGenerator (stub).")
                generator = DefaultGenerator()
        else:
            print("No OPENROUTER_API_KEY found. Using DefaultGenerator (stub).")
            generator = DefaultGenerator()

        orchestrator = RAGOrchestrator(
            [r for r in (semantic, bm25) if r is not None],
            generator=generator,
            obsolescence_filter=lambda r: not freshness.is_obsolete(r.get("metadata")),
        )

        def run_query(q: str):
            out = orchestrator.answer(q, k=k)
            print("\n--- ANSWER ---\n")
            print(out.get("answer"))
            print("\n--- SOURCES ---")
            for s in out.get("sources", []):
                print(f"- {s.get('id')}")

        if args.query:
            run_query(args.query)
            return

        # Interactive loop
        print("Entering interactive QA mode. Type 'exit' or Ctrl-D to quit.")
        try:
            while True:
                q = input("Query> ").strip()
                if not q:
                    continue
                if q.lower() in {"exit", "quit"}:
                    break
                run_query(q)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
