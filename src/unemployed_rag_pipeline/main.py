"""Command-line entry point for the Unemployed RAG Pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ingestion import ingest_directory
from .indexing import build_indexes
from . import config
from .retrieval.keyword import BM25Retriever
from .retrieval.semantic import SemanticRetriever
from .obsolescence import freshness
from .rag import RAGOrchestrator


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

        orchestrator = RAGOrchestrator(
            [r for r in (semantic, bm25) if r is not None],
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
