"""Command-line entry point for the Unemployed RAG Pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

from . import config
from .evaluation import EvalCase, evaluate_queries, load_eval_cases, ragas_available
from .indexing import build_indexes, list_tables
from .ingestion import (
    CPSLayoutError,
    FredAPIError,
    QWIAPIError,
    ingest_directory,
    ingest_file,
    read_cps_dat,
    save_fred_series,
    save_fred_series_batch,
    save_qwi_data,
    summarize_cps,
)
from .ingestion.qwi import DEFAULT_VARIABLES
from .rag import DefaultGenerator, OpenRouterGenerator, RAGOrchestrator
from .retrieval.hybrid import FUSION_METHODS, HybridRetriever
from .retrieval.keyword import BM25Retriever
from .retrieval.semantic import SemanticRetriever

DEFAULT_EVAL_QUESTIONS = config.EVAL_DIR / "questions.json"


def _warn(message: str) -> None:
    """Print a warning in a consistent, greppable format."""
    print(f"[warning] {message}")


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI value into a clean list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _ingest_paths(db_path: Path, paths: list[Path]) -> None:
    """Ingest already-downloaded files into DuckDB, reporting each table."""
    connection = duckdb.connect(str(db_path.expanduser().resolve()))
    try:
        for path in paths:
            result = ingest_file(connection, path)
            print(
                f"Ingested {result.row_count} row(s) into {db_path}: {result.table_name}"
            )
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser."""
    parser = argparse.ArgumentParser(
        prog="unemployed-rag-pipeline",
        description="Terminal workflow for the Unemployed RAG Pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Load raw files into a DuckDB database."
    )
    ingest_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Directory containing CSV, TSV, TXT, Parquet, or CPS .dat source files.",
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
        "index", help="Build retrieval indexes from the DuckDB database."
    )
    index_parser.add_argument("--db-path", type=Path, default=config.DEFAULT_DB_PATH)
    index_parser.add_argument("--index-dir", type=Path, default=config.INDEXES_DIR)
    index_parser.add_argument("--collection-name", default=config.DEFAULT_COLLECTION_NAME)
    index_parser.add_argument("--model-name", default=config.DEFAULT_EMBEDDING_MODEL)
    index_parser.add_argument(
        "--clear", action="store_true", help="Delete existing index artifacts before rebuilding."
    )

    fred_parser = subparsers.add_parser(
        "fred", help="Download a FRED series to CSV and optionally ingest it."
    )
    fred_parser.add_argument("--series-id", help="FRED series ID, e.g. UNRATE or CAUR.")
    fred_parser.add_argument(
        "--series-ids", help="Comma-separated list of FRED series IDs to download in batch."
    )
    fred_parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    fred_parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    fred_parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    fred_parser.add_argument("--file-name", help="Explicit output filename.")
    fred_parser.add_argument(
        "--api-key", default=config.FRED_API_KEY, help="Defaults to FRED_API_KEY from .env."
    )
    fred_parser.add_argument("--db-path", type=Path, default=config.DEFAULT_DB_PATH)
    fred_parser.add_argument(
        "--ingest", action="store_true", help="Ingest the downloaded CSV into DuckDB."
    )

    qwi_parser = subparsers.add_parser(
        "qwi", help="Download Census Quarterly Workforce Indicators and optionally ingest them."
    )
    qwi_parser.add_argument("--state", required=True, help="Two-digit state FIPS code, e.g. 13.")
    qwi_parser.add_argument(
        "--years", required=True, help="Comma-separated years, e.g. 2019,2020,2021."
    )
    qwi_parser.add_argument("--quarters", default="1,2,3,4", help="Comma-separated quarters 1-4.")
    qwi_parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated QWI variables.",
    )
    qwi_parser.add_argument(
        "--dataset",
        default="sa",
        choices=["sa", "se", "rh"],
        help="QWI dataset: sa (seasonally adjusted), se (sex/education), rh (race/ethnicity).",
    )
    qwi_parser.add_argument("--county", help="Optional three-digit county FIPS code.")
    qwi_parser.add_argument("--industry", help="Optional NAICS sector code, e.g. 23.")
    qwi_parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    qwi_parser.add_argument("--file-name", help="Explicit output filename.")
    qwi_parser.add_argument(
        "--api-key", default=config.CENSUS_API_KEY, help="Defaults to CENSUS_API_KEY from .env."
    )
    qwi_parser.add_argument("--db-path", type=Path, default=config.DEFAULT_DB_PATH)
    qwi_parser.add_argument(
        "--ingest", action="store_true", help="Ingest the downloaded CSV into DuckDB."
    )

    cps_parser = subparsers.add_parser(
        "cps", help="Parse a fixed-width CPS .dat microdata file and optionally ingest it."
    )
    cps_parser.add_argument("--file", type=Path, required=True, help="Path to the CPS .dat file.")
    cps_parser.add_argument(
        "--layout",
        default=config.CPS_LAYOUT_PATH,
        help="Layout to use: an IPUMS .xml DDI, a JSON field list, a CPS data dictionary "
        ".txt, or the builtin name 'cps-basic-monthly'. Auto-detected from a sidecar file "
        "when omitted.",
    )
    cps_parser.add_argument("--max-rows", type=int, help="Only read the first N records.")
    cps_parser.add_argument(
        "--raw",
        action="store_true",
        help="Keep person-level records instead of aggregating to year/month/state.",
    )
    cps_parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    cps_parser.add_argument("--file-name", help="Explicit output CSV filename.")
    cps_parser.add_argument("--db-path", type=Path, default=config.DEFAULT_DB_PATH)
    cps_parser.add_argument(
        "--ingest", action="store_true", help="Ingest the parsed CSV into DuckDB."
    )

    qa_parser = subparsers.add_parser(
        "qa", help="Run a single-question or interactive QA loop against built indexes."
    )
    _add_query_arguments(qa_parser)
    qa_parser.add_argument(
        "--query", help="If provided, run a single query and exit. Otherwise interactive."
    )
    qa_parser.add_argument(
        "--show-context", action="store_true", help="Print the retrieved context passages."
    )
    qa_parser.add_argument("--json", action="store_true", help="Emit the raw response as JSON.")
    qa_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the response as JSON to this file (e.g. data/outputs/answer.json).",
    )

    eval_parser = subparsers.add_parser(
        "eval", help="Score the query engine with RAGAS (or offline heuristics)."
    )
    _add_query_arguments(eval_parser)
    eval_parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_EVAL_QUESTIONS,
        help="JSON file of evaluation questions.",
    )
    eval_parser.add_argument("--query", action="append", help="Evaluate an ad-hoc question.")
    eval_parser.add_argument(
        "--no-ragas", action="store_true", help="Skip RAGAS and report heuristic metrics only."
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=config.EVAL_OUTPUT_DIR / "eval_results.json",
        help="Where to write the JSON evaluation report.",
    )

    status_parser = subparsers.add_parser(
        "status", help="Report database, index, and credential readiness."
    )
    status_parser.add_argument("--db-path", type=Path, default=config.DEFAULT_DB_PATH)
    status_parser.add_argument("--index-dir", type=Path, default=config.INDEXES_DIR)

    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the retrieval/generation flags shared by ``qa`` and ``eval``."""
    parser.add_argument("--index-dir", type=Path, default=config.INDEXES_DIR)
    parser.add_argument("--collection-name", default=config.DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=None,
        help="Path to the BM25 corpus JSON (defaults to <index-dir>/bm25_corpus.json).",
    )
    parser.add_argument("--k", type=int, default=5, help="Documents to retrieve.")
    parser.add_argument(
        "--fusion",
        default=config.HYBRID_FUSION_METHOD,
        choices=list(FUSION_METHODS),
        help="Hybrid fusion strategy for combining BM25 and semantic results.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=config.HYBRID_SEMANTIC_WEIGHT,
        help="Semantic share of the score under weighted fusion (0-1).",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=config.HYBRID_RRF_K, help="Reciprocal-rank-fusion constant."
    )
    parser.add_argument(
        "--min-vintage-score",
        type=float,
        default=config.MIN_VINTAGE_SCORE,
        help="Drop documents whose query-intent vintage score falls below this.",
    )
    parser.add_argument(
        "--no-obsolescence",
        action="store_true",
        help="Disable query-intent vintage scoring and rank on retrieval score alone.",
    )


def build_query_engine(args: argparse.Namespace) -> RAGOrchestrator | None:
    """Assemble the hybrid retriever, generator, and orchestrator from CLI args.

    Returns:
        A configured orchestrator, or None if no retriever could be initialized.
    """
    index_dir: Path = args.index_dir
    bm25_path: Path = args.bm25_path or index_dir / "bm25_corpus.json"

    if not index_dir.exists():
        _warn(
            f"Index directory {index_dir} does not exist. Run 'index' first "
            "(after 'ingest') to build retrieval artifacts."
        )

    retrievers: dict[str, Any] = {}
    try:
        retrievers["semantic"] = SemanticRetriever(
            index_dir=index_dir, collection_name=args.collection_name
        )
    except Exception as error:  # noqa: BLE001 - report and continue with keyword-only retrieval
        _warn(f"Semantic retriever unavailable: {error}")

    if bm25_path.exists():
        try:
            retrievers["bm25"] = BM25Retriever(bm25_path=bm25_path)
        except Exception as error:  # noqa: BLE001 - report and continue with semantic-only
            _warn(f"BM25 retriever unavailable: {error}")
    else:
        _warn(f"BM25 corpus not found at {bm25_path}; keyword retrieval is disabled.")

    if not retrievers:
        print("No retrievers could be initialized. Run 'ingest' then 'index' first.")
        return None
    if len(retrievers) == 1:
        _warn(
            f"Only the '{next(iter(retrievers))}' retriever is available; "
            "results are not hybrid."
        )

    semantic_weight = min(max(args.semantic_weight, 0.0), 1.0)
    hybrid = HybridRetriever(
        retrievers,
        method=args.fusion,
        weights={"semantic": semantic_weight, "bm25": 1.0 - semantic_weight},
        rrf_k=args.rrf_k,
    )

    generator: Any
    if config.OPENROUTER_API_KEY:
        try:
            generator = OpenRouterGenerator(
                api_key=config.OPENROUTER_API_KEY,
                model=config.OPENROUTER_MODEL,
                base_url=config.OPENROUTER_BASE_URL,
            )
            print(f"Generator: OpenRouter ({config.OPENROUTER_MODEL})")
        except Exception as error:  # noqa: BLE001 - fall back to the offline formatter
            _warn(f"Failed to initialize OpenRouter generator: {error}")
            generator = DefaultGenerator()
    else:
        _warn("OPENROUTER_API_KEY is not set; answers are formatted context, not LLM-generated.")
        generator = DefaultGenerator()

    return RAGOrchestrator(
        [hybrid],
        generator=generator,
        vintage_scoring=not args.no_obsolescence,
        min_vintage_score=args.min_vintage_score,
        warn_vintage_score=config.VINTAGE_WARN_SCORE,
    )


def _print_response(response: dict[str, Any], *, show_context: bool) -> None:
    """Render an orchestrator response with provenance and warnings."""
    print("\n--- ANSWER ---\n")
    print(response.get("answer"))

    provenance = response.get("provenance", [])
    print("\n--- SOURCES ---")
    if not provenance:
        print("(none)")
    for item in provenance:
        retrievers = ",".join(item.get("retrievers") or []) or "n/a"
        vintage = item.get("vintage_year")
        dataset = item.get("dataset") or "unknown dataset"
        source_file = item.get("source_file") or "unknown file"
        print(
            f"- {item['id']} | table={item['table']} | {dataset} <- {source_file} | "
            f"vintage={vintage if vintage is not None else 'n/a'} | matched by {retrievers}"
        )
        components = ", ".join(
            f"{name}={score:.4f}" for name, score in (item.get("component_scores") or {}).items()
        )
        print(
            f"    retrieval={item['retrieval_score']:.4f} "
            f"vintage_score={item['vintage_score']:.4f} "
            f"combined={item['combined_score']:.4f}"
            + (f" [{components}]" if components else "")
        )

    if show_context:
        print("\n--- CONTEXT ---")
        for context in response.get("contexts", []):
            print(context)
            print("-" * 40)

    warnings = response.get("warnings", [])
    if warnings:
        print("\n--- WARNINGS ---")
        for warning in warnings:
            print(f"! {warning}")


REPL_HELP = """Commands:
  :help              Show this help
  :sources           Re-print sources for the last query
  :context           Print retrieved context for the last query
  :warnings          Re-print warnings for the last query
  :save <file>       Save the last response as JSON (e.g. :save output.json)
  :set k <n>         Change how many documents are retrieved
  :exit              Leave the REPL (also: exit, quit, Ctrl-D)
Anything else is treated as a question."""


def _run_repl(orchestrator: RAGOrchestrator, args: argparse.Namespace) -> None:
    """Run the interactive QA loop."""
    print("Entering interactive QA mode. Type ':help' for commands, ':exit' to quit.")
    state: dict[str, Any] = {"k": args.k, "last": None}

    while True:
        try:
            entry = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not entry:
            continue
        if entry.lower() in {"exit", "quit", ":exit", ":quit"}:
            return

        if entry.startswith(":"):
            parts = entry.split()
            command = parts[0]
            last = state["last"]

            if command == ":help":
                print(REPL_HELP)
            elif command == ":set" and len(parts) == 3 and parts[1] == "k":
                try:
                    state["k"] = max(1, int(parts[2]))
                    print(f"k = {state['k']}")
                except ValueError:
                    print("Usage: :set k <integer>")
            elif command in {":sources", ":context", ":warnings"}:
                if last is None:
                    print("No query has been run yet.")
                elif command == ":sources":
                    _print_response({**last, "answer": last["answer"]}, show_context=False)
                elif command == ":context":
                    for context in last.get("contexts", []):
                        print(context)
                        print("-" * 40)
                else:
                    for warning in last.get("warnings", []) or ["(none)"]:
                        print(f"! {warning}")
            elif command == ":save":
                if last is None:
                    print("No query has been run yet.")
                elif len(parts) < 2:
                    print("Usage: :save <filename.json>")
                else:
                    save_path = Path(parts[1]).expanduser()
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    output_data = {
                        "query": last.get("_query", ""),
                        "answer": last["answer"],
                        "provenance": last.get("provenance", []),
                        "warnings": last.get("warnings", []),
                        "contexts": last.get("contexts", []),
                    }
                    save_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
                    print(f"Saved to {save_path}")
            else:
                print(f"Unknown command {command!r}. Type ':help'.")
            continue

        response = orchestrator.answer(entry, k=state["k"])
        response["_query"] = entry
        state["last"] = response
        _print_response(response, show_context=args.show_context)


def _command_ingest(args: argparse.Namespace) -> None:
    """Handle ``ingest``."""
    results = ingest_directory(
        raw_dir=args.raw_dir, db_path=args.db_path, clear_existing=args.clear
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


def _command_index(args: argparse.Namespace) -> None:
    """Handle ``index``."""
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


def _command_fred(args: argparse.Namespace) -> None:
    """Handle ``fred``."""
    series_ids = _split_csv(args.series_ids) or ([args.series_id] if args.series_id else [])
    if not series_ids:
        raise FredAPIError("Provide --series-id or --series-ids.")

    if len(series_ids) == 1:
        results = [
            save_fred_series(
                series_ids[0],
                output_dir=args.output_dir,
                api_key=args.api_key,
                observation_start=args.start_date,
                observation_end=args.end_date,
                file_name=args.file_name,
            )
        ]
    else:
        results = save_fred_series_batch(
            series_ids,
            output_dir=args.output_dir,
            api_key=args.api_key,
            observation_start=args.start_date,
            observation_end=args.end_date,
        )

    downloaded = [result for result in results if result.row_count]
    for result in results:
        if not result.row_count:
            _warn(f"No observations returned for FRED series {result.series_id}.")
            continue
        print(
            f"Downloaded FRED series {result.series_id} with {result.row_count} row(s) "
            f"to {result.output_path} ({result.start_date} -> {result.end_date})."
        )

    if args.ingest and downloaded:
        _ingest_paths(args.db_path, [result.output_path for result in downloaded])


def _command_qwi(args: argparse.Namespace) -> None:
    """Handle ``qwi``."""
    if not args.api_key:
        _warn(
            "CENSUS_API_KEY is not set; the QWI API allows limited keyless use and may "
            "throttle. Get a key at https://api.census.gov/data/key_signup.html"
        )

    result = save_qwi_data(
        state=args.state,
        years=[int(year) for year in _split_csv(args.years)],
        quarters=[int(quarter) for quarter in _split_csv(args.quarters)],
        variables=_split_csv(args.variables),
        dataset=args.dataset,
        county=args.county,
        industry=args.industry,
        api_key=args.api_key,
        output_dir=args.output_dir,
        file_name=args.file_name,
    )

    if not result.row_count:
        _warn(
            f"No QWI observations returned for {result.geography} "
            f"(dataset {result.dataset}); nothing was written."
        )
        return

    print(
        f"Downloaded {result.row_count} QWI row(s) for {result.geography} "
        f"[{result.dataset}] covering {', '.join(str(year) for year in result.years)} "
        f"to {result.output_path}."
    )

    if args.ingest:
        _ingest_paths(args.db_path, [result.output_path])


def _command_cps(args: argparse.Namespace) -> None:
    """Handle ``cps``."""
    frame = read_cps_dat(args.file, layout=args.layout, max_rows=args.max_rows)
    print(f"Parsed {len(frame)} CPS record(s) from {args.file.name}.")

    if not args.raw:
        frame = summarize_cps(frame)
        print(f"Aggregated to {len(frame)} year/month/state row(s).")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.file.name.removesuffix(".gz").removesuffix(".dat").removeprefix("cps_")
    suffix = "records" if args.raw else "summary"
    output_path = output_dir / (args.file_name or f"cps_{stem}_{suffix}.csv")
    frame.to_csv(output_path, index=False)
    print(f"Wrote {output_path}.")

    if args.ingest:
        _ingest_paths(args.db_path, [output_path])


def _command_qa(args: argparse.Namespace) -> None:
    """Handle ``qa``."""
    orchestrator = build_query_engine(args)
    if orchestrator is None:
        return

    if args.query:
        response = orchestrator.answer(args.query, k=args.k)
        if args.json or args.output:
            output_data = {
                "query": args.query,
                "answer": response["answer"],
                "provenance": response["provenance"],
                "warnings": response["warnings"],
                "contexts": response.get("contexts", []),
            }
            json_str = json.dumps(output_data, indent=2)
            if args.json:
                print(json_str)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json_str, encoding="utf-8")
                print(f"Response written to {args.output}")
        else:
            _print_response(response, show_context=args.show_context)
        return

    _run_repl(orchestrator, args)


def _command_eval(args: argparse.Namespace) -> None:
    """Handle ``eval``."""
    orchestrator = build_query_engine(args)
    if orchestrator is None:
        return

    if args.query:
        cases = [EvalCase(question=question) for question in args.query]
    elif args.questions.exists():
        cases = load_eval_cases(args.questions)
    else:
        print(
            f"No questions file at {args.questions}. Pass --query \"...\" or create the file."
        )
        return

    if not args.no_ragas and not ragas_available():
        _warn("ragas is not installed; falling back to heuristic metrics (pip install '.[eval]').")

    result = evaluate_queries(orchestrator, cases, k=args.k, use_ragas=not args.no_ragas)

    print(f"\nEvaluated {len(result.evaluations)} question(s) using the '{result.backend}' backend.")
    for evaluation in result.evaluations:
        metrics = " ".join(f"{name}={value:.3f}" for name, value in evaluation.metrics.items())
        print(f"- {evaluation.question}\n    {metrics}")
    print("\nAggregate:")
    for name, value in result.aggregate.items():
        print(f"  {name}: {value:.3f}")
    for note in result.notes:
        _warn(note)

    print(f"\nReport written to {result.write_json(args.output)}.")


def _command_status(args: argparse.Namespace) -> None:
    """Handle ``status``."""
    print(f"{config.APP_NAME}")
    print(f"\nDatabase: {args.db_path}")
    if args.db_path.exists():
        tables = list_tables(args.db_path)
        connection = duckdb.connect(str(args.db_path), read_only=True)
        try:
            for table in tables:
                count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                print(f"  - {table}: {count} row(s)")
        finally:
            connection.close()
        if not tables:
            print("  (no tables; run 'ingest')")
    else:
        print("  (missing; run 'ingest')")

    bm25_path = args.index_dir / "bm25_corpus.json"
    print(f"\nIndexes: {args.index_dir}")
    if bm25_path.exists():
        payload = json.loads(bm25_path.read_text(encoding="utf-8"))
        print(f"  - BM25 corpus: {len(payload.get('documents', []))} document(s)")
    else:
        print("  - BM25 corpus: missing (run 'index')")
    chroma_dir = args.index_dir / "chroma"
    print(f"  - Chroma store: {'present' if chroma_dir.exists() else 'missing (run index)'}")

    print("\nCredentials:")
    for name, value in (
        ("FRED_API_KEY", config.FRED_API_KEY),
        ("CENSUS_API_KEY", config.CENSUS_API_KEY),
        ("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY),
    ):
        print(f"  - {name}: {'set' if value else 'not set'}")
    print(f"  - ragas installed: {'yes' if ragas_available() else 'no'}")
    print(f"\nEmbedding model: {config.DEFAULT_EMBEDDING_MODEL}")


COMMANDS = {
    "ingest": _command_ingest,
    "index": _command_index,
    "fred": _command_fred,
    "qwi": _command_qwi,
    "cps": _command_cps,
    "qa": _command_qa,
    "eval": _command_eval,
    "status": _command_status,
}


def main() -> None:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args()

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return

    try:
        handler(args)
    except (FredAPIError, QWIAPIError, CPSLayoutError) as error:
        raise SystemExit(f"[error] {error}") from error


if __name__ == "__main__":
    main()
