#!/usr/bin/env python3
"""Interactive CLI for the Unemployed RAG Pipeline.

Run with:
    python cli.py

Provides a menu-driven interface to ingest data from FRED, Census QWI, and
CPS microdata files, build retrieval indexes, ask questions, and check system
status — all without memorizing command flags.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Styling helpers
# ──────────────────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"


def banner() -> None:
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║           Unemployed RAG Pipeline — Interactive CLI           ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}── {title} ──{RESET}\n")


def success(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def error(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return value or default


def confirm(msg: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        answer = input(f"  {msg} ({hint}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


# ──────────────────────────────────────────────────────────────────────────────
# Command execution
# ──────────────────────────────────────────────────────────────────────────────

BASE_CMD = [sys.executable, "-m", "unemployed_rag_pipeline.main"]


def run_pipeline(*args: str) -> int:
    """Run the pipeline CLI with given arguments."""
    cmd = BASE_CMD + list(args)
    print(f"\n  {DIM}$ {' '.join(cmd)}{RESET}\n")
    env = os.environ.copy()
    # Ensure src is on the path
    src_dir = str(Path(__file__).parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent))
    print()
    return result.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Menu actions
# ──────────────────────────────────────────────────────────────────────────────

def action_status() -> None:
    """Show system status."""
    section("System Status")
    run_pipeline("status")


def action_fred() -> None:
    """Download FRED series."""
    section("FRED Data Ingestion")
    print("  Common series: UNRATE (national), CAUR (CA), TXUR (TX), FLUR (FL),")
    print("                 NYUR (NY), GAUR (GA), PAUR (PA)")
    print()

    series = prompt("Series IDs (comma-separated)", "UNRATE,CAUR,TXUR,FLUR")
    if not series:
        return

    start = prompt("Start date (YYYY-MM-DD, or blank for all)", "2015-01-01")
    end = prompt("End date (YYYY-MM-DD, or blank for all)", "2023-12-31")
    ingest = confirm("Ingest into database after download?")

    args = ["fred", "--series-ids", series]
    if start:
        args += ["--start-date", start]
    if end:
        args += ["--end-date", end]
    if ingest:
        args.append("--ingest")

    run_pipeline(*args)


def action_qwi() -> None:
    """Download QWI data."""
    section("Census QWI Data Ingestion")
    print("  State FIPS codes: 06=CA, 12=FL, 13=GA, 36=NY, 48=TX")
    print("  Datasets: sa=seasonally adjusted, se=sex/education, rh=race/ethnicity")
    print()

    state = prompt("State FIPS code (2 digits)", "06")
    if not state:
        return

    years = prompt("Years (comma-separated)", "2019,2020,2021,2022")
    quarters = prompt("Quarters (comma-separated)", "1,2,3,4")
    dataset = prompt("Dataset (sa/se/rh)", "sa")
    ingest = confirm("Ingest into database after download?")

    args = ["qwi", "--state", state, "--years", years, "--quarters", quarters, "--dataset", dataset]
    if ingest:
        args.append("--ingest")

    run_pipeline(*args)


def action_cps() -> None:
    """Parse CPS .dat file."""
    section("CPS Microdata Ingestion")

    # Find .dat files in data/raw
    raw_dir = Path("data/raw")
    dat_files = sorted(raw_dir.glob("*.dat")) + sorted(raw_dir.glob("*.dat.gz"))

    if dat_files:
        print("  Found .dat files in data/raw/:")
        for i, f in enumerate(dat_files, 1):
            print(f"    {i}. {f.name}")
        print()
        choice = prompt(f"Select file (1-{len(dat_files)}) or type a path", "1")
        try:
            idx = int(choice) - 1
            dat_path = str(dat_files[idx])
        except (ValueError, IndexError):
            dat_path = choice
    else:
        dat_path = prompt("Path to CPS .dat file")
        if not dat_path:
            return

    print()
    print("  Available layouts:")
    print("    1. cps-basic-monthly  (Census/BLS basic monthly, 2003+ format)")
    print("    2. cps-ipums-asec     (IPUMS extract: YEAR/SERIAL/MONTH/HWTFINL/...)")
    print("    3. Auto-detect        (looks for sidecar .xml/.json next to .dat)")
    print("    4. Custom path        (provide your own .xml/.json/.txt layout)")
    print()
    layout_choice = prompt("Layout (1-4)", "2")

    layout_map = {
        "1": "cps-basic-monthly",
        "2": "cps-ipums-asec",
    }

    if layout_choice in ("1", "2"):
        layout = layout_map[layout_choice]
    elif layout_choice == "3":
        layout = None
    elif layout_choice == "4":
        layout = prompt("Path to layout file")
    else:
        layout = layout_choice  # Treat as a layout name or path

    raw = confirm("Use raw mode (person-level records, no aggregation)?", default=True)
    max_rows_str = prompt("Max rows to parse (blank for all)", "")
    ingest = confirm("Ingest into database after parsing?")

    args = ["cps", "--file", dat_path]
    if layout:
        args += ["--layout", layout]
    if raw:
        args.append("--raw")
    if max_rows_str:
        args += ["--max-rows", max_rows_str]
    if ingest:
        args.append("--ingest")

    run_pipeline(*args)


def action_ingest_all() -> None:
    """Bulk ingest all files in data/raw."""
    section("Bulk Ingest")
    print("  This loads ALL supported files in data/raw/ into DuckDB.")
    print()

    clear = confirm("Clear existing database first?", default=False)
    args = ["ingest"]
    if clear:
        args.append("--clear")
    run_pipeline(*args)


def action_index() -> None:
    """Build retrieval indexes."""
    section("Build Indexes")
    print("  This embeds all database rows and builds Chroma + BM25 indexes.")
    print("  (May take a minute on first run while downloading the embedding model.)")
    print()

    clear = confirm("Clear existing indexes and rebuild?", default=True)
    args = ["index"]
    if clear:
        args.append("--clear")
    run_pipeline(*args)


def action_ask() -> None:
    """Ask a question."""
    section("Ask a Question")

    query = prompt("Your question")
    if not query:
        return

    show_ctx = confirm("Show retrieved context passages?", default=False)
    save = confirm("Save response to a file?", default=False)

    args = ["qa", "--query", query]
    if show_ctx:
        args.append("--show-context")
    if save:
        output_path = prompt("Output file path", "data/outputs/answer.json")
        args += ["--output", output_path]

    run_pipeline(*args)


def action_chat() -> None:
    """Enter interactive QA mode."""
    section("Interactive QA Mode")
    print("  Entering interactive mode. Type questions at the prompt.")
    print("  Commands: :help, :sources, :context, :exit")
    print()
    run_pipeline("qa", "--show-context")


def action_batch_qa() -> None:
    """Run a batch of pre-created questions and save all answers."""
    section("Batch QA (Run Questions File)")

    # Find question files
    eval_dir = Path("data/eval")
    json_files = sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []

    if json_files:
        print("  Available question files:")
        for i, f in enumerate(json_files, 1):
            print(f"    {i}. {f}")
        print()
        choice = prompt(f"Select file (1-{len(json_files)}) or type a path", "1")
        try:
            idx = int(choice) - 1
            questions_path = str(json_files[idx])
        except (ValueError, IndexError):
            questions_path = choice
    else:
        questions_path = prompt("Path to questions JSON file")
        if not questions_path:
            return

    output_path = prompt("Output file path", "data/outputs/batch_answers.json")

    print()
    print(f"  Will run all questions from: {questions_path}")
    print(f"  Answers will be saved to:    {output_path}")
    print()

    if not confirm("Run batch?"):
        return

    run_pipeline("qa", "--questions", questions_path, "--output", output_path)


def action_eval() -> None:
    """Run evaluation."""
    section("Evaluate Pipeline")
    print("  Scores the pipeline against evaluation questions.")
    print()

    use_custom = confirm("Use custom questions (instead of data/eval/questions.json)?", default=False)
    args = ["eval", "--no-ragas"]

    if use_custom:
        questions = []
        print("  Enter questions (blank line to finish):")
        while True:
            q = prompt("  Question")
            if not q:
                break
            questions.append(q)
        for q in questions:
            args += ["--query", q]

    run_pipeline(*args)


def action_full_pipeline() -> None:
    """Run the full pipeline end-to-end."""
    section("Full Pipeline (End-to-End)")
    print("  This will:")
    print("    1. Download FRED unemployment series")
    print("    2. Download QWI workforce data")
    print("    3. Parse your CPS .dat file")
    print("    4. Build retrieval indexes")
    print("    5. Drop you into interactive QA mode")
    print()

    if not confirm("Run the full pipeline?"):
        return

    # FRED
    print(f"\n{BOLD}Step 1/5: FRED{RESET}")
    series = prompt("FRED series", "UNRATE,CAUR,TXUR,FLUR")
    if series:
        run_pipeline("fred", "--series-ids", series, "--start-date", "2015-01-01",
                     "--end-date", "2023-12-31", "--ingest")

    # QWI
    print(f"\n{BOLD}Step 2/5: QWI{RESET}")
    state = prompt("State FIPS for QWI", "06")
    years = prompt("Years", "2019,2020,2021,2022")
    if state and years:
        run_pipeline("qwi", "--state", state, "--years", years, "--ingest")

    # CPS
    print(f"\n{BOLD}Step 3/5: CPS{RESET}")
    raw_dir = Path("data/raw")
    dat_files = sorted(raw_dir.glob("*.dat")) + sorted(raw_dir.glob("*.dat.gz"))
    if dat_files:
        dat_path = str(dat_files[0])
        print(f"  Using: {dat_path}")
        run_pipeline("cps", "--file", dat_path, "--layout", "cps-ipums-asec", "--raw", "--ingest")
    else:
        warn("No .dat files found in data/raw/. Skipping CPS.")

    # Index
    print(f"\n{BOLD}Step 4/5: Building indexes{RESET}")
    run_pipeline("index", "--clear")

    # QA
    print(f"\n{BOLD}Step 5/5: Interactive QA{RESET}")
    run_pipeline("qa", "--show-context")


# ──────────────────────────────────────────────────────────────────────────────
# Main menu
# ──────────────────────────────────────────────────────────────────────────────

MENU = [
    ("1", "Full Pipeline (end-to-end guided setup)", action_full_pipeline),
    ("2", "Download FRED data", action_fred),
    ("3", "Download QWI data", action_qwi),
    ("4", "Parse CPS .dat file", action_cps),
    ("5", "Bulk ingest (all files in data/raw/)", action_ingest_all),
    ("6", "Build indexes", action_index),
    ("7", "Ask a question", action_ask),
    ("8", "Interactive QA (chat mode)", action_chat),
    ("9", "Batch QA (run questions file → JSON output)", action_batch_qa),
    ("10", "Evaluate pipeline", action_eval),
    ("0", "System status", action_status),
    ("q", "Quit", None),
]


def main() -> None:
    os.chdir(Path(__file__).parent)
    banner()

    # Quick status check
    from dotenv import load_dotenv
    load_dotenv()

    checks = {
        "FRED_API_KEY": bool(os.getenv("FRED_API_KEY")),
        "CENSUS_API_KEY": bool(os.getenv("CENSUS_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    for name, present in checks.items():
        if present:
            success(f"{name} is set")
        else:
            warn(f"{name} is not set")

    db_path = Path(os.getenv("DUCKDB_PATH", "data/processed/unemployed_rag.duckdb"))
    idx_dir = Path(os.getenv("INDEX_DIR", "data/indexes"))
    if db_path.exists():
        success(f"Database exists: {db_path}")
    else:
        warn(f"No database yet: {db_path}")
    if (idx_dir / "chroma").exists():
        success(f"Indexes built: {idx_dir}")
    else:
        warn(f"No indexes yet (run option 6 after ingesting data)")

    while True:
        print(f"\n{BOLD}What would you like to do?{RESET}\n")
        for key, label, _ in MENU:
            prefix = f"  {CYAN}{key}{RESET}"
            print(f"{prefix}  {label}")

        print()
        choice = prompt("Choice").lower()

        if choice in ("q", "quit", "exit"):
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break

        matched = next((action for key, _, action in MENU if key == choice), "NOMATCH")
        if matched == "NOMATCH":
            error(f"Unknown option: {choice!r}")
        elif matched is None:
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        else:
            try:
                matched()
            except KeyboardInterrupt:
                print(f"\n\n  {DIM}(interrupted){RESET}")


if __name__ == "__main__":
    main()
