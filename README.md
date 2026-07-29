# Unemployed — RAG Pipeline

A terminal-first RAG pipeline for exploring U.S. labor market data. Ingest real government data, build hybrid search indexes, ask questions in plain English, and get cited answers with freshness-aware scoring.

**CS 6365: Introduction to Enterprise Computing — Summer 2026**  
**Author:** Aiden Dowling

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Getting Your Data](#getting-your-data)
6. [Running the Pipeline](#running-the-pipeline)
7. [Commands Reference](#commands-reference)
8. [How It Works](#how-it-works)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
git clone https://github.com/aidendowling/Unemployed---RAG-Pipeline.git
cd Unemployed---RAG-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env       # Edit with your API keys
python cli.py              # Interactive guided setup
```

---

## Prerequisites

- **Python 3.10+**
- **API Keys** (all free):
  - **FRED API Key** — [Get one here](https://fredaccount.stlouisfed.org/apikeys) (instant, free)
  - **Census API Key** — [Get one here](https://api.census.gov/data/key_signup.html) (instant, free)
  - **OpenRouter API Key** (optional, for LLM answers) — [Get one here](https://openrouter.ai/keys) (free tier available)
- **CPS .dat file** (optional) — See [Getting Your Data](#cps-microdata-dat-file) below

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aidendowling/Unemployed---RAG-Pipeline.git
cd Unemployed---RAG-Pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install the package
pip install -e .

# macOS users: fix SSL certificates
pip install certifi
```

---

## Configuration

### 1. Create your `.env` file

```bash
cp .env.example .env
```

### 2. Add your API keys

Edit `.env` and fill in:

```dotenv
# Required for FRED data
FRED_API_KEY=your_fred_api_key_here

# Required for QWI data
CENSUS_API_KEY=your_census_api_key_here

# Optional: enables LLM-generated answers (free tier)
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

### 3. Verify your setup

```bash
unemployed-rag-pipeline status
```

This shows which keys are configured and what data/indexes exist.

---

## Getting Your Data

The pipeline ingests data from three sources. You need **at least one** to get started.

### FRED API (Recommended — Easiest)

No file download needed. The CLI fetches data directly from the FRED API.

```bash
unemployed-rag-pipeline fred --series-ids UNRATE,CAUR,TXUR,FLUR --start-date 2015-01-01 --end-date 2023-12-31 --ingest
```

**Common series IDs:**

| Series | Description |
|--------|-------------|
| `UNRATE` | U.S. national unemployment rate |
| `CAUR` | California unemployment rate |
| `TXUR` | Texas unemployment rate |
| `FLUR` | Florida unemployment rate |
| `NYUR` | New York unemployment rate |
| `GAUR` | Georgia unemployment rate |

Browse all series at [fred.stlouisfed.org](https://fred.stlouisfed.org/).

### Census QWI (Quarterly Workforce Indicators)

No file download needed. The CLI fetches data directly from the Census API.

```bash
unemployed-rag-pipeline qwi --state 06 --years 2019,2020,2021,2022 --ingest
```

**Common state FIPS codes:**

| Code | State | Code | State |
|------|-------|------|-------|
| `06` | California | `36` | New York |
| `12` | Florida | `48` | Texas |
| `13` | Georgia | `42` | Pennsylvania |

### CPS Microdata (.dat file)

CPS (Current Population Survey) data comes as fixed-width `.dat` files that require a layout to parse.

#### Where to download:

**Option A: IPUMS CPS (Recommended)**

1. Go to [https://cps.ipums.org/cps/](https://cps.ipums.org/cps/)
2. Create a free account
3. Click **"Get Data"** → Select variables you want (e.g., YEAR, MONTH, EMPSTAT, EARNWEEK, OCC, IND, EDUC, INCWAGE)
4. Click **"Create Data Extract"** → Select **.dat** format
5. Submit and wait for the extract (usually a few minutes)
6. Download the `.dat` file **AND the DDI codebook (.xml file)** — you need both!
7. Place both in `data/raw/`:
   ```
   data/raw/cps_00001.dat
   data/raw/cps_00001.xml    ← This is your layout file
   ```

**Option B: Census Bureau direct**

1. Go to [https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html](https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html)
2. Download a Basic Monthly `.dat` file
3. Place in `data/raw/` — the built-in `cps-basic-monthly` layout works for these files

#### How to ingest CPS data:

```bash
# If you have an IPUMS extract with .xml codebook next to the .dat:
unemployed-rag-pipeline cps --file data/raw/cps_00001.dat --raw --ingest

# If you have an IPUMS extract without .xml (use built-in IPUMS ASEC layout):
unemployed-rag-pipeline cps --file data/raw/cps_00001.dat --layout cps-ipums-asec --raw --ingest

# If you have a Census Bureau Basic Monthly file:
unemployed-rag-pipeline cps --file data/raw/cps_basic.dat --layout cps-basic-monthly --ingest

# If you have a custom layout file:
unemployed-rag-pipeline cps --file data/raw/cps_00001.dat --layout data/raw/cps_00001.xml --raw --ingest
```

> **Note:** Use `--raw` when your extract doesn't have the `PEMLR` (labor force status) variable. Without PEMLR, the pipeline can't compute unemployment rates, so it ingests person-level records as-is.

### Sample Data (No Downloads Needed)

For testing without any API keys or downloads:

```bash
cp data/sample/unemployment_2015_2022.csv data/raw/
unemployed-rag-pipeline ingest --clear
unemployed-rag-pipeline index --clear
unemployed-rag-pipeline qa --query "unemployment in California 2020"
```

---

## Running the Pipeline

### Option A: Interactive CLI (Recommended)

```bash
python cli.py
```

This launches a menu-driven interface that walks you through everything:

```
╔══════════════════════════════════════════════════════════════╗
║           Unemployed RAG Pipeline — Interactive CLI           ║
╚══════════════════════════════════════════════════════════════╝

  ✓ FRED_API_KEY is set
  ✓ CENSUS_API_KEY is set
  ✓ Database exists
  ✓ Indexes built

  1   Full Pipeline (end-to-end guided setup)
  2   Download FRED data
  3   Download QWI data
  4   Parse CPS .dat file
  5   Bulk ingest (all files in data/raw/)
  6   Build indexes
  7   Ask a question
  8   Interactive QA (chat mode)
  9   Batch QA (run questions file → JSON output)
  10  Evaluate pipeline
  0   System status
  q   Quit
```

**Pick option 1** to run the full pipeline end-to-end with guided prompts.

### Option B: Command Line (Step by Step)

```bash
# Step 1: Ingest data from sources
unemployed-rag-pipeline fred --series-ids UNRATE,CAUR,TXUR,FLUR --start-date 2015-01-01 --end-date 2023-12-31 --ingest
unemployed-rag-pipeline qwi --state 06 --years 2019,2020,2021,2022 --ingest
unemployed-rag-pipeline cps --file data/raw/cps_00001.dat --layout cps-ipums-asec --raw --ingest

# Step 2: Build search indexes
unemployed-rag-pipeline index --clear

# Step 3: Ask questions
unemployed-rag-pipeline qa --query "What was the unemployment rate in California in 2020?" --show-context

# Step 4: Batch run questions and save output
unemployed-rag-pipeline qa --questions data/eval/demo_questions.json --output data/outputs/batch_answers.json
```

---

## Commands Reference

| Command | What it does |
|---------|-------------|
| `fred` | Download FRED series → CSV → optionally ingest into DuckDB |
| `qwi` | Download Census QWI data → CSV → optionally ingest into DuckDB |
| `cps` | Parse a CPS .dat file → CSV → optionally ingest into DuckDB |
| `ingest` | Bulk-load all files in `data/raw/` into DuckDB |
| `index` | Build Chroma vector + BM25 keyword indexes from DuckDB |
| `qa` | Ask questions (single, interactive, or batch with JSON output) |
| `eval` | Score pipeline quality (faithfulness, relevancy, precision, recall) |
| `status` | Show database tables, index status, and configured credentials |

### Output Files

| Command | Output |
|---------|--------|
| `fred --ingest` | `data/raw/fred_*.csv` + DuckDB table |
| `qwi --ingest` | `data/raw/qwi_*.csv` + DuckDB table |
| `cps --ingest` | `data/raw/cps_*.csv` + DuckDB table |
| `index` | `data/indexes/chroma/` + `data/indexes/bm25_corpus.json` |
| `qa --output file.json` | JSON with answer, provenance, contexts, warnings |
| `qa --questions file.json` | Batch JSON report with all answers |
| `eval` | `data/outputs/eval_results.json` |

---

## How It Works

### Architecture

```
FRED API ──┐
Census QWI ─┤──→ Ingestion ──→ DuckDB ──→ Indexing ──→ ChromaDB + BM25
CPS .dat ──┘                                                │
                                                            ▼
User Query ──→ Hybrid Retrieval (BM25 + Semantic) ──→ RRF Fusion
                                                            │
                                                            ▼
                                               Obsolescence Scoring
                                          (query-intent vs. vintage_year)
                                                            │
                                                            ▼
                                               LLM Generation (Llama 3.1)
                                                            │
                                                            ▼
                                            Answer + Sources + Warnings
```

### Key Innovation: Knowledge Obsolescence

The system detects **what time period your question is about** and penalizes data that doesn't match:

| Query | Inferred intent | 2020 doc score | 2010 doc score |
|-------|----------------|---------------|---------------|
| "unemployment in 2020" | wants 2020 data | 1.0 | 0.5 |
| "trends from 2015 to 2019" | wants 2015-2019 | 0.93 | 0.7 |
| "current labor market" | no time filter | 1.0 | 1.0 |

This prevents the system from answering time-specific questions with stale data.

### Hybrid Retrieval

Two search methods combined via Reciprocal Rank Fusion:

- **BM25 (Keyword):** Exact matches — "UNRATE", "California", "NAICS 23"
- **Semantic (Vector):** Meaning matches — "joblessness" finds "unemployment"

Every result shows which retriever(s) found it and the component scores.

---

## Project Structure

```
Unemployed---RAG-Pipeline/
├── cli.py                              # Interactive menu CLI
├── pyproject.toml                      # Dependencies & package config
├── .env.example                        # Template for API keys
├── data/
│   ├── raw/                            # Downloaded/ingested source files
│   ├── processed/                      # DuckDB database
│   ├── indexes/                        # Chroma + BM25 indexes
│   ├── eval/                           # Evaluation question files
│   ├── outputs/                        # JSON output reports
│   └── sample/                         # Sample data for testing
├── src/unemployed_rag_pipeline/
│   ├── config.py                       # Central configuration
│   ├── main.py                         # CLI entry point (all subcommands)
│   ├── ingestion/
│   │   ├── fred.py                     # FRED API connector
│   │   ├── qwi.py                      # Census QWI API connector
│   │   ├── cps.py                      # CPS .dat fixed-width parser
│   │   ├── pipeline.py                 # Generic file → DuckDB loader
│   │   ├── http.py                     # SSL-aware HTTP helper
│   │   └── layouts/                    # Built-in CPS layouts (JSON)
│   ├── embedding/embedder.py           # SentenceTransformer wrapper
│   ├── indexing.py                     # Builds Chroma + BM25 indexes
│   ├── retrieval/
│   │   ├── semantic.py                 # Vector search (ChromaDB)
│   │   ├── keyword.py                  # BM25 keyword search
│   │   └── hybrid.py                   # RRF/weighted fusion
│   ├── obsolescence/freshness.py       # Query-intent vintage scoring
│   ├── rag.py                          # RAG orchestrator + generators
│   └── evaluation/ragas_eval.py        # RAGAS + heuristic evaluation
├── scripts/mock_api_server.py          # Fake API for offline testing
└── tests/                              # Unit tests
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `SSL: CERTIFICATE_VERIFY_FAILED` | Run `pip install certifi` — this is a macOS Python issue |
| `CENSUS_API_KEY is not set` | Add it to your `.env` file (not `.env.example`) |
| `Layout parsed implausible HRYEAR4 values` | Your .dat file needs a different layout — pass `--layout cps-ipums-asec` or your `.xml` DDI file |
| `Cannot aggregate CPS data without PEMLR` | Your extract doesn't have labor force status — use `--raw` flag |
| `No retrievers could be initialized` | Run `ingest` then `index` first to build the search indexes |
| Only 5 subcommands showing | Run `pip install -e . --force-reinstall --no-deps` to update the entry point |
| `No supported data files found` | Place files in `data/raw/` or use the `fred`/`qwi`/`cps` commands to download |
| Slow first run of `index` | Normal — downloads the embedding model (~80MB) on first use |
| Answers are just "formatted context" | Set `OPENROUTER_API_KEY` in `.env` for LLM-generated answers |

---

## Running Tests

```bash
python -m pytest -q
```

---

## License

MIT
