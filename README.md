# Unemployed RAG Pipeline

A terminal-first RAG pipeline for exploring U.S. labor market data. The project ingests raw files into DuckDB, builds keyword and semantic retrieval indexes, and exposes a query-time QA flow with obsolescence filtering.

CS 4365/6365: Introduction to Enterprise Computing — Summer 2026

Author: Aiden Dowling

## What the project does

1. Ingests labor-market data into DuckDB from three kinds of source:
   - **FRED API** (`fred` command) — any FRED series, e.g. `UNRATE`, `CAUR`
   - **Census QWI API** (`qwi` command) — Quarterly Workforce Indicators by state/county
   - **CPS microdata** (`cps` command) — fixed-width `.dat` extracts, parsed with a layout
     and aggregated into monthly state-level employment/unemployment summaries
   - Plus generic `.csv`, `.tsv`, `.txt`, `.parquet`, and `.dat` files dropped in `data/raw/`
   - Every ingested row gets a normalized `vintage_year` plus `source`/`source_file` provenance
2. Builds retrieval indexes from the DuckDB tables:
   - BM25 keyword index with tokenized corpus
   - Chroma semantic index with sentence embeddings
3. Implements a **query-intent-aware knowledge-obsolescence layer** (the novel differentiator):
   - Extracts implied time ranges from user queries (e.g., "in 2020" → [2015, 2025])
   - Scores documents based on alignment between vintage_year and query-implied time range
   - Perfect-match documents score 1.0; out-of-range documents receive exponential decay
   - Reranks on `retrieval_score x vintage_score` and warns about stale or vintage-less hits
4. Runs a RAG QA loop that **fuses** semantic and keyword retrieval (reciprocal rank fusion or
   weighted score fusion), generates answers through OpenRouter, and prints per-source provenance.
5. Scores the whole query engine with `eval` — RAGAS when installed, deterministic offline
   heuristics otherwise.

### End-to-end in four commands

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "unemployment in California in 2020"
PYTHONPATH=src python -m unemployed_rag_pipeline.main eval
```

Run `status` at any point to see which tables, indexes, and credentials are in place.

## The Knowledge-Obsolescence Layer

The obsolescence system is **query-intent-relative**, not generic recency decay:

- **Query: "unemployment in 2020"** → Looks for documents with vintage_year ∈ [2015, 2025]
  - 2020, 2021 documents → score 1.0 (perfect alignment)
  - 2019, 2022 documents → score 1.0 (within window)
  - 2010 documents → score ~0.7 (10 years out, exponential decay)
  - No time intent → score 1.0 (temporally irrelevant queries get all documents)

- **Query: "unemployment from 2015 to 2020"** → Strictly uses [2015, 2020]
  - Documents within range → score 1.0
  - Documents outside → exponential decay by distance

Scoring is applied in the QA path itself: results are reranked by `retrieval_score x vintage_score`,
documents below `--min-vintage-score` are dropped, and anything below `VINTAGE_WARN_SCORE` (or
missing a vintage entirely) raises a warning printed under the answer. Pass `--no-obsolescence`
to rank on retrieval score alone.

## Project layout

- `src/unemployed_rag_pipeline/config.py` — shared path, model, and threshold configuration
- `src/unemployed_rag_pipeline/ingestion/pipeline.py` — raw file ingestion into DuckDB
- `src/unemployed_rag_pipeline/ingestion/fred.py` — FRED API client
- `src/unemployed_rag_pipeline/ingestion/qwi.py` — Census QWI API client
- `src/unemployed_rag_pipeline/ingestion/cps.py` — CPS fixed-width `.dat` parser and aggregator
- `src/unemployed_rag_pipeline/ingestion/layouts/` — bundled CPS Basic Monthly layout
- `src/unemployed_rag_pipeline/indexing.py` — builds BM25 and Chroma indexes
- `src/unemployed_rag_pipeline/retrieval/keyword.py` — BM25 retrieval
- `src/unemployed_rag_pipeline/retrieval/semantic.py` — Chroma semantic retrieval
- `src/unemployed_rag_pipeline/retrieval/hybrid.py` — RRF / weighted fusion of both retrievers
- `src/unemployed_rag_pipeline/rag.py` — RAG orchestration, vintage reranking, provenance
- `src/unemployed_rag_pipeline/obsolescence/freshness.py` — freshness and obsolescence helpers
- `src/unemployed_rag_pipeline/evaluation/ragas_eval.py` — RAGAS + offline heuristic evaluation
- `src/unemployed_rag_pipeline/main.py` — CLI entry point
- `scripts/mock_api_server.py` — canned FRED/QWI responses for offline runs

## Key Features

### 1. Temporal Normalization During Ingestion
The ingestion pipeline automatically detects year/vintage columns in source data and normalizes them into a `vintage_year` field. Supported column names: `year`, `vintage`, `survey_year`, `reference_year`, `data_year`.

### 2. Query-Intent-Aware Obsolescence Scoring
Unlike generic recency decay, the obsolescence layer extracts time ranges from queries and scores documents relative to query intent:
- Query parser recognizes patterns: "in 2020", "2015-2020", "since 2019", "before 2018"
- Documents within query-implied range score 1.0 (perfect relevance)
- Out-of-range documents receive exponential decay: `0.5^(distance_years/10)`
- Queries with no time context return all documents with full relevance

### 3. Hybrid Retrieval with Orchestration
Combines lexical and semantic search into a single ranking:
- **BM25**: Fast keyword matching with term-frequency scoring
- **Semantic (Chroma)**: Vector similarity using sentence embeddings, queried with the same
  SentenceTransformer used at index time
- **Fusion**: reciprocal rank fusion (default) or weighted min-max score fusion,
  configurable per run with `--fusion`, `--semantic-weight`, `--rrf-k`
- **Orchestrator**: Deduplicates by document ID, applies vintage scores, sorts by combined
  relevance, and reports which retriever(s) matched each document
- **Degradation**: if one retriever is missing or throws, the other still answers and the
  failure is surfaced as a warning rather than crashing the query

### 4. LLM-Based Answer Generation
The pipeline can use OpenRouter to call remote LLMs for answer synthesis. Falls back gracefully to a stub generator if no API key is configured or if the API call fails.

### 5. Evaluation
`eval` runs a question set through the full query engine and reports faithfulness,
answer relevancy, context precision, and context recall. With `ragas` installed
(`pip install '.[eval]'`) the metrics are LLM-judged RAGAS scores; without it the pipeline
falls back to deterministic lexical-overlap heuristics so evaluation still runs offline.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Configure LLM Integration:

To enable answer generation via OpenRouter:

1. Create an `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

2. Add your OpenRouter API key:
   ```bash
   echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
   ```
   Get a free API key at [OpenRouter Dashboard](https://openrouter.ai/keys)

3. (Optional) Customize the model:
   ```bash
   # .env
   OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct  # default
   # other good options:
   # OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
   # OPENROUTER_MODEL=openai/gpt-4o-2024-05-13
   ```

### Configure FRED API Access

If you want to download data directly from FRED:

1. Add your FRED API key to `.env`:
   ```bash
   echo "FRED_API_KEY=your_fred_key_here" >> .env
   ```

2. Optional: override the API base URL if needed:
   ```bash
   # .env
   FRED_BASE_URL=https://api.stlouisfed.org/fred
   ```

3. Verify the key is set before downloading a series:
   ```bash
   PYTHONPATH=src python -m unemployed_rag_pipeline.main status
   ```

### Configure Census QWI API Access

The Census API tolerates a small number of keyless requests but will throttle you quickly.
Request a free key at https://api.census.gov/data/key_signup.html and add it to `.env`:

```bash
echo "CENSUS_API_KEY=your_census_key_here" >> .env
```

### Running without any API keys

`scripts/mock_api_server.py` serves canned FRED and QWI payloads so the API paths can be
exercised offline:

```bash
python scripts/mock_api_server.py 8765 &
FRED_BASE_URL=http://127.0.0.1:8765/fred QWI_BASE_URL=http://127.0.0.1:8765/qwi \
  PYTHONPATH=src python -m unemployed_rag_pipeline.main fred --series-id UNRATE --api-key mock --ingest
```

## Data directories

### Directory Structure
- **Raw input**: `data/raw/` — Place raw CSV/TSV/Parquet/CPS `.dat` files here before ingestion
  (git-ignored; downloads from `fred`/`qwi` land here too)
- **Samples**: `data/sample/` — Committed synthetic fixtures for a zero-credential demo
- **Eval**: `data/eval/questions.json` — Question set used by the `eval` command
- **Processed**: `data/processed/unemployed_rag.duckdb` — DuckDB database created after ingestion
- **Indexes**: `data/indexes/` — BM25 corpus and Chroma vector index created after indexing
- **Outputs**: `data/outputs/` — Query evaluation results and test outputs

### Getting Labor Market Data

The pipeline is designed for U.S. labor market datasets. Here are **detailed step-by-step instructions** to download real data:

#### Option A: **BLS Local Area Unemployment Statistics (LAUS)** — Recommended for state-level data

**Steps:**

1. Go to https://www.bls.gov/lau/
2. Click **"One-Screen Data Search"** (or **"Download"**)
3. Select:
   - **Seasonal Adjustment**: "Seasonally Adjusted" or "Not Seasonally Adjusted" (pick one)
   - **States**: Check the states you want (e.g., California, Texas, Florida)
   - **Years**: Select date range (e.g., 2015-2024)
4. Click **"Retrieve Data"** 
5. The page will show a table. Click **"Download"** (usually in the top-right) and select **CSV**
6. Save the file to `data/raw/` with a descriptive name like `bls_laus_state_2015_2024.csv`

**Important**: The downloaded CSV will have columns like:
- `Year` or year indicator
- `Month` or period
- `State`
- `Unemployment Rate`
- `Labor Force`
- `Employed`
- `Unemployed`

Make sure the year column is named `year` (case-insensitive). If it's `Year` or `YEAR`, the pipeline auto-detects it.

#### Option B: **FRED (Federal Reserve Economic Data)** — Good for national/single-state time series

FRED provides clean, curated economic data.

**Steps:**

1. Go to https://fred.stlouisfed.org/
2. **For national unemployment**: Search for `UNRATE`
   - Click on **[UNRATE](https://fred.stlouisfed.org/series/UNRATE)** (Civilian Unemployment Rate)
3. **For state-specific unemployment**: Search for state abbreviation + `UR` (e.g., `CAUR` for California)
   - Example: [CAUR](https://fred.stlouisfed.org/series/CAUR) (California Unemployment Rate)
4. On the series page:
   - Set **Frequency** dropdown to "Monthly" (if available)
   - Set **Units** to "Percent" or "Number" depending on what you want
   - Click the **"Download"** button (bottom-left area)
   - Choose **CSV** format
5. Save to `data/raw/` with a name like `fred_unemployment_rate_ca.csv`

**Important**: FRED CSVs have a simple format:
```
DATE,UNRATE
2015-01-01,6.2
2015-02-01,5.8
2020-04-01,16.4
```

A manual download needs a `year` column added before ingestion. The `fred` command does this for
you — it writes `date`, `year`, `month`, `value`, `series_id`, `vintage_year`, `source`, and
`retrieved_at` — so prefer:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main fred --series-id CAUR --start-date 2015-01-01 --ingest
```

#### Option C: **Census QWI** — Quarterly Workforce Indicators by state or county

No manual download needed; the `qwi` command calls the API directly:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qwi --state 13 --years 2019,2020,2021 --ingest
PYTHONPATH=src python -m unemployed_rag_pipeline.main qwi --state 13 --county 121 --years 2020 --industry 23 --ingest
```

Default variables are `Emp`, `EmpEnd`, `HirA`, `Sep`, `EarnBeg`, `FrmJbGn`; override with
`--variables`. `--dataset` chooses between `sa` (seasonally adjusted), `se` (sex/education),
and `rh` (race/ethnicity). Variable definitions: https://lehd.ces.census.gov/data/schema/latest/lehd_public_use_schema.html

#### Option D: **CPS microdata** — the `.dat` extracts

The `cps` command parses a fixed-width CPS file and, by default, aggregates person records into
monthly state-level employment, unemployment, and unemployment-rate rows (weighted by
`PWCMPWGT`/`HWHHWGT` when present):

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main cps --file data/raw/cps_2020.dat --ingest
PYTHONPATH=src python -m unemployed_rag_pipeline.main cps --file data/raw/cps_2020.dat.gz --raw   # person-level
```

**Layouts matter.** A fixed-width file is unreadable without the column positions, so the parser
resolves a layout in this order:

1. `--layout` (or `CPS_LAYOUT_PATH` in `.env`) — an IPUMS `.xml` DDI, a JSON field list, a Census
   data dictionary `.txt`/`.dct`, or the builtin name `cps-basic-monthly`
2. a sidecar file with the same stem next to the `.dat` (e.g. `cps_2020.xml`)
3. a single `.xml` in the same directory
4. the bundled `cps-basic-monthly` layout, as a last resort

The bundled layout covers the public Basic Monthly CPS record and is **only a fallback** — if your
extract is an IPUMS custom extract or a different CPS supplement, pass its layout with `--layout`.
The parser refuses to continue when the parsed survey years look implausible, so a mismatched
layout fails loudly instead of producing garbage.

#### Option E: **Use the Sample Data** — Included for immediate testing

Two synthetic fixtures ship with the repo so the pipeline can be exercised without any downloads:

- `data/sample/unemployment_2015_2022.csv` — 96 monthly snapshots across California, Texas, and
  Florida (2015-2022) with columns `year`, `month`, `state`, `unemployment_rate`,
  `total_labor_force`, `employed`, `unemployed`, `industry_focus`
- `data/sample/cps_basic_monthly_sample.dat.gz` — 2,400 synthetic CPS person records in the bundled
  Basic Monthly layout (the `cps` command reads `.dat` and `.dat.gz` alike)

Both are **synthetic, not official BLS/Census data**; they exist to demonstrate the pipeline.

```bash
cp data/sample/unemployment_2015_2022.csv data/raw/
PYTHONPATH=src python -m unemployed_rag_pipeline.main cps --file data/sample/cps_basic_monthly_sample.dat.gz
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "unemployment in California 2020"
```

### Data Format Requirements

Once downloaded, your CSV should have these characteristics:

1. **Year column** (required for obsolescence layer):
   - Column name: `year`, `Year`, `YEAR`, `vintage`, `survey_year`, etc. (case-insensitive)
   - Values: Integers like `2020`, `2021`
   - Used by the pipeline for query-intent vintage scoring

2. **Other useful columns**:
   - `month` or `period` — Month number (1-12) if available
   - `state` — State abbreviation (CA, TX, etc.) if multi-state data
   - `unemployment_rate` — Percentage (0-100 or 0-1 depending on source)
   - `labor_force` or `employed` — Count data
   - Any descriptive metadata (industry, region, etc.)

3. **Supported file formats**:
   - `.csv` — Comma-separated (most common)
   - `.tsv` / `.txt` — Tab-separated values
   - `.parquet` — Apache Parquet (faster for large datasets)
   - `.dat` — CPS fixed-width microdata (parsed with a layout, then aggregated)

### Quick Example: Download and Ingest CA Unemployment

1. **Download CA unemployment from BLS**:
   - Go to https://www.bls.gov/lau/
   - Select California, pick 2015-2024, download CSV
   - Save as `data/raw/ca_unemployment.csv`

2. **Ingest it**:
   ```bash
   PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest
   # Output: "Loaded 1 file(s) into data/processed/unemployed_rag.duckdb"
   ```

3. **Build indexes**:
   ```bash
   PYTHONPATH=src python -m unemployed_rag_pipeline.main index
   # Output: "Indexed NNN row(s) from 1 table(s) into data/indexes/"
   ```

4. **Query it**:
   ```bash
   PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "unemployment in California from 2015 to 2020"
   ```

The pipeline will:
- Auto-detect the `year` column
- Create a `vintage_year` field in the database
- Index documents with vintage metadata
- Score your query results based on the vintage alignment

### Regenerating from Scratch

If you modify source data:
```bash
# Clear and reload data
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear

# Clear and rebuild indexes
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear

# Query the updated indexes
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "your query here"
```

## CLI commands

Show available commands:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main --help
```

### Ingest data

Load supported raw files into DuckDB:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest
```

Rebuild the database from scratch:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
```

### Download data from FRED

Fetch a FRED series into `data/raw/`:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main fred --series-id UNRATE
```

Download California unemployment and ingest it immediately:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main fred --series-id CAUR --start-date 2015-01-01 --end-date 2024-12-31 --ingest
```

Common FRED series IDs:
- `UNRATE` — U.S. unemployment rate
- `CAUR` — California unemployment rate
- `TXUR` — Texas unemployment rate
- `FLUR` — Florida unemployment rate

The command saves a CSV with `date`, `year`, `month`, `value`, `series_id`, and `vintage_year` columns. You can then run `ingest`, `index`, and `qa` as usual.

### Build indexes

Create the BM25 corpus and Chroma semantic index from the DuckDB tables:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index
```

Rebuild indexes from scratch:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
```

### Download Census QWI data

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qwi --state 13 --years 2019,2020 --ingest
```

### Parse CPS microdata

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main cps --file data/raw/cps_2020.dat --layout data/raw/cps_2020.xml --ingest
```

### Run QA

Run a single query:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "unemployment in California in 2020"
```

Start interactive QA mode:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa
```

Inside the REPL: `:help`, `:sources`, `:context`, `:warnings`, `:set k <n>`, `:exit`.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--k` | number of documents to retrieve (default 5) |
| `--fusion rrf\|weighted` | how BM25 and semantic results are combined |
| `--semantic-weight` | semantic share of the score under weighted fusion |
| `--rrf-k` | reciprocal-rank-fusion constant |
| `--min-vintage-score` | drop documents whose query-intent vintage score is below this |
| `--no-obsolescence` | rank on retrieval score only |
| `--show-context` | print the retrieved passages |
| `--json` | emit answer, provenance, and warnings as JSON |

Each answer is followed by a provenance block — document ID, table, dataset, source file, vintage
year, which retrievers matched it, and the retrieval/vintage/combined scores — and a warnings
block covering retriever failures, missing vintages, and temporally misaligned sources.

If `OPENROUTER_API_KEY` is set, you'll see LLM-generated answers. Otherwise, you'll see formatted context with sources.

### Evaluate the query engine

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main eval
PYTHONPATH=src python -m unemployed_rag_pipeline.main eval --query "unemployment in Texas in 2019" --no-ragas
```

Questions default to `data/eval/questions.json` (a list of strings, or objects with `question` and
optional `ground_truth`). The report is written to `data/outputs/eval_results.json`. RAGAS is used
when `pip install '.[eval]'` has been run and an LLM key is configured; otherwise the offline
heuristic backend is used and says so in its notes.

### Check status

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main status
```

Prints DuckDB tables and row counts, index artifacts, which credentials are set, and whether
RAGAS is installed.

## Tests

```bash
python -m pytest -q
ruff check src tests scripts
```

The suite covers CPS layout resolution and parsing, FRED/QWI API clients (against mocked
responses — no network needed), hybrid fusion, obsolescence-aware orchestration, and evaluation.

## Architecture & Design

### Ingestion Pipeline
- **Input**: CSV, TSV, Parquet files with labor market data
- **Processing**: Reads data with pandas, auto-detects vintage columns, creates normalized `vintage_year` field
- **Output**: DuckDB tables with temporal metadata attached
- **Key Code**: [ingestion/pipeline.py](src/unemployed_rag_pipeline/ingestion/pipeline.py)

### Indexing
- **BM25**: Tokenizes text, builds term-frequency index for fast keyword search
- **Semantic**: Encodes document text with SentenceTransformer, stores embeddings in Chroma
- **Metadata**: Extracts and propagates `vintage_year` from ingested data to all indexed documents
- **Key Code**: [indexing.py](src/unemployed_rag_pipeline/indexing.py)

### Retrieval & Orchestration
- **Dual Search**: Runs both BM25 and semantic search, deduplicates results by document ID
- **Vintage Scoring**: For each result, extracts query-implied time range and scores document vintage alignment
- **Sorting**: Final results ranked by combined relevance (retrieval score × vintage score)
- **Key Code**: [rag.py](src/unemployed_rag_pipeline/rag.py), [obsolescence/freshness.py](src/unemployed_rag_pipeline/obsolescence/freshness.py)

### Answer Generation
- **OpenRouterGenerator**: Calls remote LLM via OpenAI-compatible API; includes retrieved documents and query in prompt
- **DefaultGenerator**: Stub that formats contexts without LLM calls; used when API key unavailable
- **Fallback**: If LLM call fails, gracefully degrades to formatted context
- **Key Code**: [rag.py](src/unemployed_rag_pipeline/rag.py#L65-L130)

### Query-Intent Extraction
The `query_intent_vintage_score()` function implements temporal alignment:
1. Parse query for year patterns (regex patterns for "in YEAR", "YEAR-YEAR", "since YEAR", etc.)
2. Extract document's `vintage_year` from metadata
3. Score: 1.0 if within range, else exponential decay by distance
4. Default to 1.0 if no vintage or no time intent detected
- **Key Code**: [obsolescence/freshness.py](src/unemployed_rag_pipeline/obsolescence/freshness.py#L1-L75)

## Implementation Details

### Configuration
All settings load from `.env` with sensible defaults (see [.env.example](.env.example)):
- `OPENROUTER_API_KEY`: Required for LLM generation (optional)
- `OPENROUTER_MODEL`: Defaults to free Llama 3.1 8B
- `FRED_API_KEY` / `FRED_BASE_URL`: FRED downloads
- `CENSUS_API_KEY` / `QWI_BASE_URL`: Census QWI downloads
- `CPS_LAYOUT_PATH`: Default layout for CPS `.dat` extracts
- `RAW_DATA_DIR`, `INDEX_DIR`, `DUCKDB_PATH`, `EVAL_DIR`: Configurable data directories
- `EMBEDDING_MODEL`: Defaults to all-MiniLM-L6-v2 (lightweight, 384-dim)
- `HYBRID_FUSION_METHOD`, `HYBRID_RRF_K`, `HYBRID_SEMANTIC_WEIGHT`: Fusion defaults
- `MIN_VINTAGE_SCORE`, `VINTAGE_WARN_SCORE`: Obsolescence drop and warn thresholds
- **Key Code**: [config.py](src/unemployed_rag_pipeline/config.py)

### Data Requirements
- **Vintage columns**: CSV should have a year-like column (any of: `year`, `vintage`, `survey_year`, etc.)
  - Ingestion auto-detects and normalizes to `vintage_year` int field
  - If no vintage column present, documents still work but ignore time intent in queries
- **Text content**: Data should have text fields; all non-null columns concatenated for retrieval

### Testing & Evaluation
- **Unit tests**: `tests/` — freshness and query-intent scoring, obsolescence-aware orchestration
  ([tests/test_query_engine.py](tests/test_query_engine.py)), hybrid fusion
  ([tests/test_hybrid.py](tests/test_hybrid.py)), CPS layout handling
  ([tests/test_cps.py](tests/test_cps.py)), FRED/QWI clients against mocked payloads
  ([tests/test_api_ingestion.py](tests/test_api_ingestion.py)), and evaluation
  ([tests/test_evaluation.py](tests/test_evaluation.py))
- **Pipeline evaluation**: the `eval` command, writing `data/outputs/eval_results.json`
- **Sample data**: [data/sample/](data/sample) — synthetic CSV and CPS `.dat` fixtures that
  demonstrate the full pipeline end-to-end without any credentials

## Troubleshooting & Notes

- **No vintage data**: If source CSVs lack a year/vintage column, the obsolescence layer still works but treats all documents as equally relevant temporally.
- **Slow semantic indexing**: First run downloads sentence-transformer model (~130MB). Subsequent runs are cached locally.
- **LLM failures**: If OpenRouter times out or returns an error, the pipeline gracefully falls back to DefaultGenerator (formatted context only).
- **FRED API errors**: Confirm `FRED_API_KEY` is set in `.env`, and verify the series ID exists on FRED if the download returns no data.
- **QWI returns nothing**: QWI lags by several quarters, so very recent years may be empty; also confirm the state/county FIPS codes and that the requested variables exist in the chosen dataset.
- **CPS parses but the numbers look wrong**: you are almost certainly using the wrong layout. Pass the codebook that came with your extract via `--layout`; the bundled Basic Monthly layout is only a fallback.
- **`eval` says "heuristic" backend**: RAGAS is not installed (`pip install '.[eval]'`) or no LLM key is configured. Heuristic metrics are lexical-overlap approximations, useful for regression tracking but not comparable to RAGAS scores. The stub `DefaultGenerator` echoes the query, which inflates answer relevancy — set `OPENROUTER_API_KEY` for meaningful numbers.
- **Reindexing required**: If you update source data files or change the embedding model, rerun `ingest` and `index` before querying.
- **Local-only**: The entire pipeline (DuckDB, BM25, Chroma embeddings) runs locally. Only LLM generation hits external APIs if enabled.
