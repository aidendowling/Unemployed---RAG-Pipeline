# Unemployed RAG Pipeline

A terminal-first RAG pipeline for exploring U.S. labor market data. The project ingests raw files into DuckDB, builds keyword and semantic retrieval indexes, and exposes a query-time QA flow with obsolescence filtering.

CS 4365/6365: Introduction to Enterprise Computing — Summer 2026

Author: Aiden Dowling

## What the project does

1. Ingests raw `.csv`, `.tsv`, `.txt`, and `.parquet` files into DuckDB.
   - Automatically detects and normalizes vintage/year columns
   - Adds `vintage_year` field to every ingested row for temporal tracking
2. Builds retrieval indexes from the DuckDB tables:
   - BM25 keyword index with tokenized corpus
   - Chroma semantic index with sentence embeddings
3. Implements a **query-intent-aware knowledge-obsolescence layer** (the novel differentiator):
   - Extracts implied time ranges from user queries (e.g., "in 2020" → [2020, 2025])
   - Scores documents based on alignment between vintage_year and query-implied time range
   - Perfect-match documents score 1.0; out-of-range documents receive exponential decay
   - Enables temporal relevance without hard filtering
4. Runs a RAG QA loop that combines semantic and keyword retrieval with obsolescence filtering.

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

See [Query Evaluation Results](data/outputs/QUERY_EVALUATION_RESULTS.txt) for measured scores on live data.

## Project layout

- `src/unemployed_rag_pipeline/config.py` — shared path and model configuration
- `src/unemployed_rag_pipeline/ingestion/pipeline.py` — raw file ingestion into DuckDB
- `src/unemployed_rag_pipeline/indexing.py` — builds BM25 and Chroma indexes
- `src/unemployed_rag_pipeline/retrieval/keyword.py` — BM25 retrieval
- `src/unemployed_rag_pipeline/retrieval/semantic.py` — Chroma semantic retrieval
- `src/unemployed_rag_pipeline/rag.py` — RAG orchestration
- `src/unemployed_rag_pipeline/obsolescence/freshness.py` — freshness and obsolescence helpers
- `src/unemployed_rag_pipeline/main.py` — CLI entry point

## Key Features

### 1. Temporal Normalization During Ingestion
The ingestion pipeline automatically detects year/vintage columns in source data and normalizes them into a `vintage_year` field. Supported column names: `year`, `vintage`, `survey_year`, `reference_year`, `data_year`.

### 2. Query-Intent-Aware Obsolescence Scoring
Unlike generic recency decay, the obsolescence layer extracts time ranges from queries and scores documents relative to query intent:
- Query parser recognizes patterns: "in 2020", "2015-2020", "since 2019", "before 2018"
- Documents within query-implied range score 1.0 (perfect relevance)
- Out-of-range documents receive exponential decay: `0.5^(distance_years/10)`
- Queries with no time context return all documents with full relevance

### 3. Dual Retrieval with Orchestration
Combines lexical and semantic search:
- **BM25**: Fast keyword matching with term-frequency scoring
- **Semantic (Chroma)**: Vector similarity using sentence embeddings
- **Orchestrator**: Deduplicates by document ID, applies vintage scores, sorts by combined relevance

### 4. LLM-Based Answer Generation
The pipeline can use OpenRouter to call remote LLMs for answer synthesis. Falls back gracefully to a stub generator if no API key is configured.

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
   python -c "from src.unemployed_rag_pipeline import config; print(bool(config.FRED_API_KEY))"
   ```

## Data directories

### Directory Structure
- **Raw input**: `data/raw/` — Place raw CSV/TSV/Parquet files here before ingestion
- **Processed**: `data/processed/unemployed_rag.duckdb` — DuckDB database created after ingestion
- **Indexes**: `data/indexes/` — BM25 corpus and Chroma vector index created after indexing
- **Outputs**: `data/outputs/` — Query evaluation results and test outputs

### Getting Labor Market Data

The pipeline is designed for U.S. labor market datasets. Here are **detailed step-by-step instructions** to download real data:

#### Option A: **BLS Local Area Unemployment Statistics (LAUS)** — Recommended for state-level data

This is the exact source used for the sample data in this repo.

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

You'll need to add a `year` column. This can be done with a simple Python script:

```python
import pandas as pd

# Read FRED data
df = pd.read_csv('data/raw/fred_unemployment_rate_ca.csv')
# Convert DATE to year/month
df['DATE'] = pd.to_datetime(df['DATE'])
df['year'] = df['DATE'].dt.year
df['month'] = df['DATE'].dt.month
# Rename the value column to something descriptive
df.rename(columns={'UNRATE': 'unemployment_rate'}, inplace=True)
# Save back
df.to_csv('data/raw/fred_unemployment_rate_ca_cleaned.csv', index=False)
```

#### Option C: **Use the Sample Data** — Included for immediate testing

We've included a pre-formatted sample dataset:
- **File**: `data/raw/unemployment_2015_2022.csv`
- **Records**: 51 monthly snapshots across California, Texas, Florida
- **Years covered**: 2015-2022
- **Columns**: `year`, `month`, `state`, `unemployment_rate`, `total_labor_force`, `employed`, `unemployed`, `industry_focus`
- **Ready to use**: No preprocessing needed; just ingest and query

Run the pipeline with the sample data:
```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest
PYTHONPATH=src python -m unemployed_rag_pipeline.main index
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
   - `.tsv` — Tab-separated values
   - `.parquet` — Apache Parquet (faster for large datasets)

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

### Run QA

Run a single query:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "What is the unemployment rate?" --index-dir data/indexes
```

Start interactive QA mode:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --index-dir data/indexes
```

The QA command uses both retrievers, applies the obsolescence filter based on vintage scores, and generates answers via the configured LLM (or stub if no API key).

**Example with real labor market data**:
```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
PYTHONPATH=src python -m unemployed_rag_pipeline.main qa --query "unemployment in California 2020"
```

If `OPENROUTER_API_KEY` is set, you'll see LLM-generated answers. Otherwise, you'll see formatted context with sources.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

Run query-intent evaluation:

```bash
python test_query_intent.py
```

This generates `data/outputs/query_test_results.json` and prints human-readable evaluation.

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
All settings load from `.env` with sensible defaults:
- `OPENROUTER_API_KEY`: Required for LLM generation (optional)
- `OPENROUTER_MODEL`: Defaults to free Llama 3.1 8B
- `RAW_DATA_DIR`, `INDEX_DIR`, `DUCKDB_PATH`: Configurable data directories
- `EMBEDDING_MODEL`: Defaults to all-MiniLM-L6-v2 (lightweight, 384-dim)
- **Key Code**: [config.py](src/unemployed_rag_pipeline/config.py)

### Data Requirements
- **Vintage columns**: CSV should have a year-like column (any of: `year`, `vintage`, `survey_year`, etc.)
  - Ingestion auto-detects and normalizes to `vintage_year` int field
  - If no vintage column present, documents still work but ignore time intent in queries
- **Text content**: Data should have text fields; all non-null columns concatenated for retrieval

### Testing & Evaluation
- **Unit tests**: [tests/test_freshness.py](tests/test_freshness.py), [tests/test_rag.py](tests/test_rag.py)
  - Test query-intent extraction (5 patterns)
  - Test vintage scoring logic and edge cases
  - Test orchestration (dedup, filtering, sorting)
- **Live evaluation**: [test_query_intent.py](test_query_intent.py)
  - Runs against indexed labor market data
  - Reports vintage scores for each retrieval result
  - Outputs JSON and human-readable results to `data/outputs/`
- **Sample data**: [data/raw/unemployment_2015_2022.csv](data/raw/unemployment_2015_2022.csv)
  - 51 rows of U.S. labor data (California, Texas, Florida; 2015-2022)
  - Demonstrates full pipeline end-to-end
  - See [Query Evaluation Results](data/outputs/QUERY_EVALUATION_RESULTS.txt) for live measurements

## Troubleshooting & Notes

- **No vintage data**: If source CSVs lack a year/vintage column, the obsolescence layer still works but treats all documents as equally relevant temporally.
- **Slow semantic indexing**: First run downloads sentence-transformer model (~130MB). Subsequent runs are cached locally.
- **LLM failures**: If OpenRouter times out or returns an error, the pipeline gracefully falls back to DefaultGenerator (formatted context only).
- **FRED API errors**: Confirm `FRED_API_KEY` is set in `.env`, and verify the series ID exists on FRED if the download returns no data.
- **Reindexing required**: If you update source data files or change the embedding model, rerun `ingest` and `index` before querying.
- **Local-only**: The entire pipeline (DuckDB, BM25, Chroma embeddings) runs locally. Only LLM generation hits external APIs if enabled.
