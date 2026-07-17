# Unemployed RAG Pipeline

A terminal-first RAG pipeline for exploring U.S. labor market data. The project ingests raw files into DuckDB, builds keyword and semantic retrieval indexes, and exposes a query-time QA flow with obsolescence filtering.

CS 4365/6365: Introduction to Enterprise Computing — Summer 2026

Author: Aiden Dowling

## What the project does

1. Ingests raw `.csv`, `.tsv`, `.txt`, and `.parquet` files into DuckDB.
2. Builds retrieval indexes from the DuckDB tables:
   - BM25 keyword index
   - Chroma semantic index
3. Runs a RAG QA loop that combines semantic and keyword retrieval.
4. Filters stale documents with the obsolescence layer.

## Project layout

- `src/unemployed_rag_pipeline/config.py` — shared path and model configuration
- `src/unemployed_rag_pipeline/ingestion/pipeline.py` — raw file ingestion into DuckDB
- `src/unemployed_rag_pipeline/indexing.py` — builds BM25 and Chroma indexes
- `src/unemployed_rag_pipeline/retrieval/keyword.py` — BM25 retrieval
- `src/unemployed_rag_pipeline/retrieval/semantic.py` — Chroma semantic retrieval
- `src/unemployed_rag_pipeline/rag.py` — RAG orchestration
- `src/unemployed_rag_pipeline/obsolescence/freshness.py` — freshness and obsolescence helpers
- `src/unemployed_rag_pipeline/main.py` — CLI entry point

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you want to use a custom environment file, copy the example first:

```bash
cp .env.example .env
```

## Data directories

- Raw input: `data/raw/`
- DuckDB output: `data/processed/unemployed_rag.duckdb`
- Retrieval indexes: `data/indexes/`

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

The QA command uses both retrievers and applies the obsolescence filter to drop stale or deprecated results.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

## Notes

- The semantic retriever expects a built Chroma index in `data/indexes/chroma/`.
- The BM25 retriever expects `data/indexes/bm25_corpus.json`.
- If you update the source data, rerun `ingest` and `index` before using `qa`.

