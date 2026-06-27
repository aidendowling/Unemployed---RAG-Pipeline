# Unemployed---RAG-Pipeline

Unemployed is a reproducible, terminal-based analytics platform for querying decades of U.S. labor market data with natural language.

CS 4365/6365: Introduction to Enterprise Computing — Summer 2026, Georgia Institute of Technology

Author: Aiden Dowling

## Current project state

The project now has three working pieces:

1. Data ingestion into DuckDB
2. Retrieval/indexing with Chroma and BM25
3. A CLI entry point that exposes both steps

## Project structure

```
.
├── .env.example
├── pyproject.toml
└── src/
    └── unemployed_rag_pipeline/
        ├── __init__.py
        ├── ingestion.py
        ├── indexing.py
        └── main.py
```

## Environment setup

Copy the example environment file before running anything:

```bash
cp .env.example .env
```

Fill in your local API key and adjust paths if needed.

## Dependencies

The project uses:

- Python 3.10+
- Pandas
- DuckDB
- PyArrow
- ChromaDB
- sentence-transformers
- rank-bm25

## CLI commands

Run the app help:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main
```

### Ingest raw files

Supported formats:

- .csv
- .tsv
- .txt
- .parquet

Default input and output paths:

- Raw inputs: data/raw
- DuckDB database: data/processed/unemployed_rag.duckdb

Command:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest
```

Force a rebuild:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
```

### Build retrieval indexes

The index step reads the DuckDB tables and creates:

- A Chroma vector collection for semantic search
- A BM25 corpus for lexical search

Default locations:

- DuckDB input: data/processed/unemployed_rag.duckdb
- Index output: data/indexes

Command:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index
```

Rebuild from scratch:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
```

## Notes

This is still the wiring stage of the pipeline. The next major step is query-time retrieval and answer generation.


