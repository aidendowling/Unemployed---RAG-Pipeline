# Unemployed---RAG-Pipeline
Unemployed is a reproducible, terminal-based analytics platform that lets researchers query decades of U.S. labor market data using natural language, stored in a RAG database.

## Baseline Python project structure

```
.
├── pyproject.toml
└── src/
    └── unemployed_rag_pipeline/
        ├── __init__.py
        └── main.py
```

Run the baseline CLI module locally:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main
```
=======

A terminal-based RAG pipeline for big labor market data integration and knowledge obsolescence detection.
CS 4365/6365: Introduction to Enterprise Computing — Summer 2026, Georgia Institute of Technology
Author: Aiden Dowling

## Why This Project

Most labor-market analytics tools are either static dashboards (no natural-language interface) or general-purpose LLMs (no grounding in real survey data, and no awareness of when their training data went stale). LaborLens sits in between: a small, reproducible RAG system purpose-built for longitudinal socioeconomic data, with explicit handling of knowledge obsolescence — the fact that labor statistics are constantly revised, re-vintaged, and structurally redefined (e.g., CPS survey redesigns in 1994 and 2003).

## Tech Stack

Data wrangling: Python, Pandas, DuckDB
Embeddings & retrieval: LangChain, sentence-transformers, ChromaDB, BM25Okapi
Generation: OpenRouter API (free-tier model) via LangChain's ChatOpenAI client
Evaluation: RAGAS
Interface: Python CLI

## Data ingestion wiring

The first working part of the pipeline is the ingestion layer. It scans a raw data directory, loads supported files into DuckDB, and creates one table per file.

Supported input formats:

- .csv
- .tsv
- .txt
- .parquet

Default locations:

- Raw inputs: data/raw
- DuckDB database: data/processed/unemployed_rag.duckdb

Run ingestion from the command line:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest
```

To force a rebuild of the database:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main ingest --clear
```

## Retrieval/indexing wiring

The next step is the retrieval layer. It reads the DuckDB tables produced by ingestion, turns each row into a text document, and builds two artifacts:

- A Chroma vector collection for semantic search
- A BM25 corpus for sparse lexical search

Default locations:

- DuckDB input: data/processed/unemployed_rag.duckdb
- Index output: data/indexes

Run indexing from the command line:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index
```

To rebuild everything from scratch:

```bash
PYTHONPATH=src python -m unemployed_rag_pipeline.main index --clear
```


