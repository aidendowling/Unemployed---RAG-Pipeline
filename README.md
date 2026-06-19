# Unemployed---RAG-Pipeline

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



