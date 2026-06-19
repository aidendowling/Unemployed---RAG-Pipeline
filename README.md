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
