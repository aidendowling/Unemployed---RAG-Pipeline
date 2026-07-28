"""RAG evaluation: RAGAS metrics with an offline heuristic fallback."""

from .ragas_eval import (
    EvalCase,
    EvalRunResult,
    QueryEvaluation,
    evaluate_queries,
    heuristic_metrics,
    load_eval_cases,
    ragas_available,
)

__all__ = [
    "EvalCase",
    "EvalRunResult",
    "QueryEvaluation",
    "evaluate_queries",
    "heuristic_metrics",
    "load_eval_cases",
    "ragas_available",
]
