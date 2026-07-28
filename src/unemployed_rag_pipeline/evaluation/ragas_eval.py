"""Evaluate the RAG query engine with RAGAS, falling back to offline heuristics.

RAGAS metrics are LLM-judged, so they need both the optional ``ragas`` extra and
a judge model. When either is missing this module computes lexical-overlap
approximations of the same four metrics so ``eval`` always produces numbers, and
labels the run ``heuristic`` so the two are never confused.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what",
    "when", "which", "who", "why", "with",
}

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation question, optionally with a reference answer."""

    question: str
    ground_truth: str | None = None


@dataclass(slots=True)
class QueryEvaluation:
    """Per-question evaluation output."""

    question: str
    answer: str
    contexts: list[str]
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class EvalRunResult:
    """Aggregate result of an evaluation run."""

    backend: str
    evaluations: list[QueryEvaluation]
    aggregate: dict[str, float]
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "backend": self.backend,
            "generated_at": self.generated_at,
            "aggregate": self.aggregate,
            "notes": self.notes,
            "evaluations": [item.to_dict() for item in self.evaluations],
        }

    def write_json(self, path: Path) -> Path:
        """Write the run to ``path`` as JSON and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def ragas_available() -> bool:
    """Return True if the optional ``ragas`` package is importable."""
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file.

    Accepts a list of strings, or a list of ``{"question": ..., "ground_truth": ...}``
    objects, or ``{"questions": [...]}``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["questions"] if isinstance(payload, dict) else payload

    cases: list[EvalCase] = []
    for item in items:
        if isinstance(item, str):
            cases.append(EvalCase(question=item))
        else:
            cases.append(
                EvalCase(
                    question=str(item["question"]),
                    ground_truth=item.get("ground_truth"),
                )
            )
    return cases


def _tokens(text: str) -> set[str]:
    """Return the informative lowercase word tokens in ``text``."""
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def _overlap(source: set[str], target: set[str]) -> float:
    """Return the fraction of ``source`` tokens also present in ``target``."""
    if not source:
        return 0.0
    return len(source & target) / len(source)


def heuristic_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, float]:
    """Compute lexical-overlap approximations of the four RAGAS metrics.

    These are deterministic and require no LLM judge:

    - ``faithfulness``: share of answer tokens that appear in the contexts.
    - ``answer_relevancy``: share of question tokens covered by the answer.
    - ``context_precision``: share of retrieved contexts that touch the question.
    - ``context_recall``: share of ground-truth tokens present in the contexts
      (``0.0`` when no reference answer was supplied).
    """
    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts))

    relevant_contexts = sum(1 for context in contexts if _tokens(context) & question_tokens)
    context_precision = relevant_contexts / len(contexts) if contexts else 0.0

    if ground_truth:
        context_recall = _overlap(_tokens(ground_truth), context_tokens)
    else:
        context_recall = 0.0

    return {
        "faithfulness": round(_overlap(answer_tokens, context_tokens), 4),
        "answer_relevancy": round(_overlap(question_tokens, answer_tokens), 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
    }


def _aggregate(evaluations: list[QueryEvaluation]) -> dict[str, float]:
    """Average each metric across evaluations."""
    if not evaluations:
        return {name: 0.0 for name in METRIC_NAMES}

    aggregate: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [item.metrics.get(name, 0.0) for item in evaluations]
        aggregate[name] = round(sum(values) / len(values), 4)
    return aggregate


def _run_ragas(
    evaluations: list[QueryEvaluation], cases: list[EvalCase]
) -> tuple[dict[str, float], list[str]]:
    """Score answers with RAGAS. Returns ``({}, [reason])`` if it cannot run."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as error:
        return {}, [f"RAGAS unavailable ({error}); install with: pip install '.[eval]'"]

    dataset = Dataset.from_dict(
        {
            "question": [item.question for item in evaluations],
            "answer": [item.answer for item in evaluations],
            "contexts": [item.contexts for item in evaluations],
            "ground_truth": [case.ground_truth or "" for case in cases],
        }
    )

    try:
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    except Exception as error:  # noqa: BLE001 - RAGAS surfaces judge/LLM errors broadly
        return {}, [f"RAGAS evaluation failed ({error}); reporting heuristic metrics instead."]

    frame = scores.to_pandas()
    for index, item in enumerate(evaluations):
        item.metrics = {
            name: round(float(frame[name].iloc[index]), 4)
            for name in METRIC_NAMES
            if name in frame.columns
        }
    return _aggregate(evaluations), []


def evaluate_queries(
    orchestrator: Any,
    cases: list[EvalCase],
    *,
    k: int = 5,
    use_ragas: bool = True,
) -> EvalRunResult:
    """Answer every case with ``orchestrator`` and score the results.

    Args:
        orchestrator: Object exposing ``answer(query, k) -> dict``.
        cases: Evaluation questions.
        k: Documents to retrieve per question.
        use_ragas: Attempt RAGAS first; heuristics are used when it is
            unavailable or errors out.

    Returns:
        EvalRunResult whose ``backend`` is ``"ragas"`` or ``"heuristic"``.
    """
    evaluations: list[QueryEvaluation] = []
    for case in cases:
        response = orchestrator.answer(case.question, k=k)
        contexts = list(response.get("contexts", []))
        answer = str(response.get("answer", ""))
        evaluations.append(
            QueryEvaluation(
                question=case.question,
                answer=answer,
                contexts=contexts,
                metrics=heuristic_metrics(case.question, answer, contexts, case.ground_truth),
                warnings=list(response.get("warnings", [])),
            )
        )

    notes: list[str] = []
    if use_ragas:
        aggregate, ragas_notes = _run_ragas(evaluations, cases)
        notes.extend(ragas_notes)
        if aggregate:
            return EvalRunResult(backend="ragas", evaluations=evaluations, aggregate=aggregate)
    else:
        notes.append("RAGAS skipped (--no-ragas); reporting heuristic metrics.")

    notes.append(
        "Heuristic metrics are lexical-overlap approximations, not LLM-judged RAGAS scores."
    )
    return EvalRunResult(
        backend="heuristic",
        evaluations=evaluations,
        aggregate=_aggregate(evaluations),
        notes=notes,
    )
