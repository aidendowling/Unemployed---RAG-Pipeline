"""Hybrid retrieval: fuse BM25 and semantic result lists into one ranking."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

RRF = "rrf"
WEIGHTED = "weighted"
FUSION_METHODS = (RRF, WEIGHTED)


class SearchRetriever(Protocol):
    """Anything exposing ``search(query, k) -> list[dict]``."""

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize scores into [0, 1]; a flat list maps to all 1.0."""
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest - lowest < 1e-12:
        return [1.0 for _ in scores]
    return [(score - lowest) / (highest - lowest) for score in scores]


class HybridRetriever:
    """Combine named retrievers with reciprocal-rank or weighted-score fusion.

    Args:
        retrievers: Mapping of retriever name to a ``search``-capable retriever.
            Names are surfaced in each result's provenance.
        method: ``"rrf"`` (rank-based, score-scale agnostic) or ``"weighted"``
            (min-max normalized scores blended by ``weights``).
        weights: Per-retriever weights for ``"weighted"`` fusion; missing
            entries default to 1.0 and weights are renormalized to sum to 1.
        rrf_k: Reciprocal-rank-fusion damping constant.
    """

    def __init__(
        self,
        retrievers: dict[str, SearchRetriever],
        *,
        method: str = RRF,
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if method not in FUSION_METHODS:
            raise ValueError(f"Unknown fusion method {method!r}; expected one of {FUSION_METHODS}.")
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever.")

        self.retrievers = retrievers
        self.method = method
        self.rrf_k = rrf_k
        self.weights = self._normalize_weights(weights or {})
        self.errors: dict[str, str] = {}

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Fill in defaults and rescale weights so they sum to 1."""
        raw = {name: float(weights.get(name, 1.0)) for name in self.retrievers}
        total = sum(raw.values())
        if total <= 0:
            return {name: 1.0 / len(raw) for name in raw}
        return {name: value / total for name, value in raw.items()}

    def _collect(self, query: str, k: int) -> dict[str, list[dict[str, Any]]]:
        """Run every retriever, recording failures instead of propagating them."""
        self.errors = {}
        collected: dict[str, list[dict[str, Any]]] = {}
        for name, retriever in self.retrievers.items():
            try:
                collected[name] = list(retriever.search(query, k))
            except Exception as error:  # noqa: BLE001 - a dead retriever must not kill the query
                self.errors[name] = str(error)
                collected[name] = []
        return collected

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the fused top-``k`` results.

        Each result carries provenance: ``retrievers`` (which lists it appeared
        in), ``component_scores``, ``component_ranks``, ``fusion_score``, and
        ``score`` (set to ``fusion_score`` so downstream consumers are unchanged).
        """
        per_retriever = self._collect(query, k)

        fused: dict[str, dict[str, Any]] = {}
        contributions: dict[str, float] = {}

        for name, results in per_retriever.items():
            normalized = _normalize([float(result.get("score", 0.0)) for result in results])
            for rank, (result, norm_score) in enumerate(zip(results, normalized), start=1):
                doc_id = str(result.get("id"))
                entry = fused.setdefault(
                    doc_id,
                    {
                        **result,
                        "id": doc_id,
                        "retrievers": [],
                        "component_scores": {},
                        "component_ranks": {},
                        "normalized_scores": {},
                    },
                )
                entry["retrievers"].append(name)
                entry["component_scores"][name] = float(result.get("score", 0.0))
                entry["component_ranks"][name] = rank
                entry["normalized_scores"][name] = norm_score
                if not entry.get("text"):
                    entry["text"] = result.get("text", "")

                if self.method == RRF:
                    contributions[doc_id] = contributions.get(doc_id, 0.0) + 1.0 / (
                        self.rrf_k + rank
                    )
                else:
                    contributions[doc_id] = contributions.get(doc_id, 0.0) + (
                        self.weights.get(name, 0.0) * norm_score
                    )

        for doc_id, entry in fused.items():
            entry["fusion_score"] = contributions.get(doc_id, 0.0)
            entry["fusion_method"] = self.method
            entry["score"] = entry["fusion_score"]

        ranked = sorted(fused.values(), key=lambda entry: entry["fusion_score"], reverse=True)
        return ranked[:k]


def fuse(
    result_lists: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    method: str = RRF,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Fuse already-retrieved result lists without instantiating retrievers.

    Useful for evaluation and tests where the result lists are fixtures.
    """
    materialized = {name: results for name, results in result_lists}

    class _Static:
        def __init__(self, results: list[dict[str, Any]]) -> None:
            self._results = results

        def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
            return self._results[:k]

    hybrid = HybridRetriever(
        {name: _Static(results) for name, results in materialized.items()},
        method=method,
        weights=weights,
        rrf_k=rrf_k,
    )
    return hybrid.search("", k=k)
