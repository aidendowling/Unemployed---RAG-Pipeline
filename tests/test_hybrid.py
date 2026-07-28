import pytest

from unemployed_rag_pipeline.retrieval.hybrid import HybridRetriever, fuse


class StaticRetriever:
    def __init__(self, results, error=None):
        self._results = results
        self._error = error

    def search(self, query, k=5):
        if self._error:
            raise RuntimeError(self._error)
        return self._results[:k]


BM25_RESULTS = [
    {"id": "a", "text": "doc a", "score": 12.0, "metadata": {}},
    {"id": "b", "text": "doc b", "score": 4.0, "metadata": {}},
]

SEMANTIC_RESULTS = [
    {"id": "b", "text": "doc b", "score": 0.91, "metadata": {}},
    {"id": "c", "text": "doc c", "score": 0.42, "metadata": {}},
]


def test_rrf_ranks_documents_found_by_both_retrievers_first():
    hybrid = HybridRetriever(
        {"bm25": StaticRetriever(BM25_RESULTS), "semantic": StaticRetriever(SEMANTIC_RESULTS)},
        method="rrf",
        rrf_k=10,
    )

    results = hybrid.search("unemployment", k=3)

    assert [result["id"] for result in results] == ["b", "a", "c"]
    assert results[0]["retrievers"] == ["bm25", "semantic"]
    assert results[0]["component_scores"] == {"bm25": 4.0, "semantic": 0.91}
    assert results[0]["component_ranks"] == {"bm25": 2, "semantic": 1}
    assert results[0]["score"] == pytest.approx(results[0]["fusion_score"])


def test_weighted_fusion_respects_retriever_weights():
    retrievers = {
        "bm25": StaticRetriever(BM25_RESULTS),
        "semantic": StaticRetriever(SEMANTIC_RESULTS),
    }

    keyword_heavy = HybridRetriever(
        retrievers, method="weighted", weights={"bm25": 0.9, "semantic": 0.1}
    ).search("unemployment", k=3)
    semantic_heavy = HybridRetriever(
        retrievers, method="weighted", weights={"bm25": 0.1, "semantic": 0.9}
    ).search("unemployment", k=3)

    assert keyword_heavy[0]["id"] == "a"
    assert semantic_heavy[0]["id"] == "b"


def test_failing_retriever_is_recorded_not_raised():
    hybrid = HybridRetriever(
        {
            "bm25": StaticRetriever([], error="corpus missing"),
            "semantic": StaticRetriever(SEMANTIC_RESULTS),
        }
    )

    results = hybrid.search("unemployment", k=2)

    assert [result["id"] for result in results] == ["b", "c"]
    assert "corpus missing" in hybrid.errors["bm25"]


def test_fuse_helper_matches_hybrid_retriever():
    fused = fuse(
        [("bm25", BM25_RESULTS), ("semantic", SEMANTIC_RESULTS)], method="rrf", rrf_k=10, k=3
    )

    assert [result["id"] for result in fused] == ["b", "a", "c"]


def test_unknown_fusion_method_rejected():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        HybridRetriever({"bm25": StaticRetriever(BM25_RESULTS)}, method="magic")
