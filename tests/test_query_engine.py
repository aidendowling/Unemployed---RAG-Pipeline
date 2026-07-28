import pytest

from unemployed_rag_pipeline.rag import (
    DefaultGenerator,
    GenerationError,
    RAGOrchestrator,
)


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def search(self, query, k=5):
        return self._results[:k]


def doc(doc_id, vintage, score, **metadata):
    return {
        "id": doc_id,
        "text": f"unemployment rate for {vintage}",
        "score": score,
        "table_name": "labor",
        "metadata": {"vintage_year": vintage, "table_name": "labor", **metadata},
    }


def test_vintage_scoring_reranks_stale_documents_below_aligned_ones():
    retriever = FakeRetriever([doc("old", 2005, 0.95), doc("aligned", 2020, 0.60)])
    orchestrator = RAGOrchestrator([retriever], vintage_scoring=True)

    response = orchestrator.answer("unemployment in 2020", k=2)

    assert [item["id"] for item in response["provenance"]] == ["aligned", "old"]
    assert response["provenance"][0]["vintage_score"] == 1.0
    assert response["provenance"][1]["vintage_score"] < 1.0
    assert response["provenance"][0]["combined_score"] == pytest.approx(0.60)


def test_vintage_scoring_disabled_keeps_raw_retrieval_order():
    retriever = FakeRetriever([doc("old", 2005, 0.95), doc("aligned", 2020, 0.60)])
    orchestrator = RAGOrchestrator([retriever], vintage_scoring=False)

    response = orchestrator.answer("unemployment in 2020", k=2)

    assert [item["id"] for item in response["provenance"]] == ["old", "aligned"]
    assert all(item["vintage_score"] == 1.0 for item in response["provenance"])


def test_min_vintage_score_drops_far_out_of_range_documents():
    retriever = FakeRetriever([doc("ancient", 1960, 0.99), doc("aligned", 2020, 0.10)])
    orchestrator = RAGOrchestrator([retriever], vintage_scoring=True, min_vintage_score=0.5)

    response = orchestrator.answer("unemployment in 2020", k=5)

    assert [item["id"] for item in response["provenance"]] == ["aligned"]


def test_warnings_flag_stale_and_vintageless_documents():
    retriever = FakeRetriever(
        [
            doc("stale", 1995, 0.9),
            {"id": "no_vintage", "text": "t", "score": 0.5, "metadata": {"table_name": "labor"}},
        ]
    )
    orchestrator = RAGOrchestrator([retriever], vintage_scoring=True, warn_vintage_score=0.6)

    warnings = orchestrator.answer("unemployment in 2020", k=5)["warnings"]

    assert any("no vintage_year" in warning for warning in warnings)
    assert any("poorly aligned" in warning for warning in warnings)


def test_warning_when_nothing_is_retrieved():
    orchestrator = RAGOrchestrator([FakeRetriever([])], vintage_scoring=True)

    response = orchestrator.answer("unemployment in 2020", k=5)

    assert response["provenance"] == []
    assert any("not grounded" in warning for warning in response["warnings"])


def test_generation_failure_falls_back_to_default_generator():
    def failing_generator(query, contexts):
        raise GenerationError("OpenRouter generation failed: 402 payment required")

    orchestrator = RAGOrchestrator(
        [FakeRetriever([doc("a", 2020, 0.9)])], generator=failing_generator
    )

    response = orchestrator.answer("unemployment in 2020", k=1)

    assert response["answer"].startswith("Retrieved contexts:")
    assert any("payment required" in warning for warning in response["warnings"])


def test_provenance_exposes_dataset_and_retriever_attribution():
    result = doc("a", 2020, 0.8, source="FRED", source_file="fred_unrate.csv")
    result["retrievers"] = ["bm25", "semantic"]
    result["component_scores"] = {"bm25": 7.1, "semantic": 0.83}
    result["fusion_method"] = "rrf"

    orchestrator = RAGOrchestrator([FakeRetriever([result])], generator=DefaultGenerator())
    provenance = orchestrator.answer("unemployment in 2020", k=1)["provenance"][0]

    assert provenance["dataset"] == "FRED"
    assert provenance["source_file"] == "fred_unrate.csv"
    assert provenance["retrievers"] == ["bm25", "semantic"]
    assert provenance["fusion_method"] == "rrf"


def test_contexts_carry_vintage_year_for_the_generator():
    orchestrator = RAGOrchestrator([FakeRetriever([doc("a", 2018, 0.8)])], vintage_scoring=True)

    contexts = orchestrator.answer("unemployment in 2018", k=1)["contexts"]

    assert "vintage_year=2018" in contexts[0]
