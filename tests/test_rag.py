from datetime import datetime, timedelta, timezone

from unemployed_rag_pipeline.obsolescence import freshness
from unemployed_rag_pipeline.rag import RAGOrchestrator, RAGPipeline


class DummyEmbedder:
    def embed(self, text):
        return [1.0, 2.0]


class DummyRetriever:
    def retrieve(self, embedding, top_k=5):
        return [
            {"id": "a", "content": "Document A", "updated_at": None},
            {"id": "b", "content": "Document B", "updated_at": None},
        ]


class DummyGenerator:
    def generate(self, prompt):
        return "This is a generated answer."


def test_rag_basic():
    emb = DummyEmbedder()
    ret = DummyRetriever()
    gen = DummyGenerator()
    pipeline = RAGPipeline(emb, ret, gen)
    out = pipeline.answer("What is X?", top_k=2)
    assert out["answer"] == "This is a generated answer."
    assert "a" in out["sources"]


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def search(self, query, k=5):
        return self._results


def test_rag_orchestrator_filters_and_dedupes():
    now = datetime.now(tz=timezone.utc)
    recent = (now - timedelta(days=5)).isoformat()
    old = (now - timedelta(days=400)).isoformat()

    # doc2 is obsolete, doc1 duplicated with lower score on retriever2
    retr1_results = [
        {"id": "doc1", "text": "Text A", "score": 0.9, "metadata": {"updated_at": recent}, "table_name": "t1"},
        {"id": "doc2", "text": "Old text", "score": 0.8, "metadata": {"updated_at": old}, "table_name": "t2"},
    ]

    retr2_results = [
        {"id": "doc1", "text": "Text A duplicate", "score": 0.1, "metadata": {"updated_at": recent}, "table_name": "t1"},
        {"id": "doc3", "text": "Text C", "score": 0.7, "metadata": {"updated_at": recent}, "table_name": "t3"},
    ]

    r1 = FakeRetriever(retr1_results)
    r2 = FakeRetriever(retr2_results)

    orchestrator = RAGOrchestrator([r1, r2], obsolescence_filter=lambda r: not freshness.is_obsolete(r.get("metadata")))

    out = orchestrator.answer("dummy query", k=5)

    # doc2 should be filtered out; doc1 should be deduped keeping the higher score
    ids = {s["id"] for s in out["sources"]}
    assert "doc2" not in ids
    assert "doc1" in ids and "doc3" in ids
    # contexts should include the source markers
    assert any("source=t1" in c for c in out["contexts"])
