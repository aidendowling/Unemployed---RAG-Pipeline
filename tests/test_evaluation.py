import json

from unemployed_rag_pipeline.evaluation import (
    EvalCase,
    evaluate_queries,
    heuristic_metrics,
    load_eval_cases,
)


class StubOrchestrator:
    def __init__(self, answer, contexts, warnings=None):
        self.answer_text = answer
        self.contexts = contexts
        self.warnings = warnings or []
        self.calls = []

    def answer(self, query, k=5):
        self.calls.append((query, k))
        return {
            "answer": self.answer_text,
            "contexts": self.contexts,
            "warnings": self.warnings,
        }


def test_heuristic_metrics_reward_grounded_answers():
    contexts = ["state: CA | year: 2020 | unemployment_rate: 10.2"]

    grounded = heuristic_metrics(
        "What was the unemployment rate in CA in 2020?",
        "CA unemployment rate was 10.2 in 2020.",
        contexts,
    )
    ungrounded = heuristic_metrics(
        "What was the unemployment rate in CA in 2020?",
        "Interest rates in Japan fell sharply.",
        contexts,
    )

    assert grounded["faithfulness"] > ungrounded["faithfulness"]
    assert grounded["answer_relevancy"] > ungrounded["answer_relevancy"]
    assert grounded["context_precision"] == 1.0


def test_context_recall_uses_ground_truth():
    contexts = ["state: FL | year: 2022 | unemployment_rate: 2.9"]

    with_truth = heuristic_metrics("q", "a", contexts, ground_truth="FL 2022 unemployment 2.9")
    without_truth = heuristic_metrics("q", "a", contexts)

    assert with_truth["context_recall"] > 0.0
    assert without_truth["context_recall"] == 0.0


def test_evaluate_queries_without_ragas_reports_heuristic_backend():
    orchestrator = StubOrchestrator(
        answer="CA unemployment was 10.2 percent in 2020.",
        contexts=["state: CA | year: 2020 | unemployment_rate: 10.2"],
        warnings=["1 document has no vintage_year"],
    )
    cases = [EvalCase(question="What was CA unemployment in 2020?")]

    result = evaluate_queries(orchestrator, cases, k=3, use_ragas=False)

    assert result.backend == "heuristic"
    assert orchestrator.calls == [("What was CA unemployment in 2020?", 3)]
    assert set(result.aggregate) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
    assert result.evaluations[0].warnings == ["1 document has no vintage_year"]
    assert any("heuristic" in note.lower() for note in result.notes)


def test_eval_run_result_round_trips_to_json(tmp_path):
    orchestrator = StubOrchestrator(answer="answer", contexts=["context"])
    result = evaluate_queries(orchestrator, [EvalCase(question="q")], use_ragas=False)

    path = result.write_json(tmp_path / "nested" / "eval.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["backend"] == "heuristic"
    assert payload["evaluations"][0]["question"] == "q"


def test_load_eval_cases_accepts_strings_and_objects(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps({"questions": ["plain question", {"question": "q2", "ground_truth": "gt"}]}),
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0] == EvalCase(question="plain question", ground_truth=None)
    assert cases[1].ground_truth == "gt"
