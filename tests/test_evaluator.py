"""Tests for the core evaluation engine."""

from src.suite import EvalSuite, EvalCase
from src.evaluator import Evaluator


def _make_suite(cases: list[tuple]) -> EvalSuite:
    suite = EvalSuite(name="test", version="1.0")
    for i, (inp, expected) in enumerate(cases):
        suite.add_case(EvalCase(case_id=f"case-{i}", input=inp, expected=expected))
    return suite


def test_perfect_agent():
    suite = _make_suite([
        ({"query": "2+2"}, "4"),
        ({"query": "3+3"}, "6"),
    ])
    agent = lambda query: str(eval(query))
    evaluator = Evaluator()
    results, summary = evaluator.run(agent, suite)
    assert summary.accuracy == 1.0
    assert summary.passed == 2
    assert summary.failed == 0


def test_partial_pass():
    suite = _make_suite([
        ({"query": "hello"}, "hello"),
        ({"query": "world"}, "wrong_answer"),
    ])
    agent = lambda query: query  # echoes input
    evaluator = Evaluator()
    _, summary = evaluator.run(agent, suite)
    assert summary.accuracy == 0.5
    assert summary.passed == 1
    assert summary.failed == 1


def test_agent_exception_counts_as_failure():
    suite = _make_suite([({"query": "boom"}, "result")])
    def crashing_agent(query): raise RuntimeError("crash")
    evaluator = Evaluator()
    results, summary = evaluator.run(crashing_agent, suite)
    assert summary.passed == 0
    assert results[0].notes == "crash"


def test_latency_recorded():
    suite = _make_suite([({"query": "x"}, "x")])
    agent = lambda query: query
    evaluator = Evaluator()
    results, _ = evaluator.run(agent, suite)
    assert results[0].latency_ms >= 0
