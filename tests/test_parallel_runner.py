"""Tests for parallel evaluator and flaky test detector."""

import threading
import pytest
from src.suite import EvalSuite, EvalCase
from src.parallel_runner import ParallelEvaluator, FlakyDetector, FlakyResult


def _suite(n=5):
    suite = EvalSuite(name="test-suite", version="1.0")
    for i in range(n):
        suite.add_case(EvalCase(
            case_id=f"case-{i}",
            input=f"input-{i}",
            expected=f"output-{i}",
        ))
    return suite


def _perfect_agent(suite):
    case_map = {c.input: c.expected for c in suite.cases}
    return lambda inp: case_map.get(inp, "wrong")


# ── ParallelEvaluator ─────────────────────────────────────────────────────────

def test_parallel_all_pass():
    suite = _suite()
    runner = ParallelEvaluator(max_workers=5)
    results, summary = runner.run(_perfect_agent(suite), suite)
    assert summary.accuracy == 1.0
    assert len(results) == 5


def test_parallel_preserves_order():
    suite = _suite()
    runner = ParallelEvaluator(max_workers=5)
    results, _ = runner.run(_perfect_agent(suite), suite)
    for i, r in enumerate(results):
        assert r.case_id == f"case-{i}"


def test_parallel_handles_agent_exception():
    suite = _suite()

    def crashing_agent(inp):
        raise RuntimeError("LLM error")

    runner = ParallelEvaluator(max_workers=3)
    results, summary = runner.run(crashing_agent, suite)
    assert summary.accuracy == 0.0
    assert all("exception" in r.notes for r in results)


def test_parallel_partial_pass():
    suite = _suite(5)
    case_map = {c.input: c.expected for c in suite.cases}

    def partial_agent(inp):
        # Only pass first 3
        idx = int(inp.split("-")[1])
        return case_map[inp] if idx < 3 else "wrong"

    runner = ParallelEvaluator(max_workers=5)
    results, summary = runner.run(partial_agent, suite)
    assert summary.accuracy == pytest.approx(0.6)


# ── FlakyDetector ─────────────────────────────────────────────────────────────

def test_stable_pass_not_flaky():
    suite = _suite()
    detector = FlakyDetector(runs_per_case=3)
    flaky_results = detector.detect(_perfect_agent(suite), suite)
    for r in flaky_results:
        assert not r.is_flaky


def test_flaky_case_detected():
    suite = EvalSuite(name="flaky-suite", version="1.0")
    suite.add_case(EvalCase("flaky-case", "input", "expected"))

    lock = threading.Lock()
    counter = [0]

    def flaky_agent(inp):
        with lock:
            counter[0] += 1
            n = counter[0]
        # Even calls pass, odd calls fail
        return "expected" if n % 2 == 0 else "wrong"

    detector = FlakyDetector(runs_per_case=4, max_workers=1)  # serial to control counter
    flaky_results = detector.detect(flaky_agent, suite)
    assert flaky_results[0].is_flaky


def test_flaky_detector_invalid_runs():
    with pytest.raises(ValueError):
        FlakyDetector(runs_per_case=1)


def test_flaky_result_str():
    r = FlakyResult(case_id="test", runs=5, pass_count=3, fail_count=2)
    assert "FLAKY" in str(r)
    assert "3/5" in str(r)


def test_stable_fail_not_flaky():
    suite = EvalSuite(name="s", version="1.0")
    suite.add_case(EvalCase("bad", "input", "expected"))
    detector = FlakyDetector(runs_per_case=3)
    results = detector.detect(lambda inp: "wrong", suite)
    assert not results[0].is_flaky
    assert "STABLE_FAIL" in str(results[0])
