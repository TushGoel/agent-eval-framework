"""Tests for the regression detector."""

from src.metrics import MetricSummary
from src.regression import RegressionDetector


def _summary(accuracy: float, p95_ms: float) -> MetricSummary:
    return MetricSummary(
        total=100, passed=int(accuracy * 100), failed=int((1 - accuracy) * 100),
        accuracy=accuracy, avg_latency_ms=p95_ms * 0.7, p95_latency_ms=p95_ms,
        avg_tokens=100,
    )


def test_first_run_sets_baseline():
    detector = RegressionDetector()
    report = detector.check("my-agent", _summary(0.90, 500))
    assert not report.has_regression
    assert "baseline established" in report.reason


def test_no_regression():
    detector = RegressionDetector()
    detector.set_baseline("agent", _summary(0.90, 500))
    report = detector.check("agent", _summary(0.92, 480))
    assert not report.has_regression


def test_accuracy_regression_detected():
    detector = RegressionDetector(accuracy_drop_threshold=0.05)
    detector.set_baseline("agent", _summary(0.90, 500))
    report = detector.check("agent", _summary(0.80, 500))  # 10pp drop
    assert report.has_regression
    assert "Accuracy dropped" in report.reason


def test_latency_regression_detected():
    detector = RegressionDetector(latency_budget_ms=1000)
    detector.set_baseline("agent", _summary(0.90, 500))
    report = detector.check("agent", _summary(0.90, 1500))  # exceeds budget
    assert report.has_regression
    assert "latency" in report.reason.lower()


def test_improvement_not_regression():
    detector = RegressionDetector()
    detector.set_baseline("agent", _summary(0.80, 800))
    report = detector.check("agent", _summary(0.95, 300))
    assert not report.has_regression
    assert report.accuracy_delta > 0
