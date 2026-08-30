"""
Regression detector — compare a new eval run against a stored baseline.

Ship agents with confidence: if accuracy drops more than the threshold
or latency spikes beyond the budget, the regression detector blocks the release.
"""

from dataclasses import dataclass
from .metrics import MetricSummary


@dataclass
class RegressionReport:
    baseline_accuracy: float
    current_accuracy: float
    accuracy_delta: float        # positive = improvement, negative = regression
    baseline_latency_ms: float
    current_latency_ms: float
    latency_delta_ms: float
    has_regression: bool
    reason: str = ""

    def __str__(self) -> str:
        status = "REGRESSION DETECTED" if self.has_regression else "OK"
        return (
            f"[{status}] Accuracy: {self.baseline_accuracy:.1%} → {self.current_accuracy:.1%} "
            f"({self.accuracy_delta:+.1%}) | "
            f"Latency: {self.baseline_latency_ms:.0f}ms → {self.current_latency_ms:.0f}ms "
            f"({self.latency_delta_ms:+.0f}ms)"
            + (f" | {self.reason}" if self.reason else "")
        )


class RegressionDetector:
    """
    Compares current eval results to a stored baseline.

    Default thresholds (override per suite as needed):
    - accuracy_drop_threshold: block if accuracy drops by more than 5pp
    - latency_budget_ms: block if p95 latency exceeds this value
    """

    def __init__(
        self,
        accuracy_drop_threshold: float = 0.05,
        latency_budget_ms: float = 5000.0,
    ) -> None:
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.latency_budget_ms = latency_budget_ms
        self._baselines: dict[str, MetricSummary] = {}

    def set_baseline(self, suite_name: str, summary: MetricSummary) -> None:
        self._baselines[suite_name] = summary

    def check(self, suite_name: str, current: MetricSummary) -> RegressionReport:
        baseline = self._baselines.get(suite_name)
        if baseline is None:
            # No baseline yet — first run becomes the baseline
            self.set_baseline(suite_name, current)
            return RegressionReport(
                baseline_accuracy=current.accuracy,
                current_accuracy=current.accuracy,
                accuracy_delta=0.0,
                baseline_latency_ms=current.p95_latency_ms,
                current_latency_ms=current.p95_latency_ms,
                latency_delta_ms=0.0,
                has_regression=False,
                reason="First run — baseline established.",
            )

        accuracy_delta = current.accuracy - baseline.accuracy
        latency_delta = current.p95_latency_ms - baseline.p95_latency_ms

        reasons = []
        has_regression = False

        if accuracy_delta < -self.accuracy_drop_threshold:
            has_regression = True
            reasons.append(
                f"Accuracy dropped {-accuracy_delta:.1%} (threshold: {self.accuracy_drop_threshold:.1%})"
            )
        if current.p95_latency_ms > self.latency_budget_ms:
            has_regression = True
            reasons.append(
                f"p95 latency {current.p95_latency_ms:.0f}ms exceeds budget {self.latency_budget_ms:.0f}ms"
            )

        return RegressionReport(
            baseline_accuracy=baseline.accuracy,
            current_accuracy=current.accuracy,
            accuracy_delta=accuracy_delta,
            baseline_latency_ms=baseline.p95_latency_ms,
            current_latency_ms=current.p95_latency_ms,
            latency_delta_ms=latency_delta,
            has_regression=has_regression,
            reason="; ".join(reasons),
        )
