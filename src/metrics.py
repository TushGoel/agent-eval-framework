"""Core metrics for measuring agent quality in production."""

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    case_id: str
    input: dict[str, Any]
    expected: Any
    actual: Any
    passed: bool
    latency_ms: float
    tokens_used: int = 0
    notes: str = ""


@dataclass
class MetricSummary:
    total: int
    passed: int
    failed: int
    accuracy: float           # pass rate 0.0–1.0
    avg_latency_ms: float
    p95_latency_ms: float
    avg_tokens: float

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.1%} ({self.passed}/{self.total}) | "
            f"Latency p50={self.avg_latency_ms:.0f}ms p95={self.p95_latency_ms:.0f}ms | "
            f"Avg tokens: {self.avg_tokens:.0f}"
        )


def compute_summary(results: list[EvalResult]) -> MetricSummary:
    if not results:
        return MetricSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    passed = sum(1 for r in results if r.passed)
    latencies = sorted(r.latency_ms for r in results)
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)

    return MetricSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        accuracy=passed / len(results),
        avg_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=latencies[p95_idx],
        avg_tokens=sum(r.tokens_used for r in results) / len(results),
    )
