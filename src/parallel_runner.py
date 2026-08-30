"""
Parallel eval runner and flaky test detector.

Production agents are non-deterministic: the same input can produce different
outputs on different runs. Standard eval frameworks run each case once — which
means non-deterministic failures look like real regressions, and non-deterministic
passes hide real bugs.

This module adds two capabilities:

1. ParallelEvaluator — runs all test cases concurrently (threads), not sequentially.
   At 100 test cases × 500ms each, sequential = 50s. Parallel = ~500ms.

2. FlakyDetector — runs each case N times and marks it flaky if it passes some
   runs and fails others. Flaky cases are excluded from regression gating until
   they're stabilized.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .evaluator import EvalResult
from .suite import EvalCase, EvalSuite
from .metrics import MetricSummary, compute_summary


@dataclass
class FlakyResult:
    case_id: str
    runs: int
    pass_count: int
    fail_count: int

    @property
    def is_flaky(self) -> bool:
        return self.pass_count > 0 and self.fail_count > 0

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.runs if self.runs > 0 else 0.0

    def __str__(self) -> str:
        tag = "FLAKY" if self.is_flaky else ("STABLE_PASS" if self.fail_count == 0 else "STABLE_FAIL")
        return f"[{tag}] {self.case_id} pass={self.pass_count}/{self.runs} ({self.pass_rate:.0%})"


class ParallelEvaluator:
    """
    Runs all eval cases concurrently using a thread pool.

    For I/O-bound agents (LLM API calls, tool invocations), parallelism
    reduces wall-clock eval time from O(N × latency) to O(latency).

    Usage:
        runner = ParallelEvaluator(max_workers=20)
        results, summary = runner.run(agent_fn, suite)
        # All cases ran concurrently
    """

    def __init__(self, max_workers: int = 10) -> None:
        self.max_workers = max_workers

    def run(
        self,
        agent: Callable[[str], str],
        suite: EvalSuite,
    ) -> tuple[list[EvalResult], MetricSummary]:
        """Run all cases in parallel. Returns results in original case order."""
        cases = list(suite.cases)
        results: list[Optional[EvalResult]] = [None] * len(cases)

        def run_case(idx: int, case: EvalCase) -> tuple[int, EvalResult]:
            import time
            start = time.monotonic()
            try:
                actual = agent(case.input)
                passed = suite.evaluate_case(case.expected, actual)
                latency_ms = (time.monotonic() - start) * 1000
                return idx, EvalResult(
                    case_id=case.case_id,
                    input=case.input,
                    expected=case.expected,
                    actual=actual,
                    passed=passed,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                return idx, EvalResult(
                    case_id=case.case_id,
                    input=case.input,
                    expected=case.expected,
                    actual="",
                    passed=False,
                    latency_ms=latency_ms,
                    notes=f"exception: {exc}",
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(run_case, i, case) for i, case in enumerate(cases)]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        final = [r for r in results if r is not None]
        return final, compute_summary(final)


class FlakyDetector:
    """
    Detects non-deterministic (flaky) test cases by running each case N times.

    A case is flaky if it passes at least once AND fails at least once across
    N runs. Flaky cases should be excluded from regression gating — they'll
    cause false positives that erode trust in the eval suite.

    Usage:
        detector = FlakyDetector(runs_per_case=5)
        flaky_results = detector.detect(agent_fn, suite)

        for r in flaky_results:
            if r.is_flaky:
                print(f"Flaky: {r}")
    """

    def __init__(self, runs_per_case: int = 5, max_workers: int = 5) -> None:
        if runs_per_case < 2:
            raise ValueError("runs_per_case must be >= 2 to detect flakiness")
        self.runs_per_case = runs_per_case
        self.max_workers = max_workers

    def detect(
        self,
        agent: Callable[[str], str],
        suite: EvalSuite,
    ) -> list[FlakyResult]:
        """
        Run each case runs_per_case times and report which are flaky.
        Cases are run with limited parallelism to avoid rate limiting.
        """
        results: list[FlakyResult] = []

        for case in suite.cases:
            pass_count = 0
            fail_count = 0

            def run_once(c=case) -> bool:
                try:
                    actual = agent(c.input)
                    return suite.evaluate_case(c.expected, actual)
                except Exception:
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(run_once) for _ in range(self.runs_per_case)]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        pass_count += 1
                    else:
                        fail_count += 1

            results.append(FlakyResult(
                case_id=case.case_id,
                runs=self.runs_per_case,
                pass_count=pass_count,
                fail_count=fail_count,
            ))

        return results

    def flaky_case_ids(self, results: list[FlakyResult]) -> list[str]:
        return [r.case_id for r in results if r.is_flaky]
