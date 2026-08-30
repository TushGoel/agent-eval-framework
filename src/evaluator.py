"""
Core evaluation engine — runs an agent against a suite and returns results.
"""

import time
from typing import Callable, Any
from .suite import EvalSuite
from .metrics import EvalResult, MetricSummary, compute_summary


AgentFn = Callable[[dict[str, Any]], Any]


class Evaluator:
    """
    Runs an agent function against every case in an eval suite.

    Usage:
        evaluator = Evaluator()
        summary = evaluator.run(agent_fn=my_agent, suite=my_suite)
        print(summary)
    """

    def run(self, agent_fn: AgentFn, suite: EvalSuite) -> tuple[list[EvalResult], MetricSummary]:
        results: list[EvalResult] = []

        for case in suite.cases:
            start = time.monotonic()
            error = None
            actual = None

            try:
                actual = agent_fn(**case.input)
            except Exception as exc:
                error = str(exc)
                actual = None

            latency_ms = (time.monotonic() - start) * 1000
            passed = False if error else suite.evaluate_case(case.expected, actual)

            results.append(EvalResult(
                case_id=case.case_id,
                input=case.input,
                expected=case.expected,
                actual=actual,
                passed=passed,
                latency_ms=latency_ms,
                notes=error or "",
            ))

        summary = compute_summary(results)
        return results, summary

    def run_and_print(self, agent_fn: AgentFn, suite: EvalSuite) -> MetricSummary:
        results, summary = self.run(agent_fn, suite)
        print(f"\nEval: {suite.name} v{suite.version} ({len(suite)} cases)")
        print(f"  {summary}")
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n  Failed cases ({len(failures)}):")
            for r in failures[:5]:  # show first 5 failures
                print(f"    [{r.case_id}] expected={r.expected!r} actual={r.actual!r}")
            if len(failures) > 5:
                print(f"    ... and {len(failures) - 5} more")
        return summary
