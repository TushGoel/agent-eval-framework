"""
LLM evaluation as a CI/CD deployment gate.

Pattern from production: before promoting an agent update to production,
run a regression eval suite. If accuracy drops below threshold, block
the deployment — same as a failing unit test blocks a code merge.

This example shows how to wire agent-eval-framework into a CI/CD pipeline:
  1. Retrieve baseline metrics from the last good deployment
  2. Run the current agent version against the eval suite
  3. Compare with RegressionDetector
  4. Exit 1 (block deployment) if regression detected

In production, this runs as a Step Functions Lambda step in the deployment
pipeline — after integration tests, before promotion to production.

Usage (in CI/CD pipeline):
    python examples/cicd_gate.py --agent-version 1.4.2 --suite triage-accuracy

Exit codes:
    0 — eval passed, safe to deploy
    1 — regression detected, block deployment
    2 — first run, baseline established
"""

import sys
from src.suite import EvalSuite, EvalCase
from src.evaluator import Evaluator
from src.regression import RegressionDetector


# ── Example agent (replace with real agent callable) ─────────────────────────

def triage_agent(incident_type: str, error_message: str) -> str:
    """Stub: replace with your actual agent function."""
    known_patterns = {
        "ThrottlingException": "Tier 1 — platform throttling, retry after 5 minutes",
        "DataSetNotFoundException": "Tier 2 — builder error, dataset ID invalid",
        "FilterGroupNotFoundException": "Tier 2 — builder error, filter group missing",
        "TimeoutError": "Tier 1 — platform timeout, retry deployment",
    }
    for pattern, classification in known_patterns.items():
        if pattern in error_message:
            return classification
    return "Tier 3 — unknown pattern, escalate to human"


# ── Eval suite definition ─────────────────────────────────────────────────────

def build_triage_suite() -> EvalSuite:
    suite = EvalSuite(name="triage-accuracy", version="1.0")

    cases = [
        ("ThrottlingException on step 4", "Tier 1 — platform throttling, retry after 5 minutes"),
        ("DataSetNotFoundException: ds-abc not found", "Tier 2 — builder error, dataset ID invalid"),
        ("FilterGroupNotFoundException in dashboard", "Tier 2 — builder error, filter group missing"),
        ("TimeoutError after 900 seconds", "Tier 1 — platform timeout, retry deployment"),
        ("Unrecognized error XYZ-999", "Tier 3 — unknown pattern, escalate to human"),
    ]

    for i, (error_msg, expected) in enumerate(cases):
        suite.add_case(EvalCase(
            case_id=f"triage-{i+1}",
            input={"incident_type": "deployment_failure", "error_message": error_msg},
            expected=expected,
        ))

    return suite


# ── CI/CD gate ────────────────────────────────────────────────────────────────

def run_cicd_gate(
    accuracy_drop_threshold: float = 0.05,
    latency_budget_ms: float = 2000.0,
) -> int:
    """
    Run the eval gate. Returns exit code.

    In production:
    - Load baseline from DynamoDB (last deployed version's metrics)
    - Store new baseline in DynamoDB on success
    - Emit metrics to CloudWatch for dashboard
    """
    suite = build_triage_suite()
    evaluator = Evaluator()
    detector = RegressionDetector(
        accuracy_drop_threshold=accuracy_drop_threshold,
        latency_budget_ms=latency_budget_ms,
    )

    print(f"\nRunning eval gate: {suite.name} v{suite.version} ({len(suite)} cases)")
    results, summary = evaluator.run(triage_agent, suite)
    print(f"  {summary}")

    report = detector.check(suite.name, summary)
    print(f"\nRegression check: {report}")

    if report.reason == "First run — baseline established.":
        print("\n✅ Baseline established. First deployment — proceed.")
        return 2

    if report.has_regression:
        print("\n❌ REGRESSION DETECTED — blocking deployment")
        print(f"   Reason: {report.reason}")
        return 1

    print("\n✅ Eval passed — safe to deploy")
    return 0


if __name__ == "__main__":
    exit_code = run_cicd_gate()
    sys.exit(exit_code)
