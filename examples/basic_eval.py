"""
Standalone example — evaluate an agent and detect regressions.
Run: python examples/basic_eval.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.suite import EvalSuite, EvalCase, SuiteRegistry
from src.evaluator import Evaluator
from src.regression import RegressionDetector


# ── Define a simple agent (swap with your real agent) ────────────────────────

def triage_agent(incident_description: str) -> str:
    """Classifies an incident as sev1, sev2, or sev3 based on keywords."""
    desc = incident_description.lower()
    if any(k in desc for k in ("outage", "down", "unavailable")):
        return "sev1"
    if any(k in desc for k in ("degraded", "slow", "high latency")):
        return "sev2"
    return "sev3"


# ── Build an eval suite ───────────────────────────────────────────────────────

suite = EvalSuite(name="triage-agent", version="1.0")
suite.add_case(EvalCase("case-1", {"incident_description": "Service is down and unavailable"}, "sev1"))
suite.add_case(EvalCase("case-2", {"incident_description": "API high latency observed"}, "sev2"))
suite.add_case(EvalCase("case-3", {"incident_description": "Minor config drift detected"}, "sev3"))
suite.add_case(EvalCase("case-4", {"incident_description": "Complete outage in prod"}, "sev1"))
suite.add_case(EvalCase("case-5", {"incident_description": "Degraded performance on search"}, "sev2"))

# Register for continuous evaluation
registry = SuiteRegistry()
registry.enroll(suite)

# ── Run evaluation ────────────────────────────────────────────────────────────

evaluator = Evaluator()
summary = evaluator.run_and_print(triage_agent, suite)

# ── Check for regressions ─────────────────────────────────────────────────────

detector = RegressionDetector(accuracy_drop_threshold=0.05, latency_budget_ms=100)
report = detector.check("triage-agent", summary)  # sets baseline on first run

print(f"\nRegression check: {report}")

# Simulate a degraded agent to show regression detection
print("\n--- Simulating degraded agent ---")

def degraded_agent(incident_description: str) -> str:
    return "sev3"  # always returns sev3 regardless of input

_, degraded_summary = evaluator.run(degraded_agent, suite)
report2 = detector.check("triage-agent", degraded_summary)
print(f"Regression check: {report2}")
