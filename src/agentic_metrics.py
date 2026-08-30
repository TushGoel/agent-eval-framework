"""
Agentic evaluation metrics — test agent BEHAVIOR, not just output.

Standard eval frameworks test what an agent said. Agentic metrics test
what an agent DID: did it call the right tools? In the right order?
Did it follow the procedure defined in the SOP or plan?

Two metrics:
  ToolCorrectness   — did the agent call the expected tools with correct args?
  PlanAdherence     — did the agent follow the expected step sequence?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """A single tool invocation recorded during agent execution."""
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    step: int = 0  # position in the agent's execution sequence


@dataclass
class ToolCorrectnessResult:
    case_id: str
    expected_tools: list[str]
    actual_tools: list[str]
    precision: float   # of tools called, what fraction were expected
    recall: float      # of expected tools, what fraction were called
    f1: float
    passed: bool
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] ToolCorrectness {self.case_id} | "
            f"precision={self.precision:.0%} recall={self.recall:.0%} f1={self.f1:.0%} | "
            f"expected={self.expected_tools} actual={self.actual_tools}"
        )


@dataclass
class PlanAdherenceResult:
    case_id: str
    expected_steps: list[str]
    actual_steps: list[str]
    adherence_score: float  # 0.0–1.0: fraction of expected steps present in order
    out_of_order: list[str]
    missing_steps: list[str]
    passed: bool
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] PlanAdherence {self.case_id} | "
            f"score={self.adherence_score:.0%} | "
            f"missing={self.missing_steps} out_of_order={self.out_of_order}"
        )


class ToolCorrectnessEvaluator:
    """
    Measures whether an agent called the correct tools.

    Precision = (correct tools called) / (total tools called) — no spurious calls.
    Recall    = (correct tools called) / (expected tools)     — no missing calls.
    F1        = harmonic mean — overall correctness.

    Usage:
        evaluator = ToolCorrectnessEvaluator(threshold=0.8)

        actual_calls = [
            ToolCall("get_logs", {"deployment_id": "d-123"}),
            ToolCall("create_ticket", {"severity": "P2"}),
        ]
        result = evaluator.evaluate(
            case_id="triage-44",
            expected_tools=["get_logs", "create_ticket"],
            actual_calls=actual_calls,
        )
        print(result)
        # [PASS] ToolCorrectness triage-44 | precision=100% recall=100% f1=100%
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        case_id: str,
        expected_tools: list[str],
        actual_calls: list[ToolCall],
    ) -> ToolCorrectnessResult:
        actual_names = [c.tool_name for c in actual_calls]
        expected_set = set(expected_tools)
        actual_set = set(actual_names)

        # Both empty = perfect score (agent correctly called nothing)
        if not expected_set and not actual_set:
            return ToolCorrectnessResult(
                case_id=case_id, expected_tools=expected_tools, actual_tools=actual_names,
                precision=1.0, recall=1.0, f1=1.0, passed=True,
            )

        true_positives = len(expected_set & actual_set)
        precision = true_positives / len(actual_set) if actual_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        spurious = sorted(actual_set - expected_set)
        missing = sorted(expected_set - actual_set)
        notes = []
        if spurious:
            notes.append(f"spurious calls: {spurious}")
        if missing:
            notes.append(f"missing calls: {missing}")

        return ToolCorrectnessResult(
            case_id=case_id,
            expected_tools=expected_tools,
            actual_tools=actual_names,
            precision=precision,
            recall=recall,
            f1=f1,
            passed=f1 >= self.threshold,
            notes="; ".join(notes),
        )


class PlanAdherenceEvaluator:
    """
    Measures whether an agent followed the expected step sequence.

    A plan is a sequence of named steps (e.g. SOP steps). The evaluator checks:
    - Are all expected steps present?
    - Are they in the correct order?

    Adherence score = (steps present in correct order) / (total expected steps)

    Usage:
        evaluator = PlanAdherenceEvaluator(threshold=0.9)

        result = evaluator.evaluate(
            case_id="triage-44",
            expected_steps=["route", "gather_context", "diagnose", "remediate"],
            actual_steps=["route", "gather_context", "remediate"],  # skipped diagnose
        )
        print(result)
        # [FAIL] PlanAdherence triage-44 | score=75% | missing=['diagnose']
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        case_id: str,
        expected_steps: list[str],
        actual_steps: list[str],
    ) -> PlanAdherenceResult:
        if not expected_steps:
            return PlanAdherenceResult(
                case_id=case_id,
                expected_steps=[],
                actual_steps=actual_steps,
                adherence_score=1.0,
                out_of_order=[],
                missing_steps=[],
                passed=True,
            )

        actual_set = set(actual_steps)
        missing = [s for s in expected_steps if s not in actual_set]

        # Check ordering: find expected steps that appear in actual, verify sequence
        present = [s for s in expected_steps if s in actual_set]
        out_of_order: list[str] = []
        if len(present) > 1:
            # Map each present step to its position in actual_steps
            positions = {s: actual_steps.index(s) for s in present if s in actual_steps}
            ordered = sorted(positions, key=lambda s: positions[s])
            expected_order = [s for s in expected_steps if s in actual_set]
            out_of_order = [
                s for i, s in enumerate(expected_order)
                if i < len(ordered) and ordered[i] != s
            ]

        # Score: fraction of expected steps present in correct relative order
        correct_count = len(present) - len(out_of_order)
        adherence_score = correct_count / len(expected_steps)

        notes = []
        if missing:
            notes.append(f"missing: {missing}")
        if out_of_order:
            notes.append(f"out of order: {out_of_order}")

        return PlanAdherenceResult(
            case_id=case_id,
            expected_steps=expected_steps,
            actual_steps=actual_steps,
            adherence_score=adherence_score,
            out_of_order=out_of_order,
            missing_steps=missing,
            passed=adherence_score >= self.threshold,
            notes="; ".join(notes),
        )
