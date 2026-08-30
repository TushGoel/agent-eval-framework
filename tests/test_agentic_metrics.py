"""Tests for agentic evaluation metrics — tool correctness and plan adherence."""

import pytest
from src.agentic_metrics import (
    ToolCall,
    ToolCorrectnessEvaluator,
    PlanAdherenceEvaluator,
)


# ── Tool Correctness ──────────────────────────────────────────────────────────

def test_perfect_tool_correctness():
    evaluator = ToolCorrectnessEvaluator(threshold=0.8)
    calls = [ToolCall("get_logs"), ToolCall("create_ticket")]
    result = evaluator.evaluate("case-1", ["get_logs", "create_ticket"], calls)
    assert result.passed
    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_spurious_tool_call_lowers_precision():
    evaluator = ToolCorrectnessEvaluator(threshold=0.8)
    calls = [ToolCall("get_logs"), ToolCall("create_ticket"), ToolCall("rollback")]
    result = evaluator.evaluate("case-1", ["get_logs", "create_ticket"], calls)
    assert result.precision < 1.0
    assert "rollback" in result.notes


def test_missing_tool_call_lowers_recall():
    evaluator = ToolCorrectnessEvaluator(threshold=0.8)
    calls = [ToolCall("get_logs")]  # missing create_ticket
    result = evaluator.evaluate("case-1", ["get_logs", "create_ticket"], calls)
    assert result.recall < 1.0
    assert "create_ticket" in result.notes


def test_wrong_tools_fails():
    evaluator = ToolCorrectnessEvaluator(threshold=0.8)
    calls = [ToolCall("delete_deployment"), ToolCall("rollback")]
    result = evaluator.evaluate("case-1", ["get_logs", "create_ticket"], calls)
    assert not result.passed
    assert result.f1 == pytest.approx(0.0)


def test_empty_expected_tools():
    evaluator = ToolCorrectnessEvaluator(threshold=0.8)
    result = evaluator.evaluate("case-1", [], [])
    assert result.passed


# ── Plan Adherence ────────────────────────────────────────────────────────────

def test_perfect_plan_adherence():
    evaluator = PlanAdherenceEvaluator(threshold=0.8)
    steps = ["route", "gather_context", "diagnose", "remediate"]
    result = evaluator.evaluate("case-1", steps, steps)
    assert result.passed
    assert result.adherence_score == pytest.approx(1.0)
    assert result.missing_steps == []


def test_missing_step_lowers_score():
    evaluator = PlanAdherenceEvaluator(threshold=0.8)
    expected = ["route", "gather_context", "diagnose", "remediate"]
    actual = ["route", "gather_context", "remediate"]  # skipped diagnose
    result = evaluator.evaluate("case-1", expected, actual)
    assert "diagnose" in result.missing_steps
    assert result.adherence_score < 1.0


def test_all_steps_missing_fails():
    evaluator = PlanAdherenceEvaluator(threshold=0.8)
    result = evaluator.evaluate("case-1", ["route", "gather", "diagnose"], [])
    assert not result.passed
    assert result.adherence_score == pytest.approx(0.0)


def test_extra_steps_do_not_penalize():
    evaluator = PlanAdherenceEvaluator(threshold=0.8)
    expected = ["route", "diagnose"]
    actual = ["route", "gather_context", "diagnose", "extra_step"]
    result = evaluator.evaluate("case-1", expected, actual)
    assert result.missing_steps == []
    assert result.passed


def test_empty_expected_passes():
    evaluator = PlanAdherenceEvaluator(threshold=0.8)
    result = evaluator.evaluate("case-1", [], ["some_step"])
    assert result.passed
    assert result.adherence_score == pytest.approx(1.0)
