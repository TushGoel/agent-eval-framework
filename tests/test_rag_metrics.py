"""Tests for RAG evaluation metrics."""

import pytest
from src.rag_metrics import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    ContextualRecallEvaluator,
)


# ── Answer Relevancy ──────────────────────────────────────────────────────────

def test_relevant_answer_passes():
    evaluator = AnswerRelevancyEvaluator(threshold=0.3)
    result = evaluator.evaluate(
        question="What caused the deployment failure?",
        answer="The deployment failure was caused by a database timeout.",
    )
    assert result.passed
    assert result.score > 0.3


def test_irrelevant_answer_fails():
    evaluator = AnswerRelevancyEvaluator(threshold=0.3)
    result = evaluator.evaluate(
        question="What caused the deployment failure?",
        answer="The weather in Seattle is often rainy in autumn.",
    )
    assert not result.passed


def test_exact_answer_high_score():
    evaluator = AnswerRelevancyEvaluator(threshold=0.5)
    result = evaluator.evaluate(
        question="deployment failure database timeout",
        answer="deployment failure database timeout error in production",
    )
    assert result.score > 0.7


# ── Faithfulness ──────────────────────────────────────────────────────────────

def test_faithful_answer_passes():
    evaluator = FaithfulnessEvaluator(threshold=0.5)
    context = "Deployment d-123 failed at step 4. Logs show timeout errors on port 5432."
    answer = "Deployment d-123 failed with timeout errors."
    result = evaluator.evaluate(answer=answer, context=context)
    assert result.passed
    assert result.hallucinated_fraction < 0.5


def test_hallucinated_answer_fails():
    evaluator = FaithfulnessEvaluator(threshold=0.5)
    context = "Deployment d-123 failed at step 4."
    answer = "The system was hacked by external attackers using zero-day exploits."
    result = evaluator.evaluate(answer=answer, context=context)
    assert not result.passed
    assert result.hallucinated_fraction > 0.5


def test_faithfulness_score_range():
    evaluator = FaithfulnessEvaluator()
    result = evaluator.evaluate("some answer text", "some context text")
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.hallucinated_fraction <= 1.0


# ── Contextual Recall ─────────────────────────────────────────────────────────

def test_relevant_docs_pass():
    # Use low relevance threshold — TF-IDF word overlap, not semantic embedding
    evaluator = ContextualRecallEvaluator(relevance_threshold=0.1, recall_threshold=0.3)
    result = evaluator.evaluate(
        question="deployment failure",
        retrieved_docs=[
            "deployment failure database timeout",
            "deployment failure connection pool",
            "SPICE refresh success",
        ],
    )
    assert result.passed
    assert result.relevant_doc_count >= 1


def test_no_docs_fails():
    evaluator = ContextualRecallEvaluator()
    result = evaluator.evaluate("question", [])
    assert not result.passed
    assert result.recall_score == 0.0


def test_irrelevant_docs_fail():
    evaluator = ContextualRecallEvaluator(relevance_threshold=0.5, recall_threshold=0.5)
    result = evaluator.evaluate(
        question="What caused the deployment failure?",
        retrieved_docs=[
            "The weather today is sunny",
            "Python version 3.11 released",
        ],
    )
    assert not result.passed


def test_recall_score_bounds():
    evaluator = ContextualRecallEvaluator()
    result = evaluator.evaluate("question text", ["doc one", "doc two"])
    assert 0.0 <= result.recall_score <= 1.0
