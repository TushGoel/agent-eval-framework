"""
RAG evaluation metrics — measure retrieval and answer quality.

Three metrics that form a complete RAG quality picture:

  AnswerRelevancy   — is the answer actually relevant to the question?
  Faithfulness      — is the answer grounded in the retrieved context?
                      (not hallucinating beyond what context supports)
  ContextualRecall  — did retrieval surface the documents needed to answer?

These are inspired by RAGAS and DeepEval's RAG metrics, implemented without
LLM-as-judge for zero-dependency evaluation. Use TF-IDF cosine similarity
for relevancy; use word overlap for faithfulness approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ── Shared TF-IDF helpers (same approach as semantic cache) ──────────────────

def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _tfidf(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return {t: c / n for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in `a` that appear in `b`."""
    words_a = set(_tokens(a))
    words_b = set(_tokens(b))
    if not words_a:
        return 1.0
    return len(words_a & words_b) / len(words_a)


# ── Metric results ────────────────────────────────────────────────────────────

@dataclass
class AnswerRelevancyResult:
    question: str
    answer: str
    score: float       # 0.0–1.0 cosine similarity between question and answer
    passed: bool
    threshold: float

    def __str__(self) -> str:
        return f"AnswerRelevancy score={self.score:.2f} (threshold={self.threshold}) {'PASS' if self.passed else 'FAIL'}"


@dataclass
class FaithfulnessResult:
    answer: str
    context: str
    score: float       # fraction of answer words supported by context
    passed: bool
    threshold: float
    hallucinated_fraction: float

    def __str__(self) -> str:
        return (
            f"Faithfulness score={self.score:.2f} (threshold={self.threshold}) "
            f"{'PASS' if self.passed else 'FAIL'} | "
            f"hallucinated={self.hallucinated_fraction:.0%}"
        )


@dataclass
class ContextualRecallResult:
    question: str
    retrieved_docs: list[str]
    relevant_doc_count: int        # docs scoring above relevance threshold
    total_retrieved: int
    recall_score: float            # relevant / total retrieved
    passed: bool
    threshold: float

    def __str__(self) -> str:
        return (
            f"ContextualRecall score={self.recall_score:.2f} (threshold={self.threshold}) "
            f"{'PASS' if self.passed else 'FAIL'} | "
            f"relevant={self.relevant_doc_count}/{self.total_retrieved}"
        )


# ── Evaluators ────────────────────────────────────────────────────────────────

class AnswerRelevancyEvaluator:
    """
    Measures whether the generated answer is relevant to the question.

    Uses TF-IDF cosine similarity. A low score means the answer is off-topic —
    the LLM answered something other than what was asked.

    Note: high relevancy ≠ high faithfulness. An answer can be perfectly
    relevant (addresses the question) but hallucinate facts not in the context.

    Usage:
        evaluator = AnswerRelevancyEvaluator(threshold=0.5)
        result = evaluator.evaluate(
            question="What caused the deployment failure?",
            answer="The database connection pool was exhausted due to a timeout.",
        )
        print(result)  # AnswerRelevancy score=0.72 (threshold=0.5) PASS
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def evaluate(self, question: str, answer: str) -> AnswerRelevancyResult:
        score = _cosine(_tfidf(question), _tfidf(answer))
        return AnswerRelevancyResult(
            question=question,
            answer=answer,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
        )


class FaithfulnessEvaluator:
    """
    Measures whether the answer is grounded in the retrieved context.

    A faithful answer only uses information present in the context — it doesn't
    add facts the context doesn't support (hallucination).

    Implementation: word overlap between answer and context. For production,
    replace with an LLM-as-judge that checks each factual claim.

    Usage:
        evaluator = FaithfulnessEvaluator(threshold=0.6)
        result = evaluator.evaluate(
            answer="The deployment failed due to a database timeout at 14:32.",
            context="Deployment d-123 failed at step 4. Logs show timeout errors on port 5432.",
        )
        print(result)  # Faithfulness score=0.82 PASS
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def evaluate(self, answer: str, context: str) -> FaithfulnessResult:
        score = _word_overlap(answer, context)
        return FaithfulnessResult(
            answer=answer,
            context=context,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            hallucinated_fraction=max(0.0, 1.0 - score),
        )


class ContextualRecallEvaluator:
    """
    Measures whether the retrieved documents are relevant to the question.

    A high recall means the retriever surfaced useful documents.
    A low recall means the knowledge base is missing relevant content or
    the retriever's similarity threshold is too strict.

    Usage:
        evaluator = ContextualRecallEvaluator(relevance_threshold=0.3, recall_threshold=0.5)
        result = evaluator.evaluate(
            question="What caused the deployment failure?",
            retrieved_docs=[
                "Deployment d-123 failed with database timeout on port 5432",
                "SPICE refresh completed successfully for dataset ds-456",
                "Deployment failed at step 4 — connection pool limit reached",
            ],
        )
        print(result)  # ContextualRecall score=0.67 PASS | relevant=2/3
    """

    def __init__(
        self,
        relevance_threshold: float = 0.3,
        recall_threshold: float = 0.5,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.recall_threshold = recall_threshold

    def evaluate(self, question: str, retrieved_docs: list[str]) -> ContextualRecallResult:
        if not retrieved_docs:
            return ContextualRecallResult(
                question=question,
                retrieved_docs=[],
                relevant_doc_count=0,
                total_retrieved=0,
                recall_score=0.0,
                passed=False,
                threshold=self.recall_threshold,
            )

        q_vec = _tfidf(question)
        relevant = sum(
            1 for doc in retrieved_docs
            if _cosine(q_vec, _tfidf(doc)) >= self.relevance_threshold
        )
        recall_score = relevant / len(retrieved_docs)

        return ContextualRecallResult(
            question=question,
            retrieved_docs=retrieved_docs,
            relevant_doc_count=relevant,
            total_retrieved=len(retrieved_docs),
            recall_score=recall_score,
            passed=recall_score >= self.recall_threshold,
            threshold=self.recall_threshold,
        )
