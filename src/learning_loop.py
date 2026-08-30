"""
Learning loop — accumulate corrections, validate, promote to classifier.

Pattern from production: raw agent corrections accumulate in a staging area.
Once a correction is confirmed across N independent evaluations, it graduates
to the classifier as a new rule. This prevents premature codification of
one-off errors as permanent patterns.

Three stages:
  1. CANDIDATE   — observed once, not yet confirmed
  2. CONFIRMED   — observed N times, ready to promote
  3. PROMOTED    — incorporated into the agent's knowledge base

The loop never promotes a pattern from a single data point. Every promoted
rule has a confirmation count >= min_confirmations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PatternStage(str, Enum):
    CANDIDATE = "candidate"     # observed once — watch and wait
    CONFIRMED = "confirmed"     # confirmed N times — ready to promote
    PROMOTED = "promoted"       # incorporated into agent knowledge base
    REJECTED = "rejected"       # contradicted by later evidence — discard


@dataclass
class LearningEntry:
    """A candidate correction or pattern observed during evaluation."""
    pattern_id: str
    description: str                # what the agent got wrong
    correction: str                 # what it should do instead
    observed_count: int = 1
    confirmation_count: int = 0
    rejection_count: int = 0
    stage: PatternStage = PatternStage.CANDIDATE
    examples: list[str] = field(default_factory=list)  # eval cases that triggered this

    def observe(self) -> None:
        self.observed_count += 1

    def confirm(self) -> None:
        self.confirmation_count += 1

    def reject(self) -> None:
        self.rejection_count += 1

    @property
    def confidence(self) -> float:
        """Confidence score: confirmations / (confirmations + rejections)."""
        total = self.confirmation_count + self.rejection_count
        if total == 0:
            return 0.0
        return self.confirmation_count / total


class LearningLoop:
    """
    Accumulate → validate → promote pipeline for agent corrections.

    The three-stage model prevents premature codification:
    - Stage 1 (CANDIDATE): pattern observed once. Don't promote yet.
    - Stage 2 (CONFIRMED): pattern confirmed >= min_confirmations times AND
      confidence >= min_confidence. Safe to promote.
    - Stage 3 (PROMOTED): incorporated into agent knowledge base (call export()).

    Usage:
        loop = LearningLoop(min_confirmations=3, min_confidence=0.8)

        # Record a correction from an eval run
        loop.observe(
            pattern_id="timeout-classification",
            description="Agent classifies database timeouts as 'unknown' instead of 'infrastructure'",
            correction="Timeouts on port 5432 → classify as infrastructure/database",
            example_case="case_id=triage-44",
        )

        # On next eval run — same pattern triggers again
        loop.confirm("timeout-classification")
        loop.confirm("timeout-classification")
        loop.confirm("timeout-classification")

        # Pattern is now CONFIRMED — promote to knowledge base
        to_promote = loop.ready_to_promote()
        for entry in to_promote:
            loop.promote(entry.pattern_id)

        # Export all promoted patterns for injection into agent context
        classifier_update = loop.export_promoted()
    """

    def __init__(
        self,
        min_confirmations: int = 3,
        min_confidence: float = 0.75,
    ) -> None:
        if min_confirmations < 1:
            raise ValueError("min_confirmations must be >= 1")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        self.min_confirmations = min_confirmations
        self.min_confidence = min_confidence
        self._entries: dict[str, LearningEntry] = {}

    def observe(
        self,
        pattern_id: str,
        description: str,
        correction: str,
        example_case: Optional[str] = None,
    ) -> LearningEntry:
        """
        Record an observed correction. If the pattern_id already exists,
        increment the observation count. Otherwise create a new CANDIDATE entry.
        """
        if pattern_id in self._entries:
            entry = self._entries[pattern_id]
            entry.observe()
            if example_case:
                entry.examples.append(example_case)
            return entry

        entry = LearningEntry(
            pattern_id=pattern_id,
            description=description,
            correction=correction,
            examples=[example_case] if example_case else [],
        )
        self._entries[pattern_id] = entry
        return entry

    def confirm(self, pattern_id: str) -> LearningEntry:
        """
        Confirm that a candidate pattern is correct (another eval run validated it).
        Automatically transitions to CONFIRMED when thresholds are met.
        """
        entry = self._get(pattern_id)
        if entry.stage == PatternStage.REJECTED:
            raise ValueError(f"Pattern {pattern_id!r} was rejected — cannot confirm.")
        entry.confirm()
        if (
            entry.stage == PatternStage.CANDIDATE
            and entry.confirmation_count >= self.min_confirmations
            and entry.confidence >= self.min_confidence
        ):
            entry.stage = PatternStage.CONFIRMED
        return entry

    def reject(self, pattern_id: str) -> LearningEntry:
        """
        Record that a candidate pattern was contradicted by a later eval.
        If rejections dominate, move to REJECTED.
        """
        entry = self._get(pattern_id)
        entry.reject()
        # Reject when a majority of evidence is against this pattern
        if entry.confidence < 0.5:
            entry.stage = PatternStage.REJECTED
        return entry

    def promote(self, pattern_id: str) -> LearningEntry:
        """
        Promote a CONFIRMED pattern to PROMOTED (incorporated into knowledge base).
        Raises if the pattern is not yet CONFIRMED.
        """
        entry = self._get(pattern_id)
        if entry.stage != PatternStage.CONFIRMED:
            raise ValueError(
                f"Pattern {pattern_id!r} is {entry.stage.value} — "
                f"must be CONFIRMED before promotion. "
                f"Current confirmations: {entry.confirmation_count}/{self.min_confirmations}"
            )
        entry.stage = PatternStage.PROMOTED
        return entry

    def ready_to_promote(self) -> list[LearningEntry]:
        """Return all entries that are CONFIRMED and ready for promotion."""
        return [e for e in self._entries.values() if e.stage == PatternStage.CONFIRMED]

    def export_promoted(self) -> str:
        """
        Export all PROMOTED patterns as a formatted string for injection
        into agent context (system prompt or knowledge base document).
        """
        promoted = [e for e in self._entries.values() if e.stage == PatternStage.PROMOTED]
        if not promoted:
            return "No promoted patterns."
        lines = ["# Learned Corrections (validated, promoted)\n"]
        for e in promoted:
            lines.append(f"## {e.pattern_id}")
            lines.append(f"**Problem:** {e.description}")
            lines.append(f"**Correction:** {e.correction}")
            lines.append(f"**Confidence:** {e.confidence:.0%} ({e.confirmation_count} confirmations)")
            if e.examples:
                lines.append(f"**Seen in:** {', '.join(e.examples[:3])}")
            lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        counts = {stage: 0 for stage in PatternStage}
        for e in self._entries.values():
            counts[e.stage] += 1
        return (
            f"LearningLoop | "
            f"candidate={counts[PatternStage.CANDIDATE]} "
            f"confirmed={counts[PatternStage.CONFIRMED]} "
            f"promoted={counts[PatternStage.PROMOTED]} "
            f"rejected={counts[PatternStage.REJECTED]}"
        )

    def _get(self, pattern_id: str) -> LearningEntry:
        if pattern_id not in self._entries:
            raise KeyError(f"Pattern {pattern_id!r} not found. Call observe() first.")
        return self._entries[pattern_id]

    def __len__(self) -> int:
        return len(self._entries)
