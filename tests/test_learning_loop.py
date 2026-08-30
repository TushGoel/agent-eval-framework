"""Tests for the accumulate → validate → promote learning loop."""

import pytest
from src.learning_loop import LearningLoop, PatternStage


def _loop(min_confirmations=3, min_confidence=0.75):
    return LearningLoop(min_confirmations=min_confirmations, min_confidence=min_confidence)


def test_new_observation_is_candidate():
    loop = _loop()
    entry = loop.observe("p1", "agent misclassifies timeouts", "classify port 5432 as db")
    assert entry.stage == PatternStage.CANDIDATE
    assert entry.observed_count == 1


def test_second_observation_increments_count():
    loop = _loop()
    loop.observe("p1", "desc", "correction")
    loop.observe("p1", "desc", "correction")
    entry = loop._get("p1")
    assert entry.observed_count == 2


def test_confirm_promotes_to_confirmed_at_threshold():
    loop = _loop(min_confirmations=3)
    loop.observe("p1", "desc", "correction")
    loop.confirm("p1")
    loop.confirm("p1")
    assert loop._get("p1").stage == PatternStage.CANDIDATE  # not yet
    loop.confirm("p1")
    assert loop._get("p1").stage == PatternStage.CONFIRMED   # now confirmed


def test_confirmed_pattern_can_be_promoted():
    loop = _loop(min_confirmations=2)
    loop.observe("p1", "desc", "correction")
    loop.confirm("p1")
    loop.confirm("p1")
    loop.promote("p1")
    assert loop._get("p1").stage == PatternStage.PROMOTED


def test_unconfirmed_pattern_cannot_be_promoted():
    loop = _loop(min_confirmations=3)
    loop.observe("p1", "desc", "correction")
    loop.confirm("p1")  # only 1, need 3
    with pytest.raises(ValueError, match="CONFIRMED"):
        loop.promote("p1")


def test_reject_removes_from_promotion_candidates():
    loop = _loop(min_confirmations=2, min_confidence=0.75)
    loop.observe("p1", "desc", "correction")
    loop.confirm("p1")
    loop.confirm("p1")
    # Now confirmed — but then evidence comes in that it's wrong
    loop.reject("p1")
    loop.reject("p1")
    loop.reject("p1")
    loop.reject("p1")
    # Confidence dropped below threshold — rejected
    assert loop._get("p1").stage == PatternStage.REJECTED


def test_rejected_pattern_cannot_be_confirmed():
    loop = _loop(min_confirmations=2, min_confidence=0.75)
    loop.observe("p1", "desc", "correction")
    loop.reject("p1")
    loop.reject("p1")
    loop.reject("p1")
    loop.reject("p1")
    with pytest.raises(ValueError, match="rejected"):
        loop.confirm("p1")


def test_ready_to_promote_returns_confirmed_only():
    loop = _loop(min_confirmations=2)
    loop.observe("p1", "desc", "fix")
    loop.observe("p2", "desc2", "fix2")
    loop.confirm("p1")
    loop.confirm("p1")  # p1 confirmed
    # p2 still candidate
    assert len(loop.ready_to_promote()) == 1
    assert loop.ready_to_promote()[0].pattern_id == "p1"


def test_export_promoted_output():
    loop = _loop(min_confirmations=1)
    loop.observe("timeout-classifier", "misclassifies db timeouts", "use port to classify")
    loop.confirm("timeout-classifier")
    loop.promote("timeout-classifier")
    output = loop.export_promoted()
    assert "timeout-classifier" in output
    assert "misclassifies db timeouts" in output
    assert "use port to classify" in output


def test_confidence_score():
    loop = _loop()
    loop.observe("p1", "d", "c")
    loop.confirm("p1")
    loop.confirm("p1")
    loop.confirm("p1")
    loop.reject("p1")
    # 3 confirm, 1 reject → 75%
    assert loop._get("p1").confidence == pytest.approx(0.75)


def test_summary_counts():
    loop = _loop(min_confirmations=1)
    loop.observe("a", "d", "c")
    loop.confirm("a")
    loop.promote("a")
    loop.observe("b", "d", "c")  # candidate
    assert "promoted=1" in loop.summary()
    assert "candidate=1" in loop.summary()


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        LearningLoop(min_confirmations=0)
    with pytest.raises(ValueError):
        LearningLoop(min_confidence=0.0)
