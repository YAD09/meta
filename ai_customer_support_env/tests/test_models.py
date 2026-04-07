"""
Tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError

from env.models import (
    Action,
    ActionType,
    Difficulty,
    HistoryEntry,
    Observation,
    Reward,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TestAction:
    def test_reply_requires_content(self):
        with pytest.raises(ValidationError, match="content"):
            Action(action_type=ActionType.REPLY)  # missing content

    def test_reply_with_content_ok(self):
        a = Action(action_type=ActionType.REPLY, content="Here is how to resolve this.")
        assert a.action_type == ActionType.REPLY
        assert a.content == "Here is how to resolve this."

    def test_refund_without_content_ok(self):
        a = Action(action_type=ActionType.REFUND)
        assert a.action_type == ActionType.REFUND
        assert a.content is None

    def test_escalate_without_content_ok(self):
        a = Action(action_type=ActionType.ESCALATE)
        assert a.action_type == ActionType.ESCALATE

    def test_close_without_content_ok(self):
        a = Action(action_type=ActionType.CLOSE)
        assert a.action_type == ActionType.CLOSE

    def test_invalid_action_type_raises(self):
        with pytest.raises(ValidationError):
            Action(action_type="fly_to_moon")

    def test_reason_field_optional(self):
        a = Action(
            action_type=ActionType.ESCALATE,
            reason="Customer is very upset and needs tier-2 support",
        )
        assert a.reason is not None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_minimal_observation(self):
        obs = Observation(
            ticket_id="TKT-001",
            user_message="I need help",
            user_type="standard",
            task_id="task_easy",
        )
        assert obs.status == "open"
        assert obs.history == []
        assert obs.step_number == 0

    def test_observation_with_history(self):
        h = HistoryEntry(role="customer", content="I need a refund")
        obs = Observation(
            ticket_id="TKT-002",
            user_message="I need a refund",
            user_type="premium",
            history=[h],
            task_id="task_easy",
        )
        assert len(obs.history) == 1
        assert obs.history[0].role == "customer"

    def test_observation_metadata_flexible(self):
        obs = Observation(
            ticket_id="TKT-003",
            user_message="Test",
            user_type="trial",
            task_id="task_hard",
            metadata={"account_age_days": 30, "plan": "trial", "custom_key": True},
        )
        assert obs.metadata["custom_key"] is True


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class TestReward:
    def test_score_bounds_lower(self):
        r = Reward(score=0.0, feedback="Nothing done")
        assert r.score == 0.0

    def test_score_bounds_upper(self):
        r = Reward(score=1.0, feedback="Perfect")
        assert r.score == 1.0

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            Reward(score=1.5, feedback="Over max")

    def test_score_negative_raises(self):
        with pytest.raises(ValidationError):
            Reward(score=-0.1, feedback="Below min")

    def test_components_optional(self):
        r = Reward(score=0.5)
        assert r.components == {}

    def test_done_defaults_false(self):
        r = Reward(score=0.5)
        assert r.done is False


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

class TestTaskSpec:
    def test_task_spec_creation(self):
        t = TaskSpec(
            task_id="task_easy",
            title="Simple Refund",
            difficulty=Difficulty.EASY,
            description="Handle a basic refund",
        )
        assert t.max_steps == 10
        assert t.success_threshold == 0.7

    def test_easy_medium_hard_enum(self):
        for d in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            t = TaskSpec(
                task_id="x",
                title="X",
                difficulty=d,
                description="d",
            )
            assert t.difficulty == d
