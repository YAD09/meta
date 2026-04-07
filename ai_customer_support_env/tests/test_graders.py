"""
Unit tests for the three graders: grade_easy, grade_medium, grade_hard.

Each grader is tested for:
  1. Correctness dimension
  2. Action selection dimension
  3. Response quality / keyword coverage
  4. Efficiency
  5. Score bounds [0.0, 1.0]
  6. Determinism (same input → same output)
"""

import pytest
from env.models import Action, ActionType, Reward
from env.graders import grade_easy, grade_medium, grade_hard, grade
from env.scenarios import TASK1_SCENARIOS, TASK2_SCENARIOS, TASK3_SCENARIOS


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def reply(text: str) -> Action:
    return Action(action_type=ActionType.REPLY, content=text)

def refund(text: str = "Refund processed.") -> Action:
    return Action(action_type=ActionType.REFUND, content=text)

def escalate(reason: str = "Needs specialist") -> Action:
    return Action(action_type=ActionType.ESCALATE, reason=reason)

def close(text: str = "Resolved.") -> Action:
    return Action(action_type=ActionType.CLOSE, content=text)

# Ideal reply for password reset
_PWD_REPLY = (
    "To reset your password, click the 'Forgot Password' link on the login page. "
    "We'll email you a reset link — click it within 30 minutes before it expires. "
    "If it doesn't arrive, check spam or try a different browser."
)

# Ideal reply for refund
_REFUND_REPLY = (
    "We sincerely apologize for the inconvenience. We certainly understand your frustration. "
    "Your refund has been processed and will appear within 5–7 business days."
)

# Ideal reply for angry/unclear
_EMPATHY_REPLY = (
    "We sincerely apologize for the frustration you're experiencing. "
    "As a premium customer, you are our top priority. "
    "Could you describe exactly what is not working so we can help you urgently? "
    "Alternatively, I can escalate this to a senior specialist right now."
)


# ---------------------------------------------------------------------------
# TASK-1  (easy) — Password Reset
# ---------------------------------------------------------------------------

class TestGradeEasy:
    S = TASK1_SCENARIOS[0]

    def test_ideal_path_high_score(self):
        actions = [reply(_PWD_REPLY), close()]
        r = grade_easy(actions, self.S, step_number=2)
        assert r.score >= 0.70, f"Ideal path expected ≥ 0.70, got {r.score}"
        assert r.done is True

    def test_no_actions_zero_score(self):
        r = grade_easy([], self.S, step_number=0)
        assert r.score == 0.0

    def test_reply_without_keywords_partial_score(self):
        r = grade_easy([reply("Sure, I can help you with that.")], self.S, step_number=1)
        # correctness present (replied), but quality low
        assert r.components["correctness"] == 0.30
        assert r.components["response_quality"] < 0.10

    def test_all_keywords_max_quality(self):
        r = grade_easy([reply(_PWD_REPLY)], self.S, step_number=1)
        assert r.components["response_quality"] >= 0.25

    def test_wrong_action_escalate_penalises_action_sel(self):
        r = grade_easy([escalate()], self.S, step_number=1)
        assert r.components["action_selection"] < 0.25

    def test_close_without_reply_no_correctness(self):
        r = grade_easy([close()], self.S, step_number=1)
        assert r.components["correctness"] == 0.0

    def test_done_on_close(self):
        r = grade_easy([reply(_PWD_REPLY), close()], self.S, step_number=2)
        assert r.done is True

    def test_done_at_max_steps(self):
        r = grade_easy([reply("OK")], self.S, step_number=5)
        assert r.done is True

    def test_deterministic(self):
        actions = [reply(_PWD_REPLY), close()]
        r1 = grade_easy(actions, self.S, step_number=2)
        r2 = grade_easy(actions, self.S, step_number=2)
        assert r1.score == r2.score

    def test_score_in_bounds_all_scenarios(self):
        for s in TASK1_SCENARIOS:
            r = grade_easy([reply(_PWD_REPLY), close()], s, step_number=2)
            assert 0.0 <= r.score <= 1.0

    def test_efficiency_penalised_at_max_steps(self):
        r = grade_easy([reply(_PWD_REPLY)], self.S, step_number=5)
        assert r.components["efficiency"] == 0.0

    def test_components_keys_present(self):
        r = grade_easy([reply("Help.")], self.S, step_number=1)
        for k in ("correctness", "action_selection", "response_quality", "efficiency"):
            assert k in r.components, f"Missing: {k}"


# ---------------------------------------------------------------------------
# TASK-2  (medium) — Valid Refund Request
# ---------------------------------------------------------------------------

class TestGradeMedium:
    S = TASK2_SCENARIOS[0]   # duplicate_charge

    def test_ideal_path_high_score(self):
        actions = [reply(_REFUND_REPLY), refund(), close()]
        r = grade_medium(actions, self.S, step_number=3)
        assert r.score >= 0.70, f"Expected ≥ 0.70, got {r.score}"
        assert r.done is True

    def test_refund_without_reply_partial(self):
        r = grade_medium([refund()], self.S, step_number=1)
        assert r.components["correctness"] == 0.35
        assert r.components["response_quality"] < 0.10

    def test_no_refund_zero_correctness(self):
        r = grade_medium([reply("We are looking into it.")], self.S, step_number=1)
        assert r.components["correctness"] == 0.0

    def test_polite_keywords_boost_quality(self):
        r1 = grade_medium([reply(_REFUND_REPLY), refund()], self.S, step_number=2)
        r2 = grade_medium([reply("ok."), refund()], self.S, step_number=2)
        assert r1.components["response_quality"] > r2.components["response_quality"]

    def test_escalate_over_cautious_penalty(self):
        r = grade_medium([refund(), escalate()], self.S, step_number=2)
        assert r.components["action_selection"] < 0.20

    def test_done_on_close(self):
        r = grade_medium([reply(_REFUND_REPLY), refund(), close()], self.S, step_number=3)
        assert r.done is True

    def test_done_on_escalate(self):
        r = grade_medium([refund(), escalate()], self.S, step_number=2)
        assert r.done is True

    def test_score_in_bounds_all_scenarios(self):
        for s in TASK2_SCENARIOS:
            r = grade_medium([reply(_REFUND_REPLY), refund(), close()], s, step_number=3)
            assert 0.0 <= r.score <= 1.0

    def test_deterministic(self):
        actions = [reply(_REFUND_REPLY), refund(), close()]
        r1 = grade_medium(actions, self.S, step_number=3)
        r2 = grade_medium(actions, self.S, step_number=3)
        assert r1.score == r2.score

    def test_sub_components_present(self):
        r = grade_medium([reply(_REFUND_REPLY), refund()], self.S, step_number=2)
        assert "_politeness_sub" in r.components
        assert "_confirmation_sub" in r.components

    def test_components_keys_present(self):
        r = grade_medium([refund()], self.S, step_number=1)
        for k in ("correctness", "action_selection", "response_quality", "efficiency"):
            assert k in r.components


# ---------------------------------------------------------------------------
# TASK-3  (hard) — Angry Premium / Unclear Issue
# ---------------------------------------------------------------------------

class TestGradeHard:
    S = TASK3_SCENARIOS[0]   # furious premium

    def test_escalate_after_empathy_high_score(self):
        actions = [reply(_EMPATHY_REPLY), escalate("Escalating to senior specialist.")]
        r = grade_hard(actions, self.S, step_number=2)
        assert r.score >= 0.65, f"Expected ≥ 0.65, got {r.score}"
        assert r.done is True

    def test_empathy_keywords_boost_score(self):
        r_empathy = grade_hard([reply(_EMPATHY_REPLY)], self.S, step_number=1)
        r_cold    = grade_hard([reply("I see. What do you want?")], self.S, step_number=1)
        assert r_empathy.components["empathy"] > r_cold.components["empathy"]

    def test_clarifying_question_counts_as_correct(self):
        r = grade_hard(
            [reply("We apologize for the frustration. Could you describe what is not working "
                   "so we can help you urgently as a premium customer?")],
            self.S, step_number=1
        )
        assert r.components["correctness"] >= 0.25

    def test_close_without_resolve_zero_action_sel(self):
        r = grade_hard([close()], self.S, step_number=1)
        assert r.components["action_selection"] == 0.0

    def test_refund_without_understanding_low_score(self):
        r = grade_hard([refund()], self.S, step_number=1)
        # Refund without clarification or escalation → penalised action
        assert r.components["action_selection"] <= 0.10

    def test_done_on_escalate(self):
        r = grade_hard([reply(_EMPATHY_REPLY), escalate()], self.S, step_number=2)
        assert r.done is True

    def test_done_at_max_steps(self):
        r = grade_hard([reply("Still working.")], self.S, step_number=8)
        assert r.done is True

    def test_score_in_bounds_all_scenarios(self):
        for s in TASK3_SCENARIOS:
            r = grade_hard([reply(_EMPATHY_REPLY), escalate()], s, step_number=2)
            assert 0.0 <= r.score <= 1.0

    def test_deterministic(self):
        actions = [reply(_EMPATHY_REPLY), escalate()]
        r1 = grade_hard(actions, self.S, step_number=2)
        r2 = grade_hard(actions, self.S, step_number=2)
        assert r1.score == r2.score

    def test_components_keys_present(self):
        r = grade_hard([reply("ok")], self.S, step_number=1)
        for k in ("correctness", "empathy", "action_selection",
                  "response_quality", "efficiency"):
            assert k in r.components


# ---------------------------------------------------------------------------
# Dispatcher grade()
# ---------------------------------------------------------------------------

class TestGradeDispatcher:
    def test_dispatches_easy(self):
        r = grade("task_easy", [reply(_PWD_REPLY), close()], TASK1_SCENARIOS[0], 2)
        assert isinstance(r, Reward)

    def test_dispatches_medium(self):
        r = grade("task_medium", [reply(_REFUND_REPLY), refund(), close()], TASK2_SCENARIOS[0], 3)
        assert isinstance(r, Reward)

    def test_dispatches_hard(self):
        r = grade("task_hard", [reply(_EMPATHY_REPLY), escalate()], TASK3_SCENARIOS[0], 2)
        assert isinstance(r, Reward)

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="No grader registered"):
            grade("task_unknown", [], TASK1_SCENARIOS[0], 0)

    def test_all_scores_in_range(self):
        combos = [
            ("task_easy",   [reply(_PWD_REPLY), close()],              TASK1_SCENARIOS[0], 2),
            ("task_medium", [reply(_REFUND_REPLY), refund(), close()], TASK2_SCENARIOS[0], 3),
            ("task_hard",   [reply(_EMPATHY_REPLY), escalate()],       TASK3_SCENARIOS[0], 2),
        ]
        for task_id, actions, scenario, step in combos:
            r = grade(task_id, actions, scenario, step)
            assert 0.0 <= r.score <= 1.0, f"{task_id}: out of bounds {r.score}"
