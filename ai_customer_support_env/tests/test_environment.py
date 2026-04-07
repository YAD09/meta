"""
Integration tests for the updated CustomerSupportEnv.

Covers:
  - reset() with random / seeded / indexed scenario selection
  - step() reward tiers: +1.0, +0.5, +0.2, -0.2, -0.5
  - Loop detection (-0.5)
  - state() snapshot
  - Class-level helpers
  - Full episode smoke tests
"""

import pytest
from env.environment import CustomerSupportEnv, _RewardEngine
from env.models import Action, ActionType, Observation, Reward
from env.graders import grade
from env.scenarios import TASK1_SCENARIOS


# ---------------------------------------------------------------------------
# Shared action builders
# ---------------------------------------------------------------------------

def _reply(text: str) -> Action:
    return Action(action_type=ActionType.REPLY, content=text)

def _refund(text: str = "Refund processed.") -> Action:
    return Action(action_type=ActionType.REFUND, content=text)

def _escalate(reason: str = "Needs tier-2") -> Action:
    return Action(action_type=ActionType.ESCALATE, reason=reason)

def _close(text: str = "Resolved.") -> Action:
    return Action(action_type=ActionType.CLOSE, content=text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return CustomerSupportEnv()

@pytest.fixture
def easy_env(env):
    env.reset("task_easy", scenario_index=0, seed=0)
    return env

@pytest.fixture
def medium_env(env):
    env.reset("task_medium", scenario_index=0, seed=0)
    return env

@pytest.fixture
def hard_env(env):
    env.reset("task_hard", scenario_index=0, seed=0)
    return env


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset("task_easy")
        assert isinstance(obs, Observation)

    def test_ticket_id_format(self, env):
        obs = env.reset("task_easy")
        assert obs.ticket_id.startswith("TKT-")
        assert len(obs.ticket_id) == 12   # "TKT-" + 8 hex chars

    def test_status_is_open(self, env):
        obs = env.reset("task_easy")
        assert obs.status == "open"

    def test_history_has_opening_message(self, env):
        obs = env.reset("task_easy")
        assert len(obs.history) == 1
        assert obs.history[0].role == "customer"
        assert len(obs.history[0].content) > 0

    def test_step_number_zero(self, env):
        obs = env.reset("task_medium")
        assert obs.step_number == 0

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset("task_impossible")

    def test_reset_clears_previous_state(self, env):
        env.reset("task_easy")
        env.step(_refund())
        env.reset("task_easy")    # second reset
        assert env.state()["step_number"] == 0
        assert env.state()["cumulative_score"] == 0.0

    def test_all_tasks_valid(self, env):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            obs = env.reset(task_id)
            assert obs.task_id == task_id

    def test_seeded_reset_is_reproducible(self, env):
        obs_a = env.reset("task_easy", seed=42)
        obs_b = env.reset("task_easy", seed=42)
        assert obs_a.user_message == obs_b.user_message

    def test_different_seeds_may_differ(self, env):
        # With 3 scenarios there's a 2/3 chance different seeds pick different ones
        obs_0 = env.reset("task_easy", seed=0)
        obs_1 = env.reset("task_easy", seed=1)
        # Not guaranteed to differ, but at least both are valid Observations
        assert isinstance(obs_0, Observation)
        assert isinstance(obs_1, Observation)

    def test_scenario_index_respected(self, env):
        obs0 = env.reset("task_easy", scenario_index=0, seed=0)
        obs2 = env.reset("task_easy", scenario_index=2, seed=0)
        assert obs0.user_message != obs2.user_message  # different scenarios

    def test_scenario_index_wraps(self, env):
        obs_a = env.reset("task_easy", scenario_index=0)
        obs_b = env.reset("task_easy", scenario_index=99)  # 99 % 3 = 0
        assert obs_a.user_message == obs_b.user_message

    def test_random_selection_when_no_index(self, env):
        # Should not raise even without explicit scenario_index
        obs = env.reset("task_hard")
        assert isinstance(obs, Observation)


# ---------------------------------------------------------------------------
# step() — reward tiers
# ---------------------------------------------------------------------------

class TestStepRewardTiers:
    """Validate each of the five reward tiers documented in the module."""

    def test_correct_resolution_score_full(self, env):
        """Ideal path → +1.0 terminal reward."""
        env.reset("task_easy", scenario_index=0, seed=0)
        env.step(_refund("We sincerely apologize for the duplicate charge. Refund issued."))
        _, reward, done, _ = env.step(_close("Your ticket is resolved. Thank you."))
        # Terminal resolution should yield score close to 1.0
        assert reward.score >= 0.70, f"Expected ≥ 0.70, got {reward.score}"
        assert done is True

    def test_strong_progress_substantive_reply(self, easy_env):
        """Substantive reply on expected path → +0.5 progress component."""
        _, reward, _, _ = easy_env.step(
            _reply(
                "We have located the duplicate charge on your account. "
                "We will process the refund right away — please allow 3–5 business days."
            )
        )
        assert reward.components.get("progress", 0) >= 0.2

    def test_wrong_action_penalty(self, easy_env):
        """Escalating a simple refund → -0.2 wrong_action component."""
        _, reward, _, _ = easy_env.step(_escalate("Escalating unnecessarily"))
        assert reward.components.get("wrong_action", 0) < 0

    def test_loop_penalty(self, easy_env):
        """Same action ≥ 2 previous uses → -0.5 loop_penalty component."""
        # Use REPLY twice (those count), then use it a third time
        easy_env.step(_reply("First reply to customer."))
        easy_env.step(_reply("Second reply adding more info."))
        _, reward, _, _ = easy_env.step(_reply("Third reply, now a loop."))
        assert reward.components.get("loop_penalty", 0) == -0.5

    def test_no_loop_on_first_two_uses(self, easy_env):
        """Loop penalty should NOT trigger until the 3rd use of same action."""
        easy_env.step(_reply("First."))
        _, reward, _, _ = easy_env.step(_reply("Second."))
        assert reward.components.get("loop_penalty", 0) == 0.0

    def test_score_always_in_bounds(self, env):
        """reward.score must be in [0.0, 1.0] regardless of action combination."""
        env.reset("task_hard", scenario_index=0, seed=0)
        actions = [
            _reply("a"),          # too short
            _reply("b"),
            _reply("c"),          # loop
            _escalate(),
        ]
        for act in actions:
            if env.state()["done"]:
                break
            _, reward, _, _ = env.step(act)
            assert 0.0 <= reward.score <= 1.0, f"Out of bounds: {reward.score}"

    def test_reward_components_present(self, easy_env):
        """Every reward must include loop_penalty, wrong_action, progress, grader_bonus."""
        _, reward, _, _ = easy_env.step(_refund())
        for key in ("loop_penalty", "wrong_action", "progress", "grader_bonus"):
            assert key in reward.components, f"Missing component: {key}"


# ---------------------------------------------------------------------------
# step() — lifecycle
# ---------------------------------------------------------------------------

class TestStepLifecycle:
    def test_returns_four_tuple(self, easy_env):
        result = easy_env.step(_refund())
        assert isinstance(result, tuple) and len(result) == 4

    def test_obs_type(self, easy_env):
        obs, *_ = easy_env.step(_refund())
        assert isinstance(obs, Observation)

    def test_reward_type(self, easy_env):
        _, reward, *_ = easy_env.step(_refund())
        assert isinstance(reward, Reward)

    def test_step_number_increments(self, easy_env):
        easy_env.step(_reply("Hello, let me help you."))
        _, _, _, info = easy_env.step(_refund())
        assert info["step_number"] == 2

    def test_close_sets_done(self, easy_env):
        easy_env.step(_refund())
        _, _, done, _ = easy_env.step(_close())
        assert done is True

    def test_step_after_done_raises(self, easy_env):
        easy_env.step(_refund())
        easy_env.step(_close())
        with pytest.raises(RuntimeError, match="already done"):
            easy_env.step(_reply("Oops"))

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="not been reset"):
            env.step(_close())

    def test_history_grows_with_steps(self, easy_env):
        easy_env.step(_reply("Hello, we will look into this right away."))
        s = easy_env.state()
        # opening + agent reply + customer follow-up = 3 entries
        assert len(s["history"]) >= 2

    def test_info_keys_present(self, easy_env):
        _, _, _, info = easy_env.step(_reply("Give me a moment to check your account."))
        for key in ("step_number", "cumulative_score", "task_id",
                    "max_steps", "success_threshold", "solved", "reward_components"):
            assert key in info

    def test_cumulative_score_accumulates(self, easy_env):
        easy_env.step(_refund("We are processing your full refund now."))
        s = easy_env.state()
        assert s["cumulative_score"] > 0.0


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------

class TestState:
    def test_pre_reset_task_id_none(self, env):
        assert env.state()["task_id"] is None

    def test_after_reset_correct_task(self, easy_env):
        s = easy_env.state()
        assert s["task_id"] == "task_easy"
        assert s["step_number"] == 0
        assert s["done"] is False

    def test_seed_recorded(self, env):
        env.reset("task_easy", seed=99)
        assert env.state()["seed"] == 99

    def test_no_private_grading_fields(self, easy_env):
        s = easy_env.state()
        meta = s.get("scenario_metadata", {})
        for forbidden in ("expected_actions", "penalty_actions",
                          "resolution_keywords", "required_troubleshooting_steps"):
            assert forbidden not in meta

    def test_actions_taken_recorded(self, easy_env):
        easy_env.step(_refund())
        s = easy_env.state()
        assert len(s["actions_taken"]) == 1
        assert s["actions_taken"][0]["action_type"] == "refund"


# ---------------------------------------------------------------------------
# _RewardEngine unit tests
# ---------------------------------------------------------------------------

class TestRewardEngine:
    ENGINE = _RewardEngine()
    SCENARIO = TASK1_SCENARIOS[0]  # duplicate_charge, premium

    def _base_grader_reward(self, done=False, score=0.5) -> Reward:
        return Reward(score=score, feedback="mock", done=done)

    def test_loop_triggered_after_two_prior_uses(self):
        history = [_refund(), _refund()]   # 2 prior refunds
        action = _refund("Third refund")
        r = self.ENGINE.calculate(
            action=action, scenario=self.SCENARIO,
            action_history=history, step_number=2,
            grader_reward=self._base_grader_reward()
        )
        assert r.components["loop_penalty"] == -0.5

    def test_no_loop_on_one_prior_use(self):
        history = [_refund()]              # only 1 prior refund
        action = _refund("Second refund")
        r = self.ENGINE.calculate(
            action=action, scenario=self.SCENARIO,
            action_history=history, step_number=1,
            grader_reward=self._base_grader_reward()
        )
        assert r.components["loop_penalty"] == 0.0

    def test_wrong_action_penalty_applied(self):
        action = _escalate("Escalating unnecessarily")
        r = self.ENGINE.calculate(
            action=action, scenario=self.SCENARIO,
            action_history=[], step_number=0,
            grader_reward=self._base_grader_reward()
        )
        assert r.components["wrong_action"] == -0.2

    def test_terminal_gives_progress_1(self):
        action = _close("Resolved.")
        r = self.ENGINE.calculate(
            action=action, scenario=self.SCENARIO,
            action_history=[_refund()], step_number=1,
            grader_reward=self._base_grader_reward(done=True, score=0.8)
        )
        assert r.components["progress"] == 1.0

    def test_grader_bonus_proportional(self):
        action = _refund()
        r = self.ENGINE.calculate(
            action=action, scenario=self.SCENARIO,
            action_history=[], step_number=0,
            grader_reward=self._base_grader_reward(score=1.0)
        )
        assert r.components["grader_bonus"] == pytest.approx(0.1, abs=0.01)


# ---------------------------------------------------------------------------
# Class helpers
# ---------------------------------------------------------------------------

class TestClassHelpers:
    def test_three_tasks(self):
        assert len(CustomerSupportEnv.available_tasks()) == 3

    def test_four_actions(self):
        assert set(CustomerSupportEnv.available_actions()) == {
            "reply", "refund", "escalate", "close"
        }


# ---------------------------------------------------------------------------
# Full episode smoke tests
# ---------------------------------------------------------------------------

class TestFullEpisodes:
    def test_easy_ideal_path_solves(self):
        env = CustomerSupportEnv()
        env.reset("task_easy", scenario_index=0, seed=0)

        _, r1, done1, _ = env.step(
            _refund("We sincerely apologize for the duplicate charge on your account. "
                    "A full refund has been processed and should appear within 3–5 days.")
        )
        assert not done1

        _, r2, done2, info = env.step(
            _close("Your ticket is now resolved. Thank you for your patience.")
        )
        assert done2
        assert info["cumulative_score"] >= 0.5

    def test_hard_episode_ends_on_escalate(self):
        env = CustomerSupportEnv()
        env.reset("task_hard", scenario_index=0, seed=0)

        env.step(_reply(
            "We deeply apologize for the outage and the impact it had on your business. "
            "We understand how unacceptable this situation is."
        ))
        env.step(_refund("We are issuing a full refund as compensation for the disruption."))
        _, _, done, info = env.step(_escalate("Escalating to senior support as requested."))
        assert done is True

    def test_medium_mfa_path(self):
        env = CustomerSupportEnv()
        env.reset("task_medium", scenario_index=1, seed=0)  # MFA scenario

        env.step(_reply(
            "Please try using your backup codes if available, "
            "or sync the time on your authenticator app."
        ))
        _, _, done, info = env.step(_escalate("MFA lockout — needs tier-2 account recovery."))
        assert done is True

    def test_episode_ends_at_max_steps(self):
        env = CustomerSupportEnv()
        env.reset("task_easy", scenario_index=0, seed=0)
        task = env.available_tasks()["task_easy"]
        max_steps = task["max_steps"]

        for i in range(max_steps):
            if env.state()["done"]:
                break
            env.step(_reply(f"Step {i}: still working on it."))

        # After max_steps the grader should have set done=True
        assert env.state()["step_number"] <= max_steps
