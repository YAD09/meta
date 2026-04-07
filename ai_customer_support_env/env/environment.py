"""
CustomerSupportEnv — OpenEnv-compliant RL environment.

OpenEnv API:
  reset(task_id, scenario_index, seed)  → Observation
  step(action)                          → (Observation, Reward, bool, dict)
  state()                               → dict

Reward structure (per step):
  +1.0   Correct resolution action (ticket fully resolved as intended)
  +0.5   Significant progress (right action direction, good content)
  +0.2   Minor progress (partial step toward resolution)
  -0.2   Wrong action for the current situation
  -0.5   Looping / useless repeated action (detected via action history)
"""

from __future__ import annotations

import random
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from env.graders import grade
from env.models import Action, ActionType, HistoryEntry, Observation, Reward
from env.scenarios import get_scenario, SCENARIO_REGISTRY
from env.simulator import simulate_customer_response
from env.tasks import get_task, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Internal step-level reward engine
# ---------------------------------------------------------------------------

class _RewardEngine:
    """
    Calculates a single-step reward given the action taken and episode context.

    Reward tiers
    ────────────
    +1.0  Correct terminal resolution  — closes the episode correctly
    +0.5  Strong progress              — right action class, good content
    +0.2  Weak progress                — moves toward resolution
    -0.2  Wrong action                 — e.g. escalating a simple billing issue
    -0.5  Loop / useless repetition    — same ineffective action repeated ≥ 2×
    """

    # One-time rewards: awarded (or penalised) the first time they apply.
    # Subsequent steps only get efficiency rewards.

    def calculate(
        self,
        action: Action,
        scenario: Dict[str, Any],
        action_history: List[Action],
        step_number: int,
        grader_reward: Reward,           # from the dense grader (0-1 scale)
    ) -> Reward:
        """Return a single-step Reward combining explicit tiers + grader signal."""

        action_type: str = str(action.action_type)
        content: str = (action.content or "").lower()
        issue_type: str = scenario.get("issue_type", "")
        sentiment: str = scenario.get("sentiment", "neutral")
        mfa_enabled: bool = scenario.get("mfa_enabled", False)
        expected_actions: List[str] = scenario.get("expected_actions", [])
        penalty_actions: List[str]  = scenario.get("penalty_actions", [])

        components: Dict[str, float] = {}
        feedback: List[str] = []

        # ── 1. Loop / repetition penalty (-0.5) ──────────────────────────
        loop_penalty = self._loop_penalty(action_type, action_history)
        components["loop_penalty"] = loop_penalty
        if loop_penalty < 0:
            feedback.append(f"⚠ Loop detected: repeated '{action_type}' with no progress (-0.5)")

        # ── 2. Wrong-action penalty (-0.2) ───────────────────────────────
        wrong_penalty = 0.0
        if action_type in penalty_actions:
            wrong_penalty = -0.2
            feedback.append(f"✗ Wrong action '{action_type}' for this issue type (-0.2)")
        components["wrong_action"] = wrong_penalty

        # ── 3. Progress / resolution reward (+0.2 / +0.5 / +1.0) ────────
        progress, is_terminal = self._progress_reward(
            action_type=action_type,
            content=content,
            expected_actions=expected_actions,
            step_number=step_number,
            grader_reward=grader_reward,
            issue_type=issue_type,
            mfa_enabled=mfa_enabled,
            sentiment=sentiment,
        )
        components["progress"] = progress
        if is_terminal:
            feedback.append(f"✅ Correct resolution! (+1.0)")
        elif progress >= 0.5:
            feedback.append(f"✓ Strong progress toward resolution (+0.5)")
        elif progress >= 0.2:
            feedback.append(f"~ Minor progress (+0.2)")

        # ── 4. Grader ensemble signal (0.0–0.1 bonus) ────────────────────
        # Small bonus that blends the holistic grader score in.
        grader_bonus = round(grader_reward.score * 0.10, 3)
        components["grader_bonus"] = grader_bonus

        # ── 5. Aggregate ─────────────────────────────────────────────────
        raw = loop_penalty + wrong_penalty + progress + grader_bonus
        score = round(max(0.0, min(1.0, raw)), 4)

        return Reward(
            score=score,
            components=components,
            feedback="; ".join(feedback) or "No signal this step",
            done=grader_reward.done,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loop_penalty(action_type: str, history: List[Action]) -> float:
        """Return -0.5 if this action_type has been used ≥ 2 times already."""
        past = Counter(str(a.action_type) for a in history)
        return -0.5 if past[action_type] >= 2 else 0.0

    @staticmethod
    def _progress_reward(
        *,
        action_type: str,
        content: str,
        expected_actions: List[str],
        step_number: int,
        grader_reward: Reward,
        issue_type: str,
        mfa_enabled: bool,
        sentiment: str,
    ) -> Tuple[float, bool]:
        """
        Returns (reward_value, is_terminal).

        +1.0  Terminal: action is the correct final action for this scenario
        +0.5  Strong: action matches expected and content has substance (≥20 chars)
        +0.2  Weak:   action is in expected list but content is thin
        0.0   Neutral: non-expected action that isn't a penalty either
        """
        # Terminal resolution: grader says done AND score is ≥ 0.6
        if grader_reward.done and grader_reward.score >= 0.6:
            return 1.0, True

        # Check whether this action is on the expected path
        if action_type in expected_actions:
            if action_type in ("reply",) and len(content) >= 20:
                return 0.5, False   # substantive reply = strong progress
            if action_type in ("refund", "escalate", "close"):
                return 0.5, False   # non-reply decisive actions = strong
            return 0.2, False       # terse / short reply = minor progress

        # Partial grader signal even for unexpected actions
        if grader_reward.score >= 0.3:
            return 0.2, False

        return 0.0, False


# ---------------------------------------------------------------------------
# CustomerSupportEnv
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CustomerSupportEnv:
    """
    OpenEnv-compliant environment that simulates a SaaS customer support system.

    The agent observes a support ticket and chooses one of four actions:
    reply, refund, escalate, or close. Each step returns a structured reward
    that combines explicit tier rewards with a holistic grader signal.

    Reward tiers (per step)
    ────────────────────────
    +1.0   Correct resolution (episode ends well)
    +0.5   Strong progress   (right action, substantive content)
    +0.2   Minor progress    (action on expected path, thin content)
    -0.2   Wrong action      (penalised by scenario)
    -0.5   Loop / repetition (same action ≥ 2 previous times)

    Usage::

        env = CustomerSupportEnv()
        obs = env.reset("task_easy", seed=42)

        action = Action(action_type="refund",
                        content="Your refund has been processed.")
        obs, reward, done, info = env.step(action)

        print(reward.score, reward.feedback)
        print(env.state())
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._scenario: Optional[Dict[str, Any]] = None
        self._history: List[HistoryEntry] = []
        self._actions: List[Action] = []
        self._step_number: int = 0
        self._done: bool = False
        self._ticket_id: str = ""
        self._status: str = "open"
        self._cumulative_score: float = 0.0
        self._seed: Optional[int] = None
        self._rng: random.Random = random.Random()
        self._reward_engine = _RewardEngine()

    # ------------------------------------------------------------------
    # OpenEnv API — reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "task_easy",
        scenario_index: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            task_id:        "task_easy" | "task_medium" | "task_hard"
            scenario_index: Which scenario variant to load. If None, a random
                            one is chosen (optionally seeded by `seed`).
            seed:           Fix the RNG for reproducible random selection.

        Returns:
            Initial Observation the agent observes.
        """
        get_task(task_id)   # validate early; raises ValueError on bad id

        # Initialise RNG
        self._seed = seed
        self._rng = random.Random(seed)

        # Pick scenario
        available = SCENARIO_REGISTRY.get(task_id, [])
        if not available:
            raise ValueError(f"No scenarios registered for task_id={task_id!r}")

        if scenario_index is None:
            scenario_index = self._rng.randrange(len(available))
        else:
            scenario_index = scenario_index % len(available)

        scenario = get_scenario(task_id, scenario_index)

        # Reset all internal state
        self._task_id          = task_id
        self._scenario         = scenario
        self._history          = []
        self._actions          = []
        self._step_number      = 0
        self._done             = False
        self._ticket_id        = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        self._status           = "open"
        self._cumulative_score = 0.0

        # Seed conversation with the customer's opening message
        opening = HistoryEntry(
            role="customer",
            content=scenario["initial_message"],
            timestamp=_now_iso(),
        )
        self._history.append(opening)

        return self._build_observation(user_message=scenario["initial_message"])

    # ------------------------------------------------------------------
    # OpenEnv API — step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Reward tiers applied by _RewardEngine:
          +1.0   Correct terminal resolution
          +0.5   Strong progress
          +0.2   Minor progress
          -0.2   Wrong action (penalised by scenario spec)
          -0.5   Loop / repetition (same action type ≥ 2 previous times)

        Returns:
            (Observation, Reward, done, info)

        Raises:
            RuntimeError: if the episode is already done or `reset()` not called.
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )
        if self._task_id is None:
            raise RuntimeError(
                "Environment has not been reset. Call reset() first."
            )

        action_type = str(action.action_type)

        # Record agent action in conversation history
        self._history.append(
            HistoryEntry(
                role="agent",
                content=action.content or f"[{action_type}]",
                timestamp=_now_iso(),
            )
        )

        # Get customer's simulated response
        customer_reply = simulate_customer_response(action, self._scenario)  # type: ignore[arg-type]

        # Update ticket status
        self._status = self._derive_status(action_type)

        # ── Dense grader (episode-level holistic score) ──────────────────
        assert self._scenario is not None
        grader_reward = grade(
            task_id=self._task_id,
            actions=self._actions + [action],   # include current action
            scenario=self._scenario,
            step_number=self._step_number + 1,
        )

        # ── Step-level reward engine (explicit tier logic) ────────────────
        reward = self._reward_engine.calculate(
            action=action,
            scenario=self._scenario,
            action_history=self._actions,        # history BEFORE this action
            step_number=self._step_number,
            grader_reward=grader_reward,
        )

        # Commit action to history AFTER reward calculation
        self._actions.append(action)
        self._step_number += 1

        # Update aggregate state
        self._done = reward.done
        self._cumulative_score = round(self._cumulative_score + reward.score, 4)

        # Append customer reply to conversation history
        next_message = ""
        if customer_reply:
            self._history.append(
                HistoryEntry(
                    role="customer",
                    content=customer_reply,
                    timestamp=_now_iso(),
                )
            )
            next_message = customer_reply

        next_obs = self._build_observation(user_message=next_message)

        task = get_task(self._task_id)
        info: Dict[str, Any] = {
            "step_number":        self._step_number,
            "cumulative_score":   self._cumulative_score,
            "task_id":            self._task_id,
            "scenario_id":        self._scenario.get("scenario_id", ""),
            "max_steps":          task.max_steps,
            "success_threshold":  task.success_threshold,
            "solved": (
                self._done
                and self._cumulative_score >= task.success_threshold
            ),
            "reward_components":  reward.components,
        }

        return next_obs, reward, self._done, info

    # ------------------------------------------------------------------
    # OpenEnv API — state()
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """
        Return a complete snapshot of the current environment state.

        Useful for serialisation, logging, and Hugging Face Spaces display.
        Internal/private scenario keys (expected_actions, grading hints) are
        excluded to prevent reward-hacking.
        """
        return {
            "ticket_id":        self._ticket_id,
            "task_id":          self._task_id,
            "scenario_id":      (self._scenario or {}).get("scenario_id", ""),
            "status":           self._status,
            "step_number":      self._step_number,
            "done":             self._done,
            "cumulative_score": self._cumulative_score,
            "seed":             self._seed,
            "history":          [h.model_dump() for h in self._history],
            "actions_taken":    [a.model_dump() for a in self._actions],
            "scenario_metadata": {
                k: v
                for k, v in (self._scenario or {}).items()
                if k not in (
                    "expected_actions",
                    "penalty_actions",
                    "resolution_keywords",
                    "required_troubleshooting_steps",
                )
            },
        }

    # ------------------------------------------------------------------
    # Class-level helpers (stateless)
    # ------------------------------------------------------------------

    @staticmethod
    def available_tasks() -> Dict[str, Any]:
        """Return all registered tasks and their full specs."""
        return {tid: spec.model_dump() for tid, spec in TASK_REGISTRY.items()}

    @staticmethod
    def available_actions() -> List[str]:
        """Return all valid action type strings."""
        return [a.value for a in ActionType]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self, user_message: str) -> Observation:
        scenario = self._scenario or {}
        return Observation(
            ticket_id=self._ticket_id,
            user_message=user_message,
            user_type=scenario.get("user_type", "standard"),
            history=list(self._history),
            status=self._status,
            metadata={
                "account_age_days":  scenario.get("account_age_days", 0),
                "plan":              scenario.get("plan", "unknown"),
                "previous_refunds":  scenario.get("previous_refunds", 0),
                "charge_amount_usd": scenario.get("charge_amount_usd"),
                "issue_type":        scenario.get("issue_type", ""),
                "mfa_enabled":       scenario.get("mfa_enabled", False),
            },
            step_number=self._step_number,
            task_id=self._task_id or "",
        )

    @staticmethod
    def _derive_status(action_type: str) -> str:
        mapping = {
            "escalate": "escalated",
            "close":    "closed",
            "refund":   "pending",
            "reply":    "open",
        }
        return mapping.get(action_type, "open")
