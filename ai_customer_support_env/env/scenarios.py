"""
Realistic customer scenarios for the three OpenEnv tasks.

Task→Scenario mapping:
  task_easy   → password reset requests (3 variants)
  task_medium → valid refund requests   (3 variants)
  task_hard   → angry premium / unclear issue (3 variants)
"""

from __future__ import annotations

from typing import Any, Dict, List

Scenario = Dict[str, Any]


# ---------------------------------------------------------------------------
# TASK-1  (easy) — Password Reset Request
# ---------------------------------------------------------------------------

TASK1_SCENARIOS: List[Scenario] = [
    {
        "scenario_id": "t1_s1",
        "customer_name": "Alice",
        "user_type": "standard",
        "initial_message": (
            "Hi, I forgot my password and can't log in to my account. "
            "Can you help me reset it?"
        ),
        "account_age_days": 120,
        "plan": "standard_monthly",
        "issue_type": "password_reset",
        "mfa_enabled": False,
        # Agent should: reply with reset instructions → close
        "expected_actions": ["reply", "close"],
        "resolution_keywords": ["reset", "password", "link", "email", "click", "browser", "expire"],
        "penalty_actions": ["refund", "escalate"],
    },
    {
        "scenario_id": "t1_s2",
        "customer_name": "Ben",
        "user_type": "standard",
        "initial_message": (
            "I tried to reset my password but the link in the email doesn't work. "
            "It just says the link has expired. What should I do?"
        ),
        "account_age_days": 60,
        "plan": "standard_annual",
        "issue_type": "expired_reset_link",
        "mfa_enabled": False,
        "expected_actions": ["reply", "close"],
        "resolution_keywords": ["reset", "new link", "email", "expire", "resend", "click", "browser"],
        "penalty_actions": ["refund", "escalate"],
    },
    {
        "scenario_id": "t1_s3",
        "customer_name": "Clara",
        "user_type": "trial",
        "initial_message": (
            "I just created my account but I can't log in — I think I forgot the password "
            "I set during sign-up. How do I get back in?"
        ),
        "account_age_days": 1,
        "plan": "trial",
        "issue_type": "password_forgotten_new_account",
        "mfa_enabled": False,
        "expected_actions": ["reply", "close"],
        "resolution_keywords": ["reset", "password", "link", "email", "click", "inbox", "expire"],
        "penalty_actions": ["refund", "escalate"],
    },
]


# ---------------------------------------------------------------------------
# TASK-2  (medium) — Valid Refund Request
# ---------------------------------------------------------------------------

TASK2_SCENARIOS: List[Scenario] = [
    {
        "scenario_id": "t2_s1",
        "customer_name": "David",
        "user_type": "standard",
        "initial_message": (
            "Hello, I was charged twice for my subscription this month. "
            "I have the bank statement to prove it. Can I get a refund for the duplicate?"
        ),
        "account_age_days": 180,
        "plan": "standard_monthly",
        "charge_amount_usd": 29.00,
        "issue_type": "duplicate_charge",
        "previous_refunds": 0,
        # Must: refund + polite reply
        "expected_actions": ["reply", "refund", "close"],
        "resolution_keywords": [
            "sorry", "apologize", "refund", "processed", "days", "business", "understand"
        ],
        "penalty_actions": ["escalate"],
    },
    {
        "scenario_id": "t2_s2",
        "customer_name": "Emma",
        "user_type": "premium",
        "initial_message": (
            "I cancelled my premium plan last week but you've still charged me for this month. "
            "I need a refund immediately — this is your billing system's fault, not mine."
        ),
        "account_age_days": 365,
        "plan": "premium_monthly",
        "charge_amount_usd": 99.00,
        "issue_type": "post_cancellation_charge",
        "previous_refunds": 0,
        "expected_actions": ["reply", "refund", "close"],
        "resolution_keywords": [
            "sorry", "apologize", "refund", "processed", "days", "cancel", "understand"
        ],
        "penalty_actions": ["escalate"],
    },
    {
        "scenario_id": "t2_s3",
        "customer_name": "Frank",
        "user_type": "trial",
        "initial_message": (
            "I signed up for the free trial but got charged $49. "
            "I never agreed to any paid plan. Please refund me."
        ),
        "account_age_days": 7,
        "plan": "trial",
        "charge_amount_usd": 49.00,
        "issue_type": "erroneous_trial_charge",
        "previous_refunds": 0,
        "expected_actions": ["reply", "refund", "close"],
        "resolution_keywords": [
            "sorry", "apologize", "refund", "processed", "trial", "error", "understand"
        ],
        "penalty_actions": ["escalate"],
    },
]


# ---------------------------------------------------------------------------
# TASK-3  (hard) — Angry Premium User, Unclear Issue
# ---------------------------------------------------------------------------

TASK3_SCENARIOS: List[Scenario] = [
    {
        "scenario_id": "t3_s1",
        "customer_name": "Grace",
        "user_type": "premium",
        "initial_message": (
            "This is absolutely ridiculous! Your product completely stopped working "
            "and I'm losing money because of it. I've been a premium customer for 2 YEARS "
            "and this is how you treat me?! Fix this NOW!"
        ),
        "account_age_days": 730,
        "plan": "premium_annual",
        "issue_type": "unspecified_premium_complaint",
        "sentiment": "furious",
        "previous_refunds": 0,
        # Valid: ask what "stopped working" means, OR escalate after empathy
        "expected_actions": ["reply", "escalate"],
        "resolution_keywords": [
            "sorry", "apologize", "understand", "frustration", "premium",
            "priority", "urgent", "value", "escalate", "specialist"
        ],
        # Closing without understanding = penalty
        # Refunding without knowing the issue = penalty
        "penalty_actions": ["close", "refund"],
        "clarify_signals": ["what", "which", "can you describe", "tell me", "?"],
    },
    {
        "scenario_id": "t3_s2",
        "customer_name": "Henry",
        "user_type": "premium",
        "initial_message": (
            "Your platform is a disaster! Nothing works, everything is broken, "
            "and I can't do my job. I pay premium for THIS? "
            "I need someone competent to help me right now!"
        ),
        "account_age_days": 500,
        "plan": "premium_annual",
        "issue_type": "vague_platform_complaint",
        "sentiment": "angry",
        "previous_refunds": 1,
        "expected_actions": ["reply", "escalate"],
        "resolution_keywords": [
            "sorry", "apologize", "understand", "frustration", "premium",
            "priority", "escalate", "specialist", "senior", "team"
        ],
        "penalty_actions": ["close", "refund"],
        "clarify_signals": ["what", "which feature", "describe", "tell me more", "?"],
    },
    {
        "scenario_id": "t3_s3",
        "customer_name": "Iris",
        "user_type": "premium",
        "initial_message": (
            "I am done with this company. You keep making changes that break my workflow "
            "and nobody ever listens. I need this escalated to someone who actually cares. "
            "Last chance before I cancel everything."
        ),
        "account_age_days": 900,
        "plan": "premium_annual",
        "issue_type": "escalation_demand_with_vague_issue",
        "sentiment": "threatening",
        "previous_refunds": 0,
        "expected_actions": ["reply", "escalate"],
        "resolution_keywords": [
            "sorry", "apologize", "understand", "value", "priority",
            "escalate", "specialist", "senior", "manager", "urgent"
        ],
        "penalty_actions": ["close", "refund"],
        "clarify_signals": ["what changes", "which workflow", "can you describe", "?"],
    },
]


# ---------------------------------------------------------------------------
# Registry & helpers
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: Dict[str, List[Scenario]] = {
    "task_easy":   TASK1_SCENARIOS,
    "task_medium": TASK2_SCENARIOS,
    "task_hard":   TASK3_SCENARIOS,
}


def get_scenario(task_id: str, scenario_index: int = 0) -> Scenario:
    """Return a scenario for the given task_id and index (wraps around)."""
    scenarios = SCENARIO_REGISTRY.get(task_id)
    if not scenarios:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid: {list(SCENARIO_REGISTRY)}"
        )
    return scenarios[scenario_index % len(scenarios)]
