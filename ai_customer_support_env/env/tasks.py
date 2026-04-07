"""
Task registry for the AI Customer Support Automation — OpenEnv environment.

Tasks
─────
task_easy   Password reset request
task_medium Valid refund request
task_hard   Angry premium user with unclear issue
"""

from __future__ import annotations

from env.models import Difficulty, TaskSpec

TASK_REGISTRY: dict[str, TaskSpec] = {

    # ── Easy ──────────────────────────────────────────────────────────────
    "task_easy": TaskSpec(
        task_id="task_easy",
        title="Password Reset Request",
        difficulty=Difficulty.EASY,
        description=(
            "A customer cannot log in and needs help resetting their password. "
            "The agent must reply with clear, step-by-step reset instructions. "
            "Escalating or closing without instructions is penalised. "
            "Full marks require covering: reset link, email check, link expiry, browser tip."
        ),
        max_steps=5,
        success_threshold=0.65,
        tags=["password", "login", "instructions", "reply"],
    ),

    # ── Medium ────────────────────────────────────────────────────────────
    "task_medium": TaskSpec(
        task_id="task_medium",
        title="Valid Refund Request",
        difficulty=Difficulty.MEDIUM,
        description=(
            "A customer with a legitimate billing error requests a refund. "
            "The agent must: (1) issue the refund, and (2) reply politely with an apology "
            "and confirmation. Closing without a refund, or refunding without a polite "
            "acknowledgement, results in partial credit."
        ),
        max_steps=6,
        success_threshold=0.60,
        tags=["refund", "billing", "polite", "apology"],
    ),

    # ── Hard ──────────────────────────────────────────────────────────────
    "task_hard": TaskSpec(
        task_id="task_hard",
        title="Angry Premium User — Unclear Issue",
        difficulty=Difficulty.HARD,
        description=(
            "A premium customer is angry but their actual problem is ambiguous. "
            "The agent must either: (a) ask a clear clarifying question to identify the "
            "issue, OR (b) escalate to a specialist — but only after acknowledging the "
            "customer's frustration. Closing without clarification or escalation, "
            "or issuing a refund without understanding the issue, is penalised."
        ),
        max_steps=8,
        success_threshold=0.55,
        tags=["angry", "premium", "escalate", "clarify", "empathy"],
    ),
}


def get_task(task_id: str) -> TaskSpec:
    """Retrieve a TaskSpec by id; raises ValueError for unknown ids."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. "
            f"Available tasks: {sorted(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
