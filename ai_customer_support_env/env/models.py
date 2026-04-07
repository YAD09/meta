"""
Pydantic models for the AI Customer Support Automation OpenEnv environment.

Models:
  - Observation: What the agent sees at each step
  - ActionType:  Enum of valid agent actions
  - Action:      Agent's chosen action + payload
  - Reward:      Dense reward with partial-credit components
  - TaskSpec:    Metadata for a task (difficulty, grader reference, etc.)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All actions an agent may take in a support ticket conversation."""

    REPLY = "reply"        # Send a text reply to the customer
    REFUND = "refund"      # Issue a refund (no reply required, but one allowed)
    ESCALATE = "escalate"  # Escalate to a human agent
    CLOSE = "close"        # Mark ticket as resolved / closed


class Action(BaseModel):
    """An action the agent wants to execute."""

    action_type: ActionType = Field(..., description="Which action to perform")
    content: Optional[str] = Field(
        None,
        description="Reply text (required for REPLY; optional but encouraged otherwise)",
    )
    reason: Optional[str] = Field(
        None,
        description="Agent's stated reason — used by graders for quality scoring",
    )

    model_config = {"use_enum_values": True}

    @field_validator("content")
    @classmethod
    def content_required_for_reply(cls, v: Optional[str], info: Any) -> Optional[str]:
        # Pydantic v2 passes ValidationInfo; retrieve values from info.data
        data = info.data if hasattr(info, "data") else {}
        if data.get("action_type") == ActionType.REPLY and not v:
            raise ValueError("'content' is required when action_type is 'reply'")
        return v


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class HistoryEntry(BaseModel):
    """A single turn in the ticket conversation."""

    role: str = Field(..., description="'customer' or 'agent'")
    content: str = Field(..., description="Message text")
    timestamp: Optional[str] = Field(None, description="ISO-8601 UTC timestamp")


class Observation(BaseModel):
    """Everything the agent can observe at a given step."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    user_message: str = Field(..., description="Most recent customer message")
    user_type: str = Field(
        ...,
        description="Customer tier: 'premium' | 'standard' | 'trial'",
    )
    history: List[HistoryEntry] = Field(
        default_factory=list,
        description="Full conversation history (oldest first)",
    )
    status: str = Field(
        "open",
        description="Ticket status: 'open' | 'pending' | 'escalated' | 'resolved' | 'closed'",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context (account_age_days, plan, previous_refunds, etc.)",
    )
    step_number: int = Field(0, description="How many steps have elapsed in this episode")
    task_id: str = Field(..., description="Which task this ticket belongs to")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Dense reward returned after each step."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall normalised score for this step [0.0, 1.0]",
    )
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual reward components (resolution, tone, efficiency, …)",
    )
    feedback: str = Field(
        "",
        description="Human-readable explanation of what drove the score",
    )
    done: bool = Field(False, description="Whether the episode has ended")


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskSpec(BaseModel):
    """Static metadata attached to each task."""

    task_id: str
    title: str
    difficulty: Difficulty
    description: str
    max_steps: int = Field(10, description="Episode terminates after this many steps")
    success_threshold: float = Field(
        0.7,
        description="Minimum cumulative score considered a solved episode",
    )
    tags: List[str] = Field(default_factory=list)
