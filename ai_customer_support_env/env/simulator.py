"""
Customer response simulator.

Given the current scenario and the agent's last action, produces a deterministic
follow-up customer message. This keeps episodes reproducible (no LLM calls needed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env.models import Action


# ---------------------------------------------------------------------------
# Response templates keyed by (issue_type, action_type)
# ---------------------------------------------------------------------------

_RESPONSES: Dict[tuple[str, str], str] = {
    # --- TASK 1 (Easy): Password Reset ---
    ("password_reset", "reply"): (
        "Thank you, I followed your instructions and managed to reset my password. "
        "I'm now logged in!"
    ),
    ("password_reset", "escalate"): (
        "Why are you passing me to someone else? Isn't resetting a password a simple thing?"
    ),
    ("expired_reset_link", "reply"): (
        "Okay, I requested a new link and clicked it immediately. "
        "It worked! Thanks for the help."
    ),
    ("expired_reset_link", "escalate"): (
        "I just need a functioning link, I don't need a senior specialist to help me with this."
    ),
    ("password_forgotten_new_account", "reply"): (
        "Ah, I see. I checked my spam folder and found the email. "
        "I'm good to go now."
    ),
    ("password_forgotten_new_account", "escalate"): (
        "Escalating? But I just need to know how to set my password..."
    ),

    # --- TASK 2 (Medium): Valid Refund ---
    ("duplicate_charge", "refund"): (
        "Thank you! I really appreciate how quickly you handled this. "
        "I'll keep an eye out for the refund."
    ),
    ("duplicate_charge", "reply"): (
        "Okay, I understand. Please do process the refund as soon as possible."
    ),
    ("duplicate_charge", "escalate"): (
        "Why do I need to speak to someone else? Can't you just refund me directly?"
    ),
    ("post_cancellation_charge", "refund"): (
        "Great, thank you. I hope this doesn't happen again."
    ),
    ("post_cancellation_charge", "reply"): (
        "Alright. Will the refund show up on my card within a few days?"
    ),
    ("erroneous_trial_charge", "refund"): (
        "Finally! I was really worried. Thanks for sorting this out."
    ),
    ("erroneous_trial_charge", "reply"): (
        "OK but when exactly will the refund arrive? I need it before the end of the month."
    ),

    # --- TASK 3 (Hard): Angry Premium / Unclear ---
    ("unspecified_premium_complaint", "reply"): (
        "The file exports keep failing and I get a '500 Server Error'. "
        "I've tried it on three different computers and it's completely broken."
    ),
    ("unspecified_premium_complaint", "refund"): (
        "A refund?! I don't want a refund, I want the system to WORK so I can do my job!"
    ),
    ("unspecified_premium_complaint", "escalate"): (
        "Good. Make sure the specialist calls me back immediately. "
    ),
    ("unspecified_premium_complaint", "close"): (
        "WHY ARE YOU CLOSING THIS TICKET? NOTHING IS FIXED."
    ),

    ("vague_platform_complaint", "reply"): (
        "Specifically? The dashboard isn't loading any of my metrics. "
        "It's just spinning continuously since this morning."
    ),
    ("vague_platform_complaint", "refund"): (
        "I'm not asking for my $99 back, I'm asking you to fix the dashboard!"
    ),
    ("vague_platform_complaint", "escalate"): (
        "Finally, someone who might actually know what they're doing. I'll wait."
    ),
    ("vague_platform_complaint", "close"): (
        "If you close this ticket again I am cancelling my subscription right now."
    ),

    ("escalation_demand_with_vague_issue", "reply"): (
        "The new UI update completely removed the batch tagging feature we use every single day. "
        "We have hundreds of items to tag and now we have to do it manually!"
    ),
    ("escalation_demand_with_vague_issue", "refund"): (
        "This isn't about money, it's about my team wasting hours of their day!"
    ),
    ("escalation_demand_with_vague_issue", "escalate"): (
        "I expect to hear back from the manager today with a concrete plan to fix this."
    ),
    ("escalation_demand_with_vague_issue", "close"): (
        "Unbelievable. Cancelling my account right now."
    ),
}

_DEFAULT_FOLLOW_UPS: Dict[str, str] = {
    "reply": "Thank you for your response. Please proceed.",
    "refund": "I appreciate the refund. Thank you.",
    "escalate": "Alright, I'll wait for the specialist to contact me.",
    "close": "OK, the ticket is closed.",
}


def simulate_customer_response(
    action: Action,
    scenario: Dict[str, Any],
) -> Optional[str]:
    """
    Return a deterministic customer response given the agent's action.
    Returns None when the episode is complete (CLOSE action received and issue resolved).
    """
    issue_type: str = scenario.get("issue_type", "")
    action_type: str = str(action.action_type)

    key = (issue_type, action_type)
    response = _RESPONSES.get(key) or _DEFAULT_FOLLOW_UPS.get(action_type, "Understood.")

    # Terminal actions produce no further customer message.
    if action_type == "close":
        return response  # let the grader handle done=True

    return response
