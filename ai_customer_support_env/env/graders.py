"""
Deterministic graders for the three OpenEnv tasks.

Each grader returns a Reward with:
  score       — float in [0.0, 1.0]
  components  — breakdown of individual scoring dimensions
  feedback    — human-readable explanation
  done        — whether the episode should end

Grading dimensions (shared across all tasks)
────────────────────────────────────────────
1. Correctness     — Did the agent do the right thing for this issue?
2. Action selection — Did the agent pick the appropriate action type?
3. Response quality — Does the reply contain the expected keywords? (no LLM)

Keyword scoring
───────────────
  score = hits / total_keywords   → scaled to the dimension's weight
  Zero penalty for entirely missing optional dimensions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from env.models import Action, ActionType, Reward


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _kw_score(text: str, keywords: List[str]) -> float:
    """Fraction of keywords present in text (case-insensitive). Returns 0–1."""
    if not keywords or not text:
        return 0.0
    lo = text.lower()
    return sum(1 for kw in keywords if kw in lo) / len(keywords)


def _reply_text(actions: List[Action]) -> str:
    """Concatenate all reply content for keyword analysis."""
    return " ".join((a.content or "") for a in actions if str(a.action_type) == "reply")


def _action_types(actions: List[Action]) -> List[str]:
    return [str(a.action_type) for a in actions]


def _has(actions: List[Action], action_type: str) -> bool:
    return action_type in _action_types(actions)


def _efficiency(n_steps: int, ideal: int, max_steps: int) -> float:
    """1.0 at ideal steps; decays linearly to 0.0 at max_steps."""
    if n_steps <= ideal:
        return 1.0
    return max(0.0, 1.0 - (n_steps - ideal) / max(1, max_steps - ideal))


# ---------------------------------------------------------------------------
# TASK-1  (easy) — Password Reset Request
# ---------------------------------------------------------------------------
#
# Scoring breakdown               Weight
# ─────────────────────────────── ──────
# Correctness    (replied)          0.30
# Action selection (reply/close)    0.25
# Response quality (keywords)       0.30
# Efficiency                        0.15
#
# Keywords the agent's reply should contain to earn full response-quality marks:
#   reset, password, link, email, click, browser, expire / expiry / expiration
#   (7 words — score = hits/7 × 0.30)

_EASY_RESET_KEYWORDS = [
    "reset", "password", "link", "email", "click", "browser", "expire",
]

_EASY_CORRECT_ACTIONS = {"reply", "close"}
_EASY_WRONG_ACTIONS   = {"refund", "escalate"}


def grade_easy(
    actions: List[Action],
    scenario: Dict[str, Any],
    step_number: int,
) -> Reward:
    """
    Grade a password-reset episode.

    Ideal path: REPLY (step-by-step instructions) → CLOSE
    """
    seq      = _action_types(actions)
    reply_tx = _reply_text(actions)
    comps: Dict[str, float] = {}
    fb: List[str] = []

    # ── 1. Correctness (0.0–0.30): did agent actually reply? ─────────────
    replied = _has(actions, "reply")
    correctness = 0.30 if replied else 0.0
    if replied:
        fb.append("✓ Agent provided a reply")
    else:
        fb.append("✗ No reply given — customer left without instructions")
    comps["correctness"] = correctness

    # ── 2. Action selection (0.0–0.25) ───────────────────────────────────
    wrong_count = sum(1 for a in seq if a in _EASY_WRONG_ACTIONS)
    action_sel  = 0.0
    if wrong_count == 0 and replied:
        action_sel = 0.25
        fb.append("✓ Correct actions used (reply / close only)")
    elif wrong_count > 0:
        action_sel = max(0.0, 0.20 - wrong_count * 0.10)
        fb.append(f"✗ Wrong actions used: {[a for a in seq if a in _EASY_WRONG_ACTIONS]}")
    else:
        action_sel = 0.10   # partial: no wrong actions but also no reply yet
    comps["action_selection"] = round(action_sel, 3)

    # ── 3. Response quality (0.0–0.30): keyword coverage ─────────────────
    kw = _kw_score(reply_tx, _EASY_RESET_KEYWORDS)
    quality = round(kw * 0.30, 3)
    comps["response_quality"] = quality
    hit_count = round(kw * len(_EASY_RESET_KEYWORDS))
    if quality >= 0.20:
        fb.append(f"✓ Good instructions ({hit_count}/{len(_EASY_RESET_KEYWORDS)} keywords)")
    elif quality > 0:
        fb.append(f"~ Partial instructions ({hit_count}/{len(_EASY_RESET_KEYWORDS)} keywords)")
    else:
        fb.append("✗ Reply missing key password-reset instructions")

    # ── 4. Efficiency (0.0–0.15) ─────────────────────────────────────────
    eff = round(_efficiency(step_number, ideal=2, max_steps=5) * 0.15, 3)
    comps["efficiency"] = eff

    # ── Aggregate ─────────────────────────────────────────────────────────
    raw   = correctness + action_sel + quality + eff
    score = round(max(0.0, min(1.0, raw)), 4)
    done  = _has(actions, "close") or step_number >= 5

    return Reward(score=score, components=comps,
                  feedback="; ".join(fb) or "No progress", done=done)


# ---------------------------------------------------------------------------
# TASK-2  (medium) — Valid Refund Request
# ---------------------------------------------------------------------------
#
# Scoring breakdown               Weight
# ─────────────────────────────── ──────
# Correctness    (refund issued)    0.35
# Action selection                  0.20
# Response quality (politeness)     0.30
# Efficiency                        0.15
#
# Politeness keywords (7):
#   sorry, apologize, understand, certainly, unfortunately, unfortunately, help
# Refund confirmation keywords (4):
#   refund, processed, days, business

_MED_POLITE_KW  = ["sorry", "apologize", "understand", "certainly", "happy", "help", "please"]
_MED_CONFIRM_KW = ["refund", "processed", "days", "business"]
_MED_ALL_KW     = _MED_POLITE_KW + _MED_CONFIRM_KW   # 11 total


def grade_medium(
    actions: List[Action],
    scenario: Dict[str, Any],
    step_number: int,
) -> Reward:
    """
    Grade a valid-refund episode.

    Ideal path: REPLY (polite apology + confirmation) → REFUND → CLOSE
    """
    seq      = _action_types(actions)
    reply_tx = _reply_text(actions)
    comps: Dict[str, float] = {}
    fb: List[str] = []

    # ── 1. Correctness (0.0–0.35): was a refund issued? ──────────────────
    refunded = _has(actions, "refund")
    correctness = 0.35 if refunded else 0.0
    if refunded:
        fb.append("✓ Refund issued")
    else:
        fb.append("✗ No refund issued — core action missing")
    comps["correctness"] = correctness

    # ── 2. Action selection (0.0–0.20) ────────────────────────────────────
    escalated = _has(actions, "escalate")
    if refunded and not escalated:
        act_sel = 0.20
        fb.append("✓ Appropriate actions taken")
    elif refunded and escalated:
        act_sel = 0.10   # escalating a clear refund is over-cautious
        fb.append("⚠ Over-escalated a straightforward refund")
    else:
        act_sel = 0.05
    comps["action_selection"] = act_sel

    # ── 3. Response quality (0.0–0.30): politeness + confirmation ─────────
    # Split into two sub-dimensions
    polite_score  = _kw_score(reply_tx, _MED_POLITE_KW) * 0.18   # 60% of quality
    confirm_score = _kw_score(reply_tx, _MED_CONFIRM_KW) * 0.12  # 40% of quality
    quality = round(polite_score + confirm_score, 3)
    comps["response_quality"]      = quality
    comps["_politeness_sub"]       = round(polite_score, 3)
    comps["_confirmation_sub"]     = round(confirm_score, 3)

    if quality >= 0.20:
        fb.append(f"✓ Polite & informative reply (quality={quality:.2f})")
    elif quality > 0:
        fb.append(f"~ Partially polite reply (quality={quality:.2f})")
    else:
        fb.append("✗ Reply lacks apology or refund confirmation")

    # ── 4. Efficiency (0.0–0.15) ─────────────────────────────────────────
    eff = round(_efficiency(step_number, ideal=3, max_steps=6) * 0.15, 3)
    comps["efficiency"] = eff

    raw   = correctness + act_sel + quality + eff
    score = round(max(0.0, min(1.0, raw)), 4)
    done  = (_has(actions, "close") or escalated) or step_number >= 6

    return Reward(score=score, components=comps,
                  feedback="; ".join(fb) or "No progress", done=done)


# ---------------------------------------------------------------------------
# TASK-3  (hard) — Angry Premium User, Unclear Issue
# ---------------------------------------------------------------------------
#
# Two valid resolution paths:
#   Path A — Clarify:  REPLY with a clear clarifying question, then proceed
#   Path B — Escalate: REPLY (empathy first), then ESCALATE
#
# Scoring breakdown               Weight
# ─────────────────────────────── ──────
# Correctness    (clarified/escalated)  0.30
# Empathy        (tone keywords)        0.25
# Action selection                      0.20
# Response quality (clarify / escalate) 0.15
# Efficiency                            0.10
#
# Empathy keywords (8):
#   sorry, apologize, understand, frustration, priority, premium, urgent, value
# Clarifying-question signals:
#   "?", "could you", "can you", "what", "which", "when", "describe", "tell me"

_HARD_EMPATHY_KW  = ["sorry", "apologize", "understand", "frustration",
                      "priority", "premium", "urgent", "value"]
_HARD_CLARIFY_KW  = ["?", "could you", "can you", "what", "which",
                      "when", "describe", "tell me", "explain"]


def grade_hard(
    actions: List[Action],
    scenario: Dict[str, Any],
    step_number: int,
) -> Reward:
    """
    Grade an angry-premium / unclear-issue episode.

    Valid paths:
      A. REPLY (empathy + clarifying question) → … → CLOSE
      B. REPLY (empathy) → ESCALATE
    """
    seq      = _action_types(actions)
    reply_tx = _reply_text(actions)
    comps: Dict[str, float] = {}
    fb: List[str] = []

    # ── 1. Empathy (0.0–0.25) ─────────────────────────────────────────────
    empathy = round(_kw_score(reply_tx, _HARD_EMPATHY_KW) * 0.25, 3)
    comps["empathy"] = empathy
    if empathy >= 0.15:
        fb.append(f"✓ Strong empathy signal ({empathy:.2f})")
    elif empathy > 0:
        fb.append(f"~ Weak empathy ({empathy:.2f})")
    else:
        fb.append("✗ No empathy shown to upset premium customer")

    # ── 2. Correctness (0.0–0.30): clarified OR escalated? ───────────────
    escalated  = _has(actions, "escalate")
    replied    = _has(actions, "reply")
    clarified  = replied and _kw_score(reply_tx, _HARD_CLARIFY_KW) > 0.15

    if escalated:
        correctness = 0.30
        fb.append("✓ Escalated to specialist (valid resolution path)")
    elif clarified:
        correctness = 0.25
        fb.append("✓ Asked clarifying question (valid resolution path)")
    else:
        correctness = 0.0
        fb.append("✗ Neither clarified nor escalated — issue left unresolved")
    comps["correctness"] = correctness

    # ── 3. Action selection (0.0–0.20) ────────────────────────────────────
    refunded = _has(actions, "refund")
    closed_early = _has(actions, "close") and not escalated and not clarified

    if closed_early:
        act_sel = 0.0
        fb.append("✗ Closed without resolving the unclear issue")
    elif refunded and not (escalated or clarified):
        act_sel = 0.05
        fb.append("⚠ Issued refund without understanding the issue")
    elif (escalated or clarified) and not closed_early:
        act_sel = 0.20
        fb.append("✓ Correct action sequence")
    else:
        act_sel = 0.10
    comps["action_selection"] = round(act_sel, 3)

    # ── 4. Response quality (0.0–0.15) ────────────────────────────────────
    # Reward for either good clarifying questions or explicit escalation language
    escalate_lang = ["escalate", "specialist", "senior", "manager", "team"]
    all_quality_kw = _HARD_CLARIFY_KW + escalate_lang
    quality = round(_kw_score(reply_tx, all_quality_kw) * 0.15, 3)
    comps["response_quality"] = quality
    if quality >= 0.10:
        fb.append(f"✓ Clear action-oriented language ({quality:.2f})")
    elif quality > 0:
        fb.append(f"~ Some relevant language ({quality:.2f})")

    # ── 5. Efficiency (0.0–0.10) ─────────────────────────────────────────
    eff = round(_efficiency(step_number, ideal=2, max_steps=8) * 0.10, 3)
    comps["efficiency"] = eff

    raw   = empathy + correctness + act_sel + quality + eff
    score = round(max(0.0, min(1.0, raw)), 4)
    done  = escalated or _has(actions, "close") or step_number >= 8

    return Reward(score=score, components=comps,
                  feedback="; ".join(fb) or "No progress", done=done)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADER_MAP = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(
    task_id: str,
    actions: List[Action],
    scenario: Dict[str, Any],
    step_number: int,
) -> Reward:
    """Route to the correct grader. Raises ValueError for unknown task_id."""
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(
            f"No grader registered for task_id={task_id!r}. "
            f"Available: {sorted(GRADER_MAP.keys())}"
        )
    return grader(actions=actions, scenario=scenario, step_number=step_number)
