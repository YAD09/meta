# 🤖 AI Customer Support Automation

> **OpenEnv-compliant** reinforcement learning environment for training AI agents to handle real-world SaaS customer support tickets.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![OpenEnv 0.1.0](https://img.shields.io/badge/OpenEnv-0.1.0-green.svg)](#)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](#docker)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow.svg)](#hugging-face-spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

This environment simulates a SaaS customer support system where an AI agent processes support tickets. At each step, the agent observes a ticket and chooses one of four actions to resolve it. A deterministic grader scores every step with **dense rewards** (partial credit + penalties), enabling efficient RL training.

---

## Environment API

Implements the [OpenEnv](https://github.com/openenv) standard:

```python
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

env = CustomerSupportEnv()

# 1. Reset — start a new episode
obs = env.reset("task_easy")          # or "task_medium" / "task_hard"

# 2. Step — execute one action
action = Action(
    action_type=ActionType.REFUND,
    content="We apologize. Your refund has been processed.",
)
obs, reward, done, info = env.step(action)

print(f"Score: {reward.score:.3f}")   # 0.0 – 1.0
print(f"Feedback: {reward.feedback}")
print(f"Done: {done}")

# 3. State — full episode snapshot
state = env.state()
```

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket ID (e.g. `TKT-8A3F2C01`) |
| `user_message` | `str` | Most recent customer message |
| `user_type` | `str` | `premium` \| `standard` \| `trial` |
| `history` | `list[HistoryEntry]` | Full conversation (oldest first) |
| `status` | `str` | `open` \| `pending` \| `escalated` \| `closed` |
| `metadata` | `dict` | Account age, plan, previous refunds, charge amount, MFA status |
| `step_number` | `int` | Steps elapsed in this episode |
| `task_id` | `str` | Task identifier |

---

## Action Space

| Action | Required fields | Description |
|--------|----------------|-------------|
| `reply` | `content` (str) | Send a message to the customer |
| `refund` | — | Issue a refund |
| `escalate` | — | Hand off to a human agent |
| `close` | — | Mark ticket resolved |

```python
from env.models import Action, ActionType

Action(action_type=ActionType.REPLY, content="Please try resetting your password.")
Action(action_type=ActionType.REFUND)
Action(action_type=ActionType.ESCALATE, reason="MFA lockout — needs tier-2")
Action(action_type=ActionType.CLOSE,    content="Issue resolved. Ticket closed.")
```

---

## Reward Model

Rewards are **dense** — partial credit is given at every step.

```python
class Reward(BaseModel):
    score: float            # 0.0 – 1.0, overall step score
    components: dict        # breakdown by category
    feedback: str           # human-readable explanation
    done: bool              # True when episode ends
```

### Component breakdown per task

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| `resolution` | ✅ 0.50 | ✅ 0.10 | — |
| `tone_quality` | ✅ 0.20 | ✅ 0.20 | ✅ 0.15 |
| `efficiency` | ✅ 0.20 | ✅ 0.10 | ✅ 0.10 |
| `escalation_decision` | — | ✅ 0.25 | ✅ 0.25 |
| `troubleshooting` | — | ✅ 0.35 | — |
| `empathy` | — | — | ✅ 0.25 |
| `compensation` | — | — | ✅ 0.25 |
| `penalty` | ❌ −0.10 | ❌ varies | ❌ −0.10 |
| `bonus` | ✅ 0.10 | — | — |

---

## Tasks

### 🟢 task_easy — Simple Refund Request
- **Max steps:** 6  &nbsp;|&nbsp; **Success threshold:** 0.65
- Premium customer was billed incorrectly.
- **Ideal path:** `REFUND` → `CLOSE` (2 steps, clean).
- **Penalties:** over-escalating a simple refund.

### 🟡 task_medium — Login / Access Issue
- **Max steps:** 8  &nbsp;|&nbsp; **Success threshold:** 0.60
- Customer cannot log in; may have MFA enabled.
- **Ideal path (no MFA):** `REPLY` (troubleshoot) → `REPLY` (confirm) → `CLOSE`.
- **Ideal path (MFA):** `REPLY` (try backup codes) → `ESCALATE`.
- **Penalties:** escalating a resolvable login issue.

### 🔴 task_hard — Angry Customer Complaint
- **Max steps:** 10  &nbsp;|&nbsp; **Success threshold:** 0.55
- Upset/threatening customer (outage, data loss, unauthorized charge).
- **Ideal path:** `REPLY` (empathize) → `REFUND` (if applicable) → `ESCALATE`.
- **Penalties:** closing without escalation on furious premium customers; wrong compensation decisions.

---

## Project Structure

```
ai_customer_support_env/
├── env/
│   ├── __init__.py        # Public exports
│   ├── models.py          # Pydantic: Observation, Action, Reward, TaskSpec
│   ├── scenarios.py       # 9 realistic customer scenarios (3 per task)
│   ├── tasks.py           # Task registry
│   ├── graders.py         # Deterministic graders (grade_easy/medium/hard)
│   ├── simulator.py       # Customer response simulator
│   └── environment.py     # CustomerSupportEnv (reset/step/state)
├── tests/
│   ├── conftest.py        # Fixtures & action helpers
│   ├── test_models.py     # Pydantic model tests
│   ├── test_graders.py    # Grader unit tests
│   └── test_environment.py# Full lifecycle integration tests
├── app.py                 # FastAPI + Gradio (HF Spaces entry)
├── openenv.yaml           # OpenEnv metadata descriptor
├── Dockerfile             # Multi-stage, non-root, port 7860
├── .dockerignore
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### Local (Python)

```bash
cd ai_customer_support_env

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --tb=short

# Start server (REST API + Gradio UI)
python app.py
# → http://localhost:7860
```

### Docker

```bash
cd ai_customer_support_env

# Build
docker build -t ai-customer-support-env .

# Run
docker run -p 7860:7860 ai-customer-support-env
# → http://localhost:7860
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Environment info |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/actions` | List all valid actions |
| `POST` | `/reset` | Reset environment → Observation |
| `POST` | `/step` | Execute action → (Observation, Reward, done, info) |
| `GET` | `/state` | Full episode snapshot |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "scenario_index": 0}'
```

### Example — step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "refund",
    "content": "We sincerely apologize. Your refund has been processed immediately."
  }'
```

---

## Hugging Face Spaces

Deployment is fully automated — push this directory to an HF Space with `gradio` SDK:

```yaml
# In your Space's README.md frontmatter:
sdk: gradio
sdk_version: "4.36"
app_file: app.py
python_version: "3.11"
```

The Gradio UI provides an interactive demo at the Space URL.  
The REST API is available at `<space-url>/docs`.

---

## OpenEnv Validation Checklist

- [x] `reset()` → `Observation` (Pydantic model)
- [x] `step(action)` → `(Observation, Reward, bool, dict)`
- [x] `state()` → `dict`
- [x] `Reward.score` ∈ [0.0, 1.0]
- [x] Dense rewards (partial credit at every step)
- [x] 3 tasks (easy / medium / hard)
- [x] Deterministic graders (same input → same output)
- [x] Episode terminates after `max_steps` or terminal action
- [x] `step()` raises `RuntimeError` after `done=True`
- [x] `openenv.yaml` present and complete
- [x] Docker-ready (`docker build` + `docker run`)
- [x] HF Spaces compatible (`app_file: app.py`, port 7860)

---

## License

MIT © 2024 AI Customer Support Automation Contributors
