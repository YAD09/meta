# 🤖 AI Customer Support Automation — OpenEnv

A complete, production-ready, [OpenEnv](https://github.com/openenv)-compliant reinforcement learning environment for training AI agents to handle SaaS customer support tickets.

## 🌟 Features

- **OpenEnv Compliant:** Implements `reset()`, `step()`, and `state()` endpoints perfectly.
- **Stateless & RESTful:** Ready to be hosted on Hugging Face Spaces via FastAPI.
- **Dense Rewards:** Uses a deterministic `_RewardEngine` to provide partial credit at every step (+0.2 to +1.0) and penalize bad behavior like looping (-0.5).
- **Interactive UI:** Built-in Gradio interface to play with the environment manually.
- **Deterministic:** The `simulator.py` uses pre-defined customer responses (no LLM calls) ensuring perfect reproducibility for RL training.

## 🎯 Available Tasks

The environment provides 3 distinct tasks to train different skills.

| Task | Difficulty | Goal | Valid Agent Path | Maximum Steps |
|------|-----------|-------|------------|-----------|
| `task_easy` | 🟢 Easy | **Password Reset.** User can't log in. | `reply` (with instructions) → `close` | 5 |
| `task_medium` | 🟡 Medium | **Valid Refund.** Duplicate or erroneous charge. | `reply` (apology) → `refund` → `close` | 6 |
| `task_hard` | 🔴 Hard | **Angry Premium User.** Vague, frustrated complaints. | `reply` (empathy/clarify) → `escalate` | 8 |

## 🛠️ Usage / Installation

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/YAD09/meta.git
cd meta/ai_customer_support_env

# Install dependencies (requires Python 3.11+)
pip install -r requirements.txt

# Run the FastAPI server and Gradio UI
python app.py
```
> Navigate to `http://localhost:7860` for the interactive Gradio UI, or `http://localhost:7860/docs` for the REST API swagger docs.

### 2. Docker
```bash
docker build -t ai-customer-support-env .
docker run -p 7860:7860 ai-customer-support-env
```

### 3. Running Tests
The project features a comprehensive suite of 60+ tests for the Pydantic models, deterministic graders, and OpenEnv lifecycle.

```bash
python -m pytest tests/ -v --tb=short
```

## 🏗️ Architecture

- **`env/models.py`:** Pydantic definitions for `Observation`, `Action`, `Reward`, and `TaskSpec`.
- **`env/environment.py`:** The core stateful `CustomerSupportEnv` class enforcing OpenEnv logic.
- **`env/scenarios.py`:** 9 dynamically loaded customer scenarios.
- **`env/graders.py`:** Multi-dimensional scoring evaluating *correctness*, *action selection*, *content quality*, and *efficiency*.
- **`app.py`:** FastAPI mounting both the API and the Gradio Blocks frontend.

---
*Built for integration with Stable-Baselines3, Ray RLlib, or custom LLM Agents.*
