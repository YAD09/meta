"""
FastAPI + Gradio application entry point.

Provides:
  - REST API:  POST /reset, POST /step, GET /state, GET /tasks, GET /actions
  - Gradio UI: Interactive demo for Hugging Face Spaces
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

# ---------------------------------------------------------------------------
# FastAPI app + shared env instance
# ---------------------------------------------------------------------------

api = FastAPI(
    title="AI Customer Support Automation — OpenEnv",
    description=(
        "OpenEnv-compliant RL environment for training AI agents to handle "
        "SaaS customer support tickets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared env instance per server process (stateless via reset per request)
_shared_env = CustomerSupportEnv()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@api.get("/", tags=["meta"])
def root() -> Dict[str, str]:
    return {
        "name": "AI Customer Support Automation",
        "version": "1.0.0",
        "openenv_version": "0.1.0",
        "docs": "/docs",
    }


@api.get("/tasks", tags=["meta"])
def list_tasks() -> Dict[str, Any]:
    """Return all available tasks and their specifications."""
    return CustomerSupportEnv.available_tasks()


@api.get("/actions", tags=["meta"])
def list_actions() -> list[str]:
    """Return all valid action types."""
    return CustomerSupportEnv.available_actions()


class ResetBody(BaseModel):
    task_id: str = "task_easy"
    scenario_index: int = 0


class StepBody(BaseModel):
    action_type: str
    content: str | None = None
    reason: str | None = None


@api.post("/reset", tags=["openenv"])
def reset(body: ResetBody) -> Dict[str, Any]:
    """Reset the environment and return the initial observation."""
    try:
        obs = _shared_env.reset(body.task_id, body.scenario_index)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.post("/step", tags=["openenv"])
def step(body: StepBody) -> Dict[str, Any]:
    """Execute one agent action and return (observation, reward, done, info)."""
    try:
        action = Action(
            action_type=body.action_type,  # type: ignore[arg-type]
            content=body.content,
            reason=body.reason,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        obs, reward, done, info = _shared_env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@api.get("/state", tags=["openenv"])
def state() -> Dict[str, Any]:
    """Return a full snapshot of the current environment state."""
    return _shared_env.state()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _make_gradio_app() -> gr.Blocks:
    env = CustomerSupportEnv()
    episode_state: Dict[str, Any] = {}

    def reset_env(task_id: str, scenario_idx: int) -> tuple[str, str, str, str]:
        obs = env.reset(task_id, int(scenario_idx))
        episode_state.clear()
        episode_state["obs"] = obs
        history_md = _history_to_md(obs.history)
        obs_json = json.dumps(obs.model_dump(exclude={"history"}), indent=2)
        return (
            f"🎫 **{obs.ticket_id}** — Task: `{task_id}` | User: **{obs.user_type}**",
            history_md,
            obs_json,
            "✅ Environment reset. Make your first action.",
        )

    def take_action(
        action_type: str, content: str, reason: str
    ) -> tuple[str, str, str, str]:
        if "obs" not in episode_state:
            return ("", "", "", "⚠️ Please reset the environment first.")
        try:
            action = Action(
                action_type=action_type,  # type: ignore[arg-type]
                content=content or None,
                reason=reason or None,
            )
            obs, reward, done, info = env.step(action)
            episode_state["obs"] = obs

            history_md = _history_to_md(obs.history)
            obs_json = json.dumps(obs.model_dump(exclude={"history"}), indent=2)

            emoji = "✅" if done else "🔄"
            status = (
                f"{emoji} Step {info['step_number']} | "
                f"Score: **{reward.score:.3f}** | "
                f"Cumulative: **{info['cumulative_score']:.3f}** | "
                f"Done: {'YES ✅' if done else 'No'}\n\n"
                f"**Feedback:** {reward.feedback}\n\n"
                f"**Components:** `{json.dumps(reward.components)}`"
            )
            ticket_hdr = f"🎫 **{obs.ticket_id}** — Status: `{obs.status}`"
            return ticket_hdr, history_md, obs_json, status
        except Exception as e:
            return ("", "", "", f"❌ Error: {e}")

    def _history_to_md(history: list) -> str:
        lines = []
        for h in history:
            icon = "👤" if h.role == "customer" else "🤖"
            lines.append(f"**{icon} {h.role.title()}:** {h.content}")
        return "\n\n---\n\n".join(lines) if lines else "_No conversation yet_"

    with gr.Blocks(
        title="AI Customer Support Automation — OpenEnv",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
# 🤖 AI Customer Support Automation
### OpenEnv-Compliant RL Environment

Train and evaluate AI agents on realistic SaaS support ticket scenarios.
Select a task difficulty, reset the environment, then choose actions to resolve the ticket.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Environment Controls")
                task_dropdown = gr.Dropdown(
                    choices=["task_easy", "task_medium", "task_hard"],
                    value="task_easy",
                    label="Task Difficulty",
                    info="easy=refund | medium=login issue | hard=angry complaint",
                )
                scenario_slider = gr.Slider(
                    minimum=0, maximum=2, step=1, value=0,
                    label="Scenario Variant (0–2)",
                )
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                gr.Markdown("### 🎯 Take Action")
                action_radio = gr.Radio(
                    choices=["reply", "refund", "escalate", "close"],
                    value="reply",
                    label="Action Type",
                )
                content_box = gr.Textbox(
                    label="Reply Content (required for 'reply')",
                    placeholder="Type your response to the customer…",
                    lines=4,
                )
                reason_box = gr.Textbox(
                    label="Reason (optional)",
                    placeholder="Why are you taking this action?",
                    lines=2,
                )
                step_btn = gr.Button("▶ Execute Action", variant="secondary")

            with gr.Column(scale=2):
                ticket_header = gr.Markdown("_Reset the environment to start_")

                gr.Markdown("### 💬 Conversation History")
                history_display = gr.Markdown("_No conversation yet_")

                gr.Markdown("### 📊 Step Result")
                result_display = gr.Markdown("_Take an action to see results_")

                gr.Markdown("### 🔍 Current Observation (JSON)")
                obs_json_display = gr.Code(language="json", label="Observation")

        reset_btn.click(
            fn=reset_env,
            inputs=[task_dropdown, scenario_slider],
            outputs=[ticket_header, history_display, obs_json_display, result_display],
        )
        step_btn.click(
            fn=take_action,
            inputs=[action_radio, content_box, reason_box],
            outputs=[ticket_header, history_display, obs_json_display, result_display],
        )

        gr.Markdown(
            """
---
**REST API** available at `/docs` · **OpenEnv** compliant: `reset()` · `step()` · `state()`
"""
        )

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio on FastAPI
# ---------------------------------------------------------------------------

gradio_app = _make_gradio_app()
app = gr.mount_gradio_app(api, gradio_app, path="/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
