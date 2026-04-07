# pytest configuration and shared fixtures

import sys
import os

# Make sure `env` package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType


@pytest.fixture(scope="function")
def env():
    """Fresh CustomerSupportEnv for each test."""
    return CustomerSupportEnv()


@pytest.fixture(scope="function")
def easy_env(env):
    """Env reset to task_easy, scenario 0."""
    env.reset("task_easy", 0)
    return env


@pytest.fixture(scope="function")
def medium_env(env):
    """Env reset to task_medium, scenario 0."""
    env.reset("task_medium", 0)
    return env


@pytest.fixture(scope="function")
def hard_env(env):
    """Env reset to task_hard, scenario 0."""
    env.reset("task_hard", 0)
    return env


def reply(text: str) -> Action:
    return Action(action_type=ActionType.REPLY, content=text)


def refund(text: str = "Refund processed.") -> Action:
    return Action(action_type=ActionType.REFUND, content=text)


def escalate(reason: str = "Needs tier-2 support") -> Action:
    return Action(action_type=ActionType.ESCALATE, reason=reason)


def close(text: str = "Ticket resolved.") -> Action:
    return Action(action_type=ActionType.CLOSE, content=text)
