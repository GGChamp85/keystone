# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/rl/rollout.py — runs actual RL episodes
against the live self-hosted sandbox daemon (the same one production
agent testing uses), no mocks and no fake reward signal: reward comes
from a real pytest run's real exit code.
"""

from __future__ import annotations

import pytest

from src.rl.rollout import RLTask, RolloutCoordinator, RolloutEnvironment, trajectories_to_dpo_pairs

pytestmark = pytest.mark.integration

ADD_TEST_CODE = """
from solution import add

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
"""


def _add_task(max_steps: int = 3) -> RLTask:
    return RLTask(
        id="add-task",
        instruction="Write a Python function add(a, b) that returns a + b",
        language="python",
        test_code=ADD_TEST_CODE,
        test_command="python -m pytest test_solution.py -v --tb=short",
        max_steps=max_steps,
    )


async def test_environment_rewards_correct_solution_positively():
    coordinator = RolloutCoordinator(max_concurrent=2)
    env = RolloutEnvironment(coordinator._manager, _add_task())
    try:
        await env.reset()
        step = await env.step("def add(a, b):\n    return a + b\n")
        assert step.reward == 1.0
        assert step.done is True
        assert step.exit_code == 0
    finally:
        await env.close()
    await coordinator.aclose()


async def test_environment_rewards_wrong_solution_negatively():
    coordinator = RolloutCoordinator(max_concurrent=2)
    env = RolloutEnvironment(coordinator._manager, _add_task())
    try:
        await env.reset()
        step = await env.step("def add(a, b):\n    return a - b\n")
        assert step.reward < 0
        assert step.exit_code != 0
    finally:
        await env.close()
    await coordinator.aclose()


async def test_probe_action_does_not_score():
    coordinator = RolloutCoordinator(max_concurrent=2)
    env = RolloutEnvironment(coordinator._manager, _add_task())
    try:
        await env.reset()
        step = await env.step("$ ls -la")
        assert step.reward == 0.0
        assert step.done is False
    finally:
        await env.close()
    await coordinator.aclose()


async def test_full_trajectory_with_retry_succeeds_and_produces_dpo_pair():
    task = _add_task(max_steps=3)
    attempts = iter(
        [
            "def add(a, b):\n    return a - b\n",  # wrong first attempt
            "def add(a, b):\n    return a + b\n",  # correct second attempt
        ]
    )

    async def policy(_observation: str) -> str:
        return next(attempts)

    coordinator = RolloutCoordinator(max_concurrent=2)
    try:
        trajectories = await coordinator.collect_rollouts([task], policy)
    finally:
        await coordinator.aclose()

    traj = trajectories[0]
    assert len(traj.steps) == 2
    assert traj.succeeded is True
    assert traj.total_reward == pytest.approx(0.9)

    pairs = trajectories_to_dpo_pairs(trajectories)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "def add(a, b):\n    return a + b\n"
    assert pairs[0]["rejected"] == "def add(a, b):\n    return a - b\n"


async def test_task_exhausting_max_steps_without_success_is_not_succeeded():
    task = _add_task(max_steps=1)

    async def always_wrong(_observation: str) -> str:
        return "def add(a, b):\n    return 0\n"

    coordinator = RolloutCoordinator(max_concurrent=2)
    try:
        trajectories = await coordinator.collect_rollouts([task], always_wrong)
    finally:
        await coordinator.aclose()

    traj = trajectories[0]
    assert traj.succeeded is False
    assert traj.steps[-1].done is True  # done because max_steps was hit, not because it passed
