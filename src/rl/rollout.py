# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — RL rollout coordinator.

Drives the same self-hosted sandbox pool that production agent testing
uses (src/sandbox/manager.py) as the RL environment execution layer —
"sandboxes and training infra are native to the same stack," not a
separate execution path maintained twice.

This module implements the environment/rollout-collection layer: reset a
sandboxed task, step it with a policy-chosen action, score the result,
repeat until done, in parallel across many tasks. It does NOT implement a
PPO/GRPO training loop itself — `collect_rollouts` takes a `policy_fn`
(any async callable mapping an observation to an action) as a parameter,
so the actual policy can be a real InferenceClient-backed model call, a
TRL PPOTrainer's sampling step, or a fixed baseline for testing. Wiring an
actual PPOTrainer against a loaded policy model is a separate integration
that needs a GPU to exercise meaningfully; what's here is real and
independently testable without one — see tests/test_rl_rollout.py, which
runs full episodes against the real sandbox daemon.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog

from src.sandbox.manager import SandboxManager

logger = structlog.get_logger(__name__)

PolicyFn = Callable[[str], Awaitable[str]]  # observation -> action (a shell command)


@dataclass(frozen=True)
class RLTask:
    """One RL episode definition: a sandboxed coding task with a pass/fail test."""

    id: str
    instruction: str
    language: str = "python"
    starting_files: dict[str, str] = field(default_factory=dict)
    solution_filename: str = "solution.py"
    test_filename: str = "test_solution.py"
    test_code: str = ""
    test_command: str = "python -m pytest test_solution.py -v --tb=short"
    max_steps: int = 6


@dataclass
class RolloutStep:
    action: str
    observation: str
    reward: float
    done: bool
    exit_code: int


@dataclass
class Trajectory:
    task_id: str
    steps: list[RolloutStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def succeeded(self) -> bool:
        return bool(self.steps) and self.steps[-1].done and self.steps[-1].reward > 0


class RolloutEnvironment:
    """
    One RL episode = one sandboxed coding task. Mirrors the observation/
    action/reward/done shape a TRL-style trainer expects, backed by a real
    self-hosted sandbox rather than a simulator.
    """

    def __init__(self, manager: SandboxManager, task: RLTask, tenant_id: str = "rl-training"):
        self._manager = manager
        self._task = task
        self._tenant_id = tenant_id
        self._handle: str | None = None
        self._step_count = 0

    async def reset(self) -> str:
        """Create the sandbox and seed starting files. Returns the initial observation."""
        self._handle = await self._manager.create(template=self._task.language, tenant_id=self._tenant_id)
        self._step_count = 0

        for path, content in self._task.starting_files.items():
            await self._manager.write_file(self._handle, path, content)
        if self._task.test_code:
            await self._manager.write_file(self._handle, self._task.test_filename, self._task.test_code)

        return f"Task: {self._task.instruction}\nWrite your solution to {self._task.solution_filename}."

    async def step(self, action: str) -> RolloutStep:
        """
        `action` is either a solution file's full contents (written to
        solution_filename, then the task's test suite is run to score it)
        or a raw shell command — a leading "$ " marks it as the latter, so
        a policy can also probe the sandbox (list files, run a linter)
        before committing a final solution.
        """
        if self._handle is None:
            raise RuntimeError("step() called before reset()")

        self._step_count += 1
        is_probe = action.startswith("$ ")

        if is_probe:
            result = await self._manager.execute(self._handle, action[2:], timeout=30)
            reward = 0.0  # probing doesn't score — only a scored test run does
            done = False
            observation = f"stdout:\n{result.get('stdout', '')}\nstderr:\n{result.get('stderr', '')}"
            exit_code = result.get("exit_code", -1)
        else:
            await self._manager.write_file(self._handle, self._task.solution_filename, action)
            result = await self._manager.execute(self._handle, self._task.test_command, timeout=60)
            exit_code = result.get("exit_code", -1)
            passed = exit_code == 0
            reward = 1.0 if passed else -0.1
            done = passed or self._step_count >= self._task.max_steps
            observation = f"stdout:\n{result.get('stdout', '')}\nstderr:\n{result.get('stderr', '')}"

        logger.info(
            "rl.step",
            task_id=self._task.id,
            step=self._step_count,
            probe=is_probe,
            reward=reward,
            done=done,
            exit_code=exit_code,
        )
        return RolloutStep(action=action, observation=observation, reward=reward, done=done, exit_code=exit_code)

    async def close(self) -> None:
        if self._handle is not None:
            await self._manager.destroy(self._handle)
            self._handle = None


class RolloutCoordinator:
    """
    Runs many RolloutEnvironment episodes concurrently, bounded by
    `max_concurrent` — the same warm-pool/autoscaling sandbox capacity
    production agent testing uses, not a separate pool sized twice.
    """

    def __init__(self, manager: SandboxManager | None = None, max_concurrent: int = 8):
        self._manager = manager or SandboxManager()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run_episode(self, task: RLTask, policy_fn: PolicyFn) -> Trajectory:
        async with self._semaphore:
            env = RolloutEnvironment(self._manager, task)
            trajectory = Trajectory(task_id=task.id)
            try:
                observation = await env.reset()
                for _ in range(task.max_steps):
                    action = await policy_fn(observation)
                    step = await env.step(action)
                    trajectory.steps.append(step)
                    observation = step.observation
                    if step.done:
                        break
            finally:
                await env.close()
            return trajectory

    async def collect_rollouts(self, tasks: list[RLTask], policy_fn: PolicyFn) -> list[Trajectory]:
        return await asyncio.gather(*(self.run_episode(task, policy_fn) for task in tasks))

    async def aclose(self) -> None:
        await self._manager.aclose()


def trajectories_to_dpo_pairs(trajectories: list[Trajectory]) -> list[dict]:
    """
    Convert a batch of rollouts into DPO preference pairs — feeds directly
    into src.finetuning.data_prep.prepare_dpo_data, which already expects
    exactly this {"prompt", "chosen", "rejected"} shape. Pairs the
    highest-reward attempt against the lowest-reward attempt within each
    trajectory (skips trajectories with only one distinct outcome — no
    preference signal to learn from).
    """
    pairs: list[dict] = []
    for traj in trajectories:
        if len(traj.steps) < 2:
            continue
        best = max(traj.steps, key=lambda s: s.reward)
        worst = min(traj.steps, key=lambda s: s.reward)
        if best.reward <= worst.reward:
            continue
        pairs.append({"prompt": traj.task_id, "chosen": best.action, "rejected": worst.action})
    return pairs
