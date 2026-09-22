# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for benchmarks/eval_prompt.py — the ad hoc single-input eval
that compares Keystone Inference against frontier models on one arbitrary
prompt, unscored (raw completions) or scored (a real ad hoc test built
from CLI args, run through the real sandbox scoring path).

The unscored-mode fan-out and client-selection logic are pure and tested
directly here. Scored mode reuses benchmarks/scoring.py's real
score_completion — already covered end to end against real sandbox
execution in tests/test_benchmark_scoring.py — so this file adds one real
integration test proving build_adhoc_task's output actually scores
correctly through that same path for input that never touched
benchmarks/tasks/*.json, which is the new capability this module adds.
"""

from __future__ import annotations

import pytest

from benchmarks.eval_prompt import _build_clients, build_adhoc_task, run_eval
from benchmarks.model_clients import CompletionResult
from benchmarks.scoring import score_completion

pytestmark = pytest.mark.integration


def test_build_adhoc_task_uses_language_default_filenames():
    task = build_adhoc_task(
        prompt="write fizzbuzz",
        language="python",
        test_code="def test_x(): assert True",
        test_command="pytest test_solution.py",
        solution_filename=None,
        test_filename=None,
    )
    assert task.id == "adhoc"
    assert task.solution_filename == "solution.py"
    assert task.test_filename == "test_solution.py"
    assert task.test_command == "pytest test_solution.py"


def test_build_adhoc_task_honors_explicit_filenames():
    task = build_adhoc_task(
        prompt="write fizzbuzz",
        language="python",
        test_code="def test_x(): assert True",
        test_command="pytest",
        solution_filename="my_solution.py",
        test_filename="my_test.py",
    )
    assert task.solution_filename == "my_solution.py"
    assert task.test_filename == "my_test.py"


def test_build_clients_without_frontier_is_keystone_only():
    clients = _build_clients(with_frontier=False, keystone_role="coding", models=None)
    assert set(clients) == {"keystone-inference"}


def test_build_clients_models_filter_rejects_unknown_name():
    with pytest.raises(ValueError, match="not in"):
        _build_clients(with_frontier=False, keystone_role="coding", models=["does-not-exist"])


def test_build_clients_models_filter_narrows_to_the_named_subset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    clients = _build_clients(with_frontier=True, keystone_role="coding", models=["claude-opus"])
    assert set(clients) == {"claude-opus"}


async def test_run_eval_unscored_reports_raw_completions_and_no_pass_fail(monkeypatch):
    async def fake_complete(self, prompt):
        return CompletionResult(text="hello world", prompt_tokens=5, completion_tokens=7, model_label="fake")

    monkeypatch.setattr("benchmarks.eval_prompt.KeystoneInferenceClient.complete", fake_complete)

    from benchmarks.cost_model import GPUCostProfile

    report = await run_eval(
        prompt="say hello",
        task=None,
        with_frontier=False,
        keystone_role="coding",
        models=None,
        gpu_profile=GPUCostProfile(gpu_count=1, hourly_cost_usd_per_gpu=1.0, measured_tokens_per_second=1000.0),
    )
    assert report["scored"] is False
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["completion"] == "hello world"
    assert "passed" not in result
    assert report["cost_comparison"]  # Keystone Inference tokens got a real cost row


async def test_run_eval_unscored_reports_completion_errors_cleanly(monkeypatch):
    async def failing_complete(self, prompt):
        raise RuntimeError("no vLLM endpoint reachable")

    monkeypatch.setattr("benchmarks.eval_prompt.KeystoneInferenceClient.complete", failing_complete)

    from benchmarks.cost_model import GPUCostProfile

    report = await run_eval(
        prompt="say hello",
        task=None,
        with_frontier=False,
        keystone_role="coding",
        models=None,
        gpu_profile=GPUCostProfile(gpu_count=1, hourly_cost_usd_per_gpu=1.0, measured_tokens_per_second=1000.0),
    )
    assert report["results"][0]["error"] == "no vLLM endpoint reachable"


async def test_adhoc_task_scores_a_real_correct_solution_through_the_real_sandbox():
    """The new capability build_adhoc_task exists for: an ad hoc test, built
    from plain strings at the CLI rather than a benchmarks/tasks/*.json
    file, scored through the exact same real sandbox path."""
    task = build_adhoc_task(
        prompt="Write a function is_even(n) that returns True if n is even.",
        language="python",
        test_code=(
            "from solution import is_even\n"
            "def test_even():\n"
            "    assert is_even(4) is True\n"
            "def test_odd():\n"
            "    assert is_even(3) is False\n"
        ),
        test_command="pytest test_solution.py",
        solution_filename=None,
        test_filename=None,
    )
    correct = "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```"
    result = await score_completion(task, correct, model_label="test-known-good")
    assert result.passed, f"expected pass, got stderr: {result.stderr}"


async def test_adhoc_task_scores_a_real_wrong_solution_as_failed():
    task = build_adhoc_task(
        prompt="Write a function is_even(n) that returns True if n is even.",
        language="python",
        test_code=("from solution import is_even\ndef test_even():\n    assert is_even(4) is True\n"),
        test_command="pytest test_solution.py",
        solution_filename=None,
        test_filename=None,
    )
    wrong = "```python\ndef is_even(n: int) -> bool:\n    return False\n```"
    result = await score_completion(task, wrong, model_label="test-known-bad")
    assert not result.passed
