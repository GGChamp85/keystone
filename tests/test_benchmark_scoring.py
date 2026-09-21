# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for benchmarks/scoring.py — actually runs generated
code through the real self-hosted sandbox (Docker), no mocks. Model calls
aren't exercised here (no live LLM in this test environment); instead we
feed the scorer known-good and known-bad "model responses" to prove the
scoring pipeline itself — code extraction, sandbox execution, real
pass/fail — is correct. That's the part Keystone controls; the model call
is just an HTTP request already covered by src/inference/client.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.scoring import BenchmarkTask, extract_code, load_all_tasks, score_completion

pytestmark = pytest.mark.integration

TASKS_DIR = Path(__file__).parent.parent / "benchmarks" / "tasks"


def test_load_all_tasks_finds_the_real_task_files():
    tasks = load_all_tasks(TASKS_DIR)
    ids = {t.id for t in tasks}
    assert {"fizzbuzz", "lru_cache", "rate_limiter"}.issubset(ids)


def test_extract_code_pulls_fenced_block():
    response = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps!"
    assert extract_code(response) == "def f():\n    return 1"


def test_extract_code_falls_back_to_raw_when_no_fence():
    response = "def f():\n    return 1"
    assert extract_code(response) == response


@pytest.fixture
def fizzbuzz_task() -> BenchmarkTask:
    return BenchmarkTask.load(TASKS_DIR / "fizzbuzz.json")


async def test_correct_solution_scores_as_passed(fizzbuzz_task: BenchmarkTask):
    correct_response = """```python
def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
```"""
    result = await score_completion(fizzbuzz_task, correct_response, model_label="test-known-good")
    assert result.passed, f"expected pass, got stderr: {result.stderr}"
    assert result.exit_code == 0


async def test_wrong_solution_scores_as_failed(fizzbuzz_task: BenchmarkTask):
    wrong_response = """```python
def fizzbuzz(n: int) -> str:
    return "always wrong"
```"""
    result = await score_completion(fizzbuzz_task, wrong_response, model_label="test-known-bad")
    assert not result.passed
    assert result.exit_code != 0
    # pytest's assertion diff is written to stdout, not stderr
    assert "assertionerror" in result.stdout.lower()


async def test_syntactically_broken_solution_scores_as_failed_not_crashed(fizzbuzz_task: BenchmarkTask):
    broken_response = """```python
def fizzbuzz(n: int
    this is not valid python at all
```"""
    result = await score_completion(fizzbuzz_task, broken_response, model_label="test-broken")
    assert not result.passed  # must fail cleanly, not raise out of score_completion


async def test_lru_cache_task_scores_correctly_against_a_real_solution():
    task = BenchmarkTask.load(TASKS_DIR / "lru_cache.json")
    response = """```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value) -> None:
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
```"""
    result = await score_completion(task, response, model_label="test-lru")
    assert result.passed, f"expected pass, got stderr: {result.stderr}"
