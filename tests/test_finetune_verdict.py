# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""The promotion verdict (src/finetuning/verdict.py): pass/fail/unknown from real trainer metrics."""

from __future__ import annotations

from src.finetuning.verdict import MIN_RELATIVE_IMPROVEMENT, compute_verdict


def test_pass_when_the_adapter_beats_the_base_by_the_threshold():
    v = compute_verdict({"base_eval_loss": 1.50, "eval_loss": 1.20})
    assert v.status == "pass" and v.passed
    assert v.relative_improvement == 0.2
    assert "20.0% better" in v.reason


def test_fail_when_it_does_not_beat_the_base_or_only_by_noise():
    assert compute_verdict({"base_eval_loss": 1.50, "eval_loss": 1.60}).status == "fail"
    barely = compute_verdict({"base_eval_loss": 1.000, "eval_loss": 1.000 * (1 - MIN_RELATIVE_IMPROVEMENT / 2)})
    assert barely.status == "fail" and "not better" in barely.reason


def test_unknown_without_a_held_out_comparison():
    assert compute_verdict({"eval_loss": 1.2}).status == "unknown"
    assert compute_verdict({"base_eval_loss": 1.5}).status == "unknown"
    assert compute_verdict({}).status == "unknown"
    assert compute_verdict({"base_eval_loss": float("nan"), "eval_loss": 1.0}).status == "unknown"
    assert compute_verdict({"base_eval_loss": 0, "eval_loss": 1.0}).status == "unknown"


def test_to_dict_carries_the_threshold_for_the_ui():
    d = compute_verdict({"base_eval_loss": 2.0, "eval_loss": 1.0}).to_dict()
    assert d["status"] == "pass" and d["min_relative_improvement"] == MIN_RELATIVE_IMPROVEMENT
