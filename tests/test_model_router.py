# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Unit tests for src/inference/model_router.py — pure logic, no infra needed."""

from __future__ import annotations

from src.inference.model_router import ModelRouter, classify_task_to_role, resolve_model_role


def test_direct_role_aliases():
    assert resolve_model_role("coding") == "coding"
    assert resolve_model_role("reasoning") == "reasoning"
    assert resolve_model_role("coding_fallback") == "coding_fallback"


def test_glm_aliases_resolve_to_coding():
    for alias in ("glm", "glm5", "glm-5.3", "glm53", "glm-5.3-flash"):
        assert resolve_model_role(alias) == "coding"


def test_qwen_aliases_resolve_to_coding_fallback():
    assert resolve_model_role("qwen") == "coding_fallback"
    assert resolve_model_role("coding_fallback") == "coding_fallback"


def test_full_qwen_model_id_resolves_to_coding_fallback_not_coding():
    """Real bug this guards against: a full Qwen model ID contains "coder" (-> coding) as a substring of
    "Qwen2.5-Coder-32B-Instruct" alongside "qwen" (-> coding_fallback) — the more specific/longer alias must
    win, not whichever happened to be inserted first into _ROLE_ALIASES."""
    assert resolve_model_role("Qwen/Qwen2.5-Coder-32B-Instruct") == "coding_fallback"
    assert resolve_model_role("qwen2.5-coder-32b-instruct") == "coding_fallback"


def test_full_glm_model_id_resolves_to_coding():
    assert resolve_model_role("zai-org/GLM-5.3-Flash") == "coding"


def test_unknown_model_falls_back_to_coding():
    assert resolve_model_role("some-totally-unknown-model") == "coding"


def test_classify_task_reasoning_keywords():
    assert classify_task_to_role("Please review this code for security issues") == "reasoning"
    assert classify_task_to_role("Debug why this test is failing") == "reasoning"


def test_classify_task_coding_keywords():
    assert classify_task_to_role("Implement a new caching layer") == "coding"
    assert classify_task_to_role("Fix the login bug") == "coding"


def test_classify_task_default_is_coding():
    assert classify_task_to_role("something ambiguous") == "coding"


def test_fallback_chains_cover_every_role_and_never_self_loop_only():
    for role, chain in ModelRouter.FALLBACK_CHAINS.items():
        assert chain[0] == role, f"{role}'s own chain must start with itself"
        assert len(chain) > 1, f"{role}'s fallback chain should have alternatives"
        assert len(chain) == len(set(chain)), f"{role}'s fallback chain has duplicates"


def test_coding_fallback_chain_includes_glm_primary():
    assert "coding" in ModelRouter.FALLBACK_CHAINS["coding_fallback"]


def test_coding_chain_includes_qwen_fallback():
    assert "coding_fallback" in ModelRouter.FALLBACK_CHAINS["coding"]
