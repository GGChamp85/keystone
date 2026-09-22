# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Unit tests for src/inference/model_router.py — pure logic, no infra needed."""

from __future__ import annotations

from src.inference.model_router import (
    ModelRouter,
    classify_task_complexity,
    classify_task_to_role,
    resolve_model_role,
)


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


def test_complexity_simple_keyword_wins():
    assert classify_task_complexity("Fix a typo in the README") == "simple"
    assert classify_task_complexity("Bump the version number") == "simple"
    assert classify_task_complexity("Rename foo to bar") == "simple"


def test_complexity_complex_keyword_wins():
    assert classify_task_complexity("Refactor the auth module") == "complex"
    assert classify_task_complexity("Fix the race condition in the queue") == "complex"
    assert classify_task_complexity("Add a new payments endpoint") == "complex"


def test_complexity_complex_keyword_beats_short_length():
    """A short description naming a genuinely substantial change must not
    be misclassified as simple just because it's brief."""
    short_but_substantial = "Fix the race condition"
    assert len(short_but_substantial.split()) <= 12
    assert classify_task_complexity(short_but_substantial) == "complex"


def test_complexity_falls_back_to_word_count_when_no_keyword_matches():
    assert classify_task_complexity("Add pagination to the users list") == "simple"  # 6 words, no keyword
    long_ambiguous = " ".join(["word"] * 20)
    assert classify_task_complexity(long_ambiguous) == "complex"  # >12 words, no keyword, ambiguous -> complex


def test_complexity_long_ambiguous_description_defaults_to_complex_not_simple():
    """A long description with no keyword match either way is the least
    trustworthy case — it defaults to the more capable model rather than
    assuming brevity-adjacent simplicity just because no keyword hit."""
    long_ambiguous = "Do the thing we discussed in standup yesterday regarding the module everyone was talking about"
    assert len(long_ambiguous.split()) > 12
    assert classify_task_complexity(long_ambiguous) == "complex"


async def test_get_client_routes_simple_task_to_coding_fallback_when_enabled(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("TASK_COMPLEXITY_ROUTING_ENABLED", "true")
    get_settings.cache_clear()

    class _FakeClient:
        async def health(self):
            return True

    calls: list[str] = []

    def fake_get_inference_client(role):
        calls.append(role)
        return _FakeClient()

    monkeypatch.setattr("src.inference.model_router.get_inference_client", fake_get_inference_client)

    router = ModelRouter()
    try:
        _client, resolved_role = await router.get_client("auto", task_description="Fix a typo in the docstring")
        assert resolved_role == "coding_fallback"
        assert calls[0] == "coding_fallback"
    finally:
        get_settings.cache_clear()


async def test_get_client_leaves_complex_task_on_primary_coding_model(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("TASK_COMPLEXITY_ROUTING_ENABLED", "true")
    get_settings.cache_clear()

    class _FakeClient:
        async def health(self):
            return True

    monkeypatch.setattr("src.inference.model_router.get_inference_client", lambda role: _FakeClient())

    router = ModelRouter()
    try:
        _client, resolved_role = await router.get_client("auto", task_description="Refactor the payments module")
        assert resolved_role == "coding"
    finally:
        get_settings.cache_clear()


async def test_get_client_does_not_complexity_route_when_disabled(monkeypatch):
    from src.config import get_settings

    monkeypatch.delenv("TASK_COMPLEXITY_ROUTING_ENABLED", raising=False)
    get_settings.cache_clear()

    class _FakeClient:
        async def health(self):
            return True

    monkeypatch.setattr("src.inference.model_router.get_inference_client", lambda role: _FakeClient())

    router = ModelRouter()
    try:
        _client, resolved_role = await router.get_client("auto", task_description="Fix a typo in the docstring")
        assert resolved_role == "coding"  # flag off -> classify_task_to_role's plain "coding" result stands
    finally:
        get_settings.cache_clear()
