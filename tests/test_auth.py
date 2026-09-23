# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real, no-mock unit tests for API key generation/hashing (src/api/middleware/auth.py)."""

from __future__ import annotations

import hmac

from src.api.middleware.auth import API_KEY_PREFIX, generate_api_key, hash_key


def test_generated_key_has_openai_style_prefix():
    full_key, prefix, _key_hash = generate_api_key()
    assert full_key.startswith(API_KEY_PREFIX)
    assert prefix.startswith(API_KEY_PREFIX)


def test_generated_key_is_high_entropy_and_unique():
    keys = {generate_api_key()[0] for _ in range(1000)}
    assert len(keys) == 1000  # no collisions across 1000 generations


def test_key_hash_is_deterministic_and_never_stores_plaintext():
    full_key, _prefix, key_hash = generate_api_key()
    assert hash_key(full_key) == key_hash
    assert full_key not in key_hash  # the stored hash never contains the raw secret
    assert len(key_hash) == 64  # sha256 hex digest length


def test_different_keys_hash_differently():
    _, _, hash_a = generate_api_key()
    _, _, hash_b = generate_api_key()
    assert not hmac.compare_digest(hash_a, hash_b)


def test_prefix_is_short_lookup_token_not_the_secret():
    full_key, prefix, _ = generate_api_key()
    secret_part = full_key[len(prefix) + 1 :]  # after "ks-xxxxxxxx-"
    assert len(secret_part) == 48  # 24 raw bytes as hex
    assert prefix not in secret_part


def test_hash_is_peppered_with_the_deployment_secret_and_differs_from_the_legacy_sha256(monkeypatch):
    """ADR 0004: the stored hash is HMAC(pepper, key); a database alone cannot verify a key, and the legacy
    plain SHA-256 is a different value (accepted only through the dual-read path)."""
    from src.api.middleware.auth import legacy_hash_key
    from src.config import get_settings

    full_key, _prefix, key_hash = generate_api_key()
    assert key_hash != legacy_hash_key(full_key)
    assert len(key_hash) == 64 and len(legacy_hash_key(full_key)) == 64

    monkeypatch.setenv("VS_SECRET_KEY", "a" * 64)
    get_settings.cache_clear()
    try:
        with_pepper_a = hash_key(full_key)
        monkeypatch.setenv("VS_SECRET_KEY", "b" * 64)
        get_settings.cache_clear()
        with_pepper_b = hash_key(full_key)
    finally:
        get_settings.cache_clear()
    assert with_pepper_a != with_pepper_b  # the pepper is part of the hash, so it must be stable per deployment


def test_settings_know_whether_the_pepper_was_generated(monkeypatch):
    from src.config import Settings

    monkeypatch.delenv("VS_SECRET_KEY", raising=False)
    assert Settings(_env_file=None).secret_key_is_ephemeral is True
    assert Settings(_env_file=None, vs_secret_key="c" * 64).secret_key_is_ephemeral is False
