# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
src/inference/health.py and its use by the router and the gateway: cached
probes (one probe per TTL, not per request), the per-role breaker (opens
after N consecutive failures, skips the role, half-opens after the
cooldown), the router refusing to serve an unhealthy role, and the
readiness/liveness split of the health routes. The route tests use the
real app over ASGI with a scripted router and a fake DB failure; nothing
here needs a model endpoint.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from src.config import get_settings
from src.inference.health import EndpointHealthRegistry, NoHealthyModelError, endpoint_health
from src.inference.model_router import ModelRouter

from .conftest import requires_integration_env


class _Probe:
    def __init__(self, *outcomes: bool | Exception):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def health(self) -> bool:
        self.calls += 1
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    monkeypatch.setenv("INFERENCE_HEALTH_CACHE_SECONDS", "10")
    monkeypatch.setenv("INFERENCE_BREAKER_FAILURE_THRESHOLD", "3")
    monkeypatch.setenv("INFERENCE_BREAKER_OPEN_SECONDS", "30")
    get_settings.cache_clear()
    endpoint_health.reset()
    yield
    endpoint_health.reset()
    get_settings.cache_clear()


async def test_a_probe_result_is_cached_for_the_ttl():
    reg = EndpointHealthRegistry()
    probe = _Probe(True)
    for _ in range(5):
        assert await reg.is_healthy("coding", probe) is True
    assert probe.calls == 1  # one real /models call, not five


async def test_breaker_opens_after_consecutive_failures_and_half_opens_after_cooldown(monkeypatch):
    reg = EndpointHealthRegistry()
    probe = _Probe(False)
    now = [1000.0]
    monkeypatch.setattr("src.inference.health.time.monotonic", lambda: now[0])

    # The cache would hide repeated failures; step time past the TTL between probes.
    for _ in range(3):
        assert await reg.is_healthy("coding", probe) is False
        now[0] += 11
    st = reg.state("coding")
    assert st.consecutive_failures == 3 and reg.is_open("coding") and probe.calls == 3

    # Open: skipped without probing.
    assert await reg.is_healthy("coding", probe) is False
    assert probe.calls == 3
    assert 1 <= reg.retry_after_seconds(["coding"]) <= 31

    # Cooldown over: one probe is let through (half-open); success closes the breaker.
    now[0] += 31
    probe._outcomes = [True]
    assert await reg.is_healthy("coding", probe) is True
    assert probe.calls == 4 and not reg.is_open("coding") and reg.state("coding").consecutive_failures == 0


async def test_a_raising_probe_counts_as_a_failure_not_an_exception():
    reg = EndpointHealthRegistry()
    assert await reg.is_healthy("coding", _Probe(httpx.ConnectError("boom"))) is False
    assert reg.state("coding").last_error is not None and "boom" in reg.state("coding").last_error


async def test_router_skips_open_breakers_and_refuses_to_serve_when_nothing_is_healthy(monkeypatch):
    probes = {"coding": _Probe(False), "coding_fallback": _Probe(False), "reasoning": _Probe(True)}
    monkeypatch.setattr("src.inference.model_router.get_inference_client", lambda role: probes[role])
    router = ModelRouter()

    client, role = await router.get_client("coding")
    assert role == "reasoning" and client is probes["reasoning"]  # the only healthy role in the chain

    probes["reasoning"] = _Probe(False)
    endpoint_health.reset()
    with pytest.raises(NoHealthyModelError) as excinfo:
        await router.get_client("coding")
    assert excinfo.value.role == "coding" and excinfo.value.retry_after_seconds >= 1

    # Open the coding breaker (fresh registry, so the other roles' cached "unhealthy" from above is gone)
    # and confirm it is skipped without a probe.
    endpoint_health.reset()
    for _ in range(3):
        endpoint_health.record_failure("coding", "request failed")
    assert endpoint_health.is_open("coding")
    probes["coding"] = _Probe(True)
    probes["coding_fallback"] = _Probe(True)
    client, role = await router.get_client("coding")
    assert role == "coding_fallback" and probes["coding"].calls == 0


@pytest.mark.integration
@requires_integration_env
async def test_gateway_answers_503_with_retry_after_when_no_model_is_healthy():
    """A real request through the real route: no healthy endpoint -> 503 + Retry-After, and the
    model is never called (nothing to call)."""
    import uuid

    from src.api.middleware.auth import generate_api_key
    from src.db.connection import get_db_context
    from src.db.models import APIKey, Tenant, TenantTier
    from src.main import create_app

    class _NoRouter:
        async def get_client(self, model_input):
            raise NoHealthyModelError(model_input, 17)

    tid = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="health-503", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(APIKey(tenant_id=tid, name="k", key_prefix=prefix, key_hash=key_hash, scopes=["inference"]))
    try:
        app = create_app()
        with patch("src.api.routes._inference_common.get_model_router", return_value=_NoRouter()):
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {full_key}"},
                        json={"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
                    )
        assert resp.status_code == 503, resp.text
        assert resp.headers["retry-after"] == "17"
        assert "no healthy model endpoint" in resp.json()["detail"]
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, tid)
            if row is not None:
                await db.delete(row)


async def test_readiness_is_503_when_postgres_is_down_and_never_probes_a_model(monkeypatch):
    from src.main import create_app

    class _BrokenCtx:
        async def __aenter__(self):
            raise ConnectionError("postgres down")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("src.db.connection.get_db_context", lambda: _BrokenCtx())
    probed: list[str] = []
    monkeypatch.setattr("src.api.routes.health.get_inference_client", lambda role: probed.append(role))

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready" and body["components"]["postgres"] == "unreachable"
    assert body["components"]["vllm_coding"] == "unprobed"  # readiness reports models, never probes them
    assert probed == []
