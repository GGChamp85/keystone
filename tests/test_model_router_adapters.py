# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/inference/model_router.py's
resolve_served_model_name — a real Postgres (tests/conftest.py's
requires_integration_env), real ModelAdapter rows, not a mock of the
routing decision. Kept separate from tests/test_model_router.py, which is
deliberately pure/no-infra — this is the one DB-backed piece of that
module.
"""

from __future__ import annotations

import uuid

import pytest

from src.db.connection import get_db_context
from src.db.models import AdapterStatus, ModelAdapter, Tenant, TenantTier
from src.inference.model_router import resolve_served_model_name

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="model-router-adapter-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def _add_adapter(tenant_id, *, base_model_id, name, status, is_default) -> None:
    async with get_db_context() as db:
        db.add(
            ModelAdapter(
                tenant_id=tenant_id,
                base_model_id=base_model_id,
                name=name,
                path=f"/data/adapters/{name}",
                rank=64,
                job_type="lora",
                status=status,
                is_default=is_default,
            )
        )
        await db.flush()


async def test_falls_back_to_base_model_when_no_adapter_exists(tenant_id):
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "glm-5.3-flash"


async def test_routes_to_the_default_promoted_adapter(tenant_id):
    await _add_adapter(
        tenant_id,
        base_model_id="zai-org/GLM-5.3-Flash",
        name="tenant-glm-lora-v1",
        status=AdapterStatus.PROMOTED,
        is_default=True,
    )
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "tenant-glm-lora-v1"


async def test_ignores_a_candidate_adapter_not_yet_promoted(tenant_id):
    await _add_adapter(
        tenant_id,
        base_model_id="zai-org/GLM-5.3-Flash",
        name="tenant-glm-lora-candidate",
        status=AdapterStatus.CANDIDATE,
        is_default=True,
    )
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "glm-5.3-flash"


async def test_ignores_a_promoted_adapter_that_is_not_the_default(tenant_id):
    await _add_adapter(
        tenant_id,
        base_model_id="zai-org/GLM-5.3-Flash",
        name="tenant-glm-lora-non-default",
        status=AdapterStatus.PROMOTED,
        is_default=False,
    )
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "glm-5.3-flash"


async def test_ignores_a_retired_adapter(tenant_id):
    await _add_adapter(
        tenant_id,
        base_model_id="zai-org/GLM-5.3-Flash",
        name="tenant-glm-lora-retired",
        status=AdapterStatus.RETIRED,
        is_default=True,
    )
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "glm-5.3-flash"


async def test_does_not_route_to_another_tenants_adapter(tenant_id):
    other_tenant_id = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=other_tenant_id, name="other", email=f"{other_tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    try:
        await _add_adapter(
            other_tenant_id,
            base_model_id="zai-org/GLM-5.3-Flash",
            name="other-tenants-lora",
            status=AdapterStatus.PROMOTED,
            is_default=True,
        )
        resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
        assert resolved == "glm-5.3-flash"
    finally:
        async with get_db_context() as db:
            row = await db.get(Tenant, other_tenant_id)
            if row is not None:
                await db.delete(row)


async def test_does_not_route_to_a_promoted_adapter_for_a_different_base_model(tenant_id):
    await _add_adapter(
        tenant_id,
        base_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        name="tenant-qwen-lora",
        status=AdapterStatus.PROMOTED,
        is_default=True,
    )
    resolved = await resolve_served_model_name("zai-org/GLM-5.3-Flash", tenant_id, "glm-5.3-flash")
    assert resolved == "glm-5.3-flash"
