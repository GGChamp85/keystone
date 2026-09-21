# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/memory/extract.py — a fake reasoning-model client (same scripted-policy
pattern as tests/test_fixing_root_cause.py, tests/test_review_node.py) standing in for the model's own
creative output, but real Postgres writes via src/memory/store.create_memory underneath.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from src.db.connection import get_db_context
from src.db.models import Tenant, TenantTier
from src.memory.extract import propose_memories_from_task
from src.memory.store import list_memories

from .conftest import requires_integration_env

pytestmark = [pytest.mark.integration, requires_integration_env]


class FakeExtractClient:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.call_count = 0

    async def chat_structured(self, messages, schema, *, schema_name="response", **kwargs):
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
async def tenant_id():
    tid = uuid.uuid4()
    async with get_db_context() as db:
        db.add(Tenant(id=tid, name="memory-extract-test", email=f"{tid}@test.dev", tier=TenantTier.FREE))
        await db.flush()
    yield tid
    async with get_db_context() as db:
        row = await db.get(Tenant, tid)
        if row is not None:
            await db.delete(row)


async def test_no_signal_skips_the_llm_call_entirely(tenant_id):
    client = FakeExtractClient()
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            None,
            "a routine task with nothing notable",
            root_cause_notes=[],
            quality_findings=[],
            review_comments=[],
        )

    assert created == []
    assert client.call_count == 0


async def test_root_cause_notes_trigger_extraction_and_persist_as_proposed(tenant_id):
    client = FakeExtractClient(
        {
            "memories": [
                {"kind": "avoid", "content": "Do not edit generated_client.py directly, it's regenerated."},
            ]
        }
    )
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            "https://git/x/y",
            "fix the client bug",
            root_cause_notes=["The plan kept editing a generated file that gets overwritten on build."],
            quality_findings=[],
            review_comments=[],
        )

    assert client.call_count == 1
    assert len(created) == 1
    assert created[0].status == "proposed"
    assert created[0].source == "auto"
    assert created[0].repository == "https://git/x/y"

    stored = await list_memories(tenant_id, status="proposed")
    assert any(m.content == created[0].content for m in stored)


async def test_blocking_quality_finding_alone_triggers_extraction(tenant_id):
    client = FakeExtractClient({"memories": []})
    with patch("src.memory.extract.get_inference_client", return_value=client):
        await propose_memories_from_task(
            tenant_id,
            None,
            "add a feature",
            root_cause_notes=[],
            quality_findings=[{"tool": "bandit", "path": "app.py", "severity": "error", "message": "eval() found"}],
            review_comments=[],
        )

    assert client.call_count == 1


async def test_blocking_review_comment_alone_triggers_extraction(tenant_id):
    client = FakeExtractClient({"memories": []})
    with patch("src.memory.extract.get_inference_client", return_value=client):
        await propose_memories_from_task(
            tenant_id,
            None,
            "add a feature",
            root_cause_notes=[],
            quality_findings=[],
            review_comments=[{"severity": "critical", "file_path": "app.py", "message": "SQL injection"}],
        )

    assert client.call_count == 1


async def test_non_blocking_findings_alone_do_not_trigger_extraction(tenant_id):
    client = FakeExtractClient({"memories": []})
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            None,
            "add a feature",
            root_cause_notes=[],
            quality_findings=[{"tool": "ruff", "path": "app.py", "severity": "warning", "message": "unused import"}],
            review_comments=[{"severity": "info", "file_path": "app.py", "message": "consider renaming"}],
        )

    assert created == []
    assert client.call_count == 0


async def test_model_proposing_nothing_is_a_normal_empty_result(tenant_id):
    client = FakeExtractClient({"memories": []})
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            None,
            "fix a flaky test",
            root_cause_notes=["Turned out to be a real timing issue, nothing systemic."],
            quality_findings=[],
            review_comments=[],
        )

    assert created == []


async def test_structured_output_error_degrades_to_no_proposals_not_a_crash(tenant_id):
    from src.inference.client import StructuredOutputError

    client = FakeExtractClient(error=StructuredOutputError("model never returned valid JSON"))
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            None,
            "fix a bug",
            root_cause_notes=["something worth remembering"],
            quality_findings=[],
            review_comments=[],
        )

    assert created == []


async def test_caps_proposals_at_max_per_task(tenant_id):
    client = FakeExtractClient(
        {
            "memories": [
                {"kind": "fact", "content": f"fact number {i}"}
                for i in range(6)  # more than MAX_PROPOSALS_PER_TASK
            ]
        }
    )
    with patch("src.memory.extract.get_inference_client", return_value=client):
        created = await propose_memories_from_task(
            tenant_id,
            None,
            "big task",
            root_cause_notes=["a real finding"],
            quality_findings=[],
            review_comments=[],
        )

    assert len(created) == 3
