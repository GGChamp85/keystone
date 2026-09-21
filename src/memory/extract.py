# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — memory extraction.

At task completion, proposes new memories from real signal the task
itself generated: root-cause hypotheses (a re-plan happened because the
same mistake kept recurring — exactly the kind of thing worth
remembering), and quality/review findings that had to be fixed before the
task could complete. Every proposal is stored with status="proposed" and
source="auto" — never auto-approved. A human approves it (memory API's
`/approve`, or the `keystone` CLI) before it's ever recalled into a
future task's context; this is the only path memory content can reach a
production prompt without a human having seen it first (a repo-scope
proposal still needs a repo maintainer's approval, a tenant-scope one a
lead/admin's — see docs/deployment once the role model in Phase 4 lands).
"""

from __future__ import annotations

from uuid import UUID

import structlog

from src.inference.client import StructuredOutputError, get_inference_client
from src.memory.store import MemoryRecord, create_memory

logger = structlog.get_logger(__name__)

MAX_PROPOSALS_PER_TASK = 3

EXTRACT_SYSTEM_PROMPT = """\
You are the memory extraction module of Keystone Agents. A coding task just finished. Your job is to decide \
whether anything from it is worth remembering for FUTURE tasks on this codebase or team — not to summarize \
what happened.

Only propose a memory if it would genuinely help a future task avoid a real mistake or follow a real \
convention. Do not propose anything generic, obvious, or specific to this one task's details (e.g. "fixed a \
bug" is useless; "this repo's tests require DATABASE_URL set even for unit tests" is useful). Propose zero \
memories if nothing here is worth remembering — that is a normal, common, correct answer.

Each memory has a kind:
- convention: a real pattern this codebase/team follows
- preference: a stylistic or architectural choice this team prefers
- fact: something true about this codebase/environment that isn't obvious
- avoid: a specific mistake or approach that caused a real problem and should not be repeated
"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["convention", "preference", "fact", "avoid"]},
                    "content": {"type": "string", "description": "One or two sentences, standalone and specific."},
                },
                "required": ["kind", "content"],
            },
        },
    },
    "required": ["memories"],
}


def _build_signal(
    task_description: str,
    root_cause_notes: list[str],
    quality_findings: list[dict],
    review_comments: list[dict],
) -> str:
    parts = [f"## Task\n{task_description}"]
    if root_cause_notes:
        parts.append("\n## Root-Cause Findings (the plan had to be redone at least once)")
        parts.extend(f"- {note}" for note in root_cause_notes)
    blocking_quality = [f for f in quality_findings if f.get("severity") == "error"]
    if blocking_quality:
        parts.append("\n## Quality Findings That Had To Be Fixed")
        parts.extend(f"- [{f.get('tool')}] {f.get('path')}: {f.get('message', '')[:200]}" for f in blocking_quality)
    blocking_review = [c for c in review_comments if c.get("severity") in ("error", "critical")]
    if blocking_review:
        parts.append("\n## Review Comments That Had To Be Fixed")
        parts.extend(
            f"- [{c.get('severity')}] {c.get('file_path')}: {c.get('message', '')[:200]}" for c in blocking_review
        )
    return "\n".join(parts)


async def propose_memories_from_task(
    tenant_id: UUID,
    repository: str | None,
    task_description: str,
    *,
    root_cause_notes: list[str],
    quality_findings: list[dict],
    review_comments: list[dict],
) -> list[MemoryRecord]:
    """
    Real signal only — a task with no root-cause findings and no blocking quality/review comments has nothing
    interesting to extract, and this returns immediately without an LLM call rather than manufacturing a
    proposal out of a routine, uneventful task.
    """
    has_blocking_quality = any(f.get("severity") == "error" for f in quality_findings)
    has_blocking_review = any(c.get("severity") in ("error", "critical") for c in review_comments)
    if not root_cause_notes and not has_blocking_quality and not has_blocking_review:
        return []

    client = get_inference_client("reasoning")
    signal = _build_signal(task_description, root_cause_notes, quality_findings, review_comments)
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": signal},
    ]

    try:
        parsed = await client.chat_structured(messages, EXTRACT_SCHEMA, schema_name="memory_proposals")
    except StructuredOutputError as exc:
        logger.warning("memory.extraction_unavailable", tenant_id=str(tenant_id), error=str(exc))
        return []

    proposals = parsed.get("memories", [])[:MAX_PROPOSALS_PER_TASK]
    created: list[MemoryRecord] = []
    for proposal in proposals:
        kind = proposal.get("kind")
        content = (proposal.get("content") or "").strip()
        if not kind or not content:
            continue
        record = await create_memory(
            tenant_id,
            content,
            kind=kind,
            repository=repository,
            source="auto",
            status="proposed",
            created_by="agent",
        )
        created.append(record)

    if created:
        logger.info("memory.proposed", tenant_id=str(tenant_id), repository=repository, count=len(created))
    return created
