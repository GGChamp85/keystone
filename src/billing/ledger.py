# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the usage ledger.

`UsageRecord` (src/db/models.py) existed as a table that nothing wrote to;
token accounting lived only in Redis counters that expire. This module is
the durable record: one row per (tenant, api key, model role, hour), with
prompt/completion/total tokens, request count and the dollars they cost,
upserted atomically (`INSERT ... ON CONFLICT ... DO UPDATE` on the unique
bucket index) so concurrent requests never lose an increment.

Dollars come from `Settings.model_prices_per_million` — USD per 1,000,000
tokens per model role, set by the operator from their own GPU economics
(benchmarks/cost_model.py's GPUCostProfile is the calculator). An unpriced
role is recorded with cost 0 and the summary says pricing is not
configured, rather than inventing a number.

Written to from the two places tokens are actually spent: the gateway
(src/api/routes/completions.py, both paths) and the agent engine
(src/orchestrator/engine.py, per role from the task's execution trace).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import ModelRole, UsageRecord

logger = structlog.get_logger(__name__)


def hour_bucket(moment: datetime | None = None) -> datetime:
    now = moment or datetime.now(UTC)
    return now.replace(minute=0, second=0, microsecond=0)


def price_per_million(model_role: str) -> float | None:
    """USD per 1M tokens for `model_role`, or None when the operator has not priced it."""
    prices = get_settings().model_prices_per_million or {}
    value = prices.get(model_role)
    return float(value) if value is not None else None


def cost_usd(model_role: str, total_tokens: int) -> float:
    price = price_per_million(model_role)
    return 0.0 if price is None else (total_tokens / 1_000_000) * price


async def record_usage(
    tenant_id: UUID,
    *,
    model_role: str,
    prompt_tokens: int,
    completion_tokens: int,
    api_key_id: UUID | None = None,
    request_count: int = 1,
    moment: datetime | None = None,
) -> None:
    """
    Add one request's tokens (and their dollars) to the hour bucket. Never
    raises — billing bookkeeping must not fail the request it accounts for;
    a failure is logged with the exact numbers so it can be reconciled.
    """
    try:
        role = ModelRole(model_role)
    except ValueError:
        logger.warning("ledger.unknown_role", model_role=model_role, tenant_id=str(tenant_id))
        return
    total = prompt_tokens + completion_tokens
    dollars = cost_usd(model_role, total)
    values = {
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "date": hour_bucket(moment),
        "model_role": role,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "request_count": request_count,
        "estimated_cost_usd": dollars,
    }
    stmt = insert(UsageRecord).values(**values)
    stmt = stmt.on_conflict_do_update(
        # Infers the unique NULLS NOT DISTINCT index `uq_usage_bucket` (an index, not a named constraint).
        index_elements=["tenant_id", "api_key_id", "date", "model_role"],
        set_={
            "prompt_tokens": UsageRecord.prompt_tokens + stmt.excluded.prompt_tokens,
            "completion_tokens": UsageRecord.completion_tokens + stmt.excluded.completion_tokens,
            "total_tokens": UsageRecord.total_tokens + stmt.excluded.total_tokens,
            "request_count": UsageRecord.request_count + stmt.excluded.request_count,
            "estimated_cost_usd": func.coalesce(UsageRecord.estimated_cost_usd, 0.0) + stmt.excluded.estimated_cost_usd,
        },
    )
    try:
        async with get_db_context() as db:
            await db.execute(stmt)
    except Exception as exc:
        logger.error(
            "ledger.record_failed",
            tenant_id=str(tenant_id),
            model_role=model_role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=dollars,
            error=str(exc),
        )


async def record_task_usage(tenant_id: UUID, api_key_id: UUID | None, trace: list[dict[str, Any]]) -> None:
    """One ledger entry per model role a task spent tokens on, from its execution trace."""
    per_role: dict[str, tuple[int, int]] = {}
    for record in trace:
        role = str(record.get("model_role") or "")
        prompt = int(record.get("prompt_tokens") or 0)
        completion = int(record.get("completion_tokens") or 0)
        if role not in {m.value for m in ModelRole}:
            continue  # "sandbox"/"tools" records spend no model tokens
        if prompt == 0 and completion == 0:
            continue
        p, c = per_role.get(role, (0, 0))
        per_role[role] = (p + prompt, c + completion)
    for role, (prompt, completion) in per_role.items():
        await record_usage(
            tenant_id,
            model_role=role,
            prompt_tokens=prompt,
            completion_tokens=completion,
            api_key_id=api_key_id,
            request_count=0,  # a task is not a gateway request; its calls are counted by tokens, not requests
        )


async def month_to_date_cost_usd(tenant_id: UUID) -> float:
    """This calendar month's priced spend for a tenant (gateway requests and agent tasks), in USD."""
    now = hour_bucket()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    stmt = select(func.coalesce(func.sum(UsageRecord.estimated_cost_usd), 0.0)).where(
        UsageRecord.tenant_id == tenant_id, UsageRecord.date >= month_start
    )
    async with get_db_context() as db:
        return float((await db.execute(stmt)).scalar_one() or 0.0)


@dataclass(frozen=True)
class UsageRow:
    day: datetime
    model_role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int
    estimated_cost_usd: float


async def usage_summary(tenant_id: UUID, *, days: int = 30, api_key_id: UUID | None = None) -> dict[str, Any]:
    """Per-day, per-role totals for the last `days` days plus grand totals and whether pricing is configured."""
    since = hour_bucket() - timedelta(days=days)
    day = func.date_trunc("day", UsageRecord.date).label("day")
    stmt = (
        select(
            day,
            UsageRecord.model_role,
            func.sum(UsageRecord.prompt_tokens),
            func.sum(UsageRecord.completion_tokens),
            func.sum(UsageRecord.total_tokens),
            func.sum(UsageRecord.request_count),
            func.sum(func.coalesce(UsageRecord.estimated_cost_usd, 0.0)),
        )
        .where(UsageRecord.tenant_id == tenant_id, UsageRecord.date >= since)
        .group_by(day, UsageRecord.model_role)
        .order_by(day.desc(), UsageRecord.model_role)
    )
    if api_key_id is not None:
        stmt = stmt.where(UsageRecord.api_key_id == api_key_id)
    async with get_db_context() as db:
        result = await db.execute(stmt)
        rows = [
            UsageRow(
                day=r[0],
                model_role=r[1].value if hasattr(r[1], "value") else str(r[1]),
                prompt_tokens=int(r[2] or 0),
                completion_tokens=int(r[3] or 0),
                total_tokens=int(r[4] or 0),
                request_count=int(r[5] or 0),
                estimated_cost_usd=float(r[6] or 0.0),
            )
            for r in result.all()
        ]
    prices = get_settings().model_prices_per_million or {}
    return {
        "tenant_id": str(tenant_id),
        "days": days,
        "since": since.isoformat(),
        "pricing_configured": bool(prices),
        "prices_per_million": {k: float(v) for k, v in prices.items()},
        "rows": [r.__dict__ for r in rows],
        "totals": {
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
            "completion_tokens": sum(r.completion_tokens for r in rows),
            "total_tokens": sum(r.total_tokens for r in rows),
            "request_count": sum(r.request_count for r in rows),
            "estimated_cost_usd": round(sum(r.estimated_cost_usd for r in rows), 6),
        },
    }
