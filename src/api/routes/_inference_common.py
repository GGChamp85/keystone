# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — what every gateway request goes through before and after the model call, shared by
the OpenAI-compatible (/v1/chat/completions) and Anthropic-compatible (/v1/messages) routes so the two
never drift: the deployment's max_tokens ceiling, the per-key rate limit, the tenant token budget,
health-aware routing to a role, this tenant's promoted adapter name, and the usage accounting (token
budget + dollar ledger) afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import HTTPException

from src.api.middleware.rate_limiter import check_request_rate, check_token_budget, record_token_usage
from src.billing.ledger import month_to_date_cost_usd, record_usage
from src.config import get_settings
from src.db.models import APIKey, Tenant
from src.inference.client import InferenceClient
from src.inference.health import NoHealthyModelError
from src.inference.model_router import (
    classify_task_complexity,
    get_model_router,
    resolve_model_name_for_client,
    resolve_model_role,
)
from src.observability import budget_rejections_total, route_decisions_total, time_to_first_token_seconds
from src.security.pii_redaction import redact_pii

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Admission:
    client: InferenceClient
    role: str  # the role that will serve
    served_name: str  # the model name sent to the endpoint (the tenant's adapter, or the base model)
    requested: str  # what the caller asked for
    reason: str  # primary | fallback | cost_routing | adapter

    @property
    def decision_header(self) -> str:
        """`X-VS-Route-Decision` — the routing decision, readable by the caller."""
        return f"requested={self.requested}; served={self.role}; model={self.served_name}; reason={self.reason}"


async def check_dollar_budget(tenant: Tenant, *, surface: str) -> None:
    """429 when the tenant's monthly dollar budget (Tenant.monthly_budget_usd; 0 = none) is spent, from the
    ledger's priced cost. Headers carry the numbers so a client can show them."""
    budget = float(tenant.monthly_budget_usd or 0)
    if budget <= 0:
        return
    spent = await month_to_date_cost_usd(tenant.id)
    if spent >= budget:
        budget_rejections_total.labels(str(tenant.id), surface).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Monthly budget exhausted: ${spent:.2f} of ${budget:.2f} spent this month",
            headers={"X-VS-Budget-Spent-USD": f"{spent:.4f}", "X-VS-Budget-Limit-USD": f"{budget:.4f}"},
        )


async def admit_inference_request(
    *, model: str, max_tokens: int, api_key: APIKey, tenant: Tenant, last_user_message: str = ""
) -> Admission:
    """Everything before the model call, or the HTTP error the caller must see: 422 over the deployment
    ceiling, 429 rate / token budget / dollar budget, 503 + Retry-After when no endpoint in the role's
    chain is healthy. `last_user_message` feeds cost routing for model="auto" (GATEWAY_COST_ROUTING)."""
    settings = get_settings()
    ceiling = settings.max_tokens_per_request  # 0 = no deployment ceiling; the model's own limit applies
    if ceiling > 0 and max_tokens > ceiling:
        raise HTTPException(
            status_code=422, detail=f"max_tokens={max_tokens} exceeds this deployment's limit of {ceiling}"
        )

    rpm = api_key.rate_limit_override or settings.default_requests_per_minute  # 0 = unlimited
    await check_request_rate(tenant.id, api_key.id, rpm)

    daily_limit = api_key.daily_token_limit_override or tenant.daily_token_limit
    await check_token_budget(tenant.id, daily_limit, tenant.monthly_token_limit)
    await check_dollar_budget(tenant, surface="gateway")

    requested = model
    reason = "primary"
    if model in ("auto", "") and settings.gateway_cost_routing:
        if classify_task_complexity(last_user_message or "") == "simple":
            model, reason = "coding_fallback", "cost_routing"
        else:
            model = "coding"
    elif model in ("auto", ""):
        model = "coding"
    wanted_role = resolve_model_role(model)

    try:
        client, resolved_role = await get_model_router().get_client(model)
    except NoHealthyModelError as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": str(exc.retry_after_seconds)}
        ) from exc
    if resolved_role != wanted_role:
        reason = "fallback"
    # This tenant's own promoted LoRA adapter for this base model, if any — the direct customer-facing
    # gateway, so this is the highest-value place adapter routing applies (src/inference/model_router.py).
    served_name = await resolve_model_name_for_client(client, tenant.id)
    if served_name != client.model_id and reason == "primary":
        reason = "adapter"
    route_decisions_total.labels(requested, resolved_role, reason).inc()
    return Admission(client, resolved_role, served_name, requested, reason)


def observe_first_token(role: str, seconds: float) -> None:
    time_to_first_token_seconds.labels(role).observe(seconds)


def log_prompt_if_enabled(*, role: str, messages: list[dict[str, Any]], reply: str | None) -> None:
    """Opt-in (LOG_PROMPTS): the prompt and reply, through PII redaction, on the request's log line."""
    if not get_settings().log_prompts:
        return
    prompt_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages if m.get("content"))
    logger.info(
        "gateway.prompt",
        role=role,
        prompt=redact_pii(prompt_text).redacted_text,
        reply=redact_pii(reply or "").redacted_text,
    )


async def account_usage(
    *, tenant: Tenant, api_key: APIKey, role: str, prompt_tokens: int, completion_tokens: int
) -> None:
    """Token budget + dollar ledger for one completed request. A zero-token request records nothing."""
    total = prompt_tokens + completion_tokens
    if total <= 0:
        return
    await record_token_usage(tenant.id, total, model_role=role)
    await record_usage(
        tenant.id,
        model_role=role,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        api_key_id=api_key.id,
    )


def usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    u = usage or {}
    return int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
