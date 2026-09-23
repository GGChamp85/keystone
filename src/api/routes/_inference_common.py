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

from typing import Any

from fastapi import HTTPException

from src.api.middleware.rate_limiter import check_request_rate, check_token_budget, record_token_usage
from src.billing.ledger import record_usage
from src.config import get_settings
from src.db.models import APIKey, Tenant
from src.inference.client import InferenceClient
from src.inference.health import NoHealthyModelError
from src.inference.model_router import get_model_router, resolve_model_name_for_client


async def admit_inference_request(
    *, model: str, max_tokens: int, api_key: APIKey, tenant: Tenant
) -> tuple[InferenceClient, str, str]:
    """Returns (client, resolved_role, served_model_name) or raises the HTTP error the caller must see:
    422 over the deployment ceiling, 429 rate/budget, 503 + Retry-After when no endpoint in the role's
    chain is healthy."""
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

    try:
        client, resolved_role = await get_model_router().get_client(model)
    except NoHealthyModelError as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": str(exc.retry_after_seconds)}
        ) from exc
    # This tenant's own promoted LoRA adapter for this base model, if any — the direct customer-facing
    # gateway, so this is the highest-value place adapter routing applies (src/inference/model_router.py).
    served_name = await resolve_model_name_for_client(client, tenant.id)
    return client, resolved_role, served_name


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
