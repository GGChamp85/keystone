# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Model router.

Routes inference requests to the appropriate vLLM backend based on:
  1. Explicit model role in the request ('coding', 'coding_fallback', 'reasoning')
  2. Heuristic task classification when the user provides a free-form model string
  3. Automatic fallback chain if a model endpoint is unhealthy
"""

from __future__ import annotations

import re
from typing import ClassVar

import structlog

from src.inference.client import InferenceClient, get_inference_client

logger = structlog.get_logger(__name__)

# ── Role resolution ───────────────────────────────────────────

_ROLE_ALIASES: dict[str, str] = {
    # Explicit roles
    "coding": "coding",
    "code": "coding",
    "coder": "coding",
    "glm": "coding",
    "glm5": "coding",
    "glm-5.3": "coding",
    "glm53": "coding",
    "glm-5.3-flash": "coding",
    "coding_fallback": "coding_fallback",
    "qwen": "coding_fallback",
    "qwen2.5-coder": "coding_fallback",
    "reasoning": "reasoning",
    "reason": "reasoning",
    "deepseek": "reasoning",
    "r1": "reasoning",
    "critic": "reasoning",
}

# Keywords that hint at reasoning vs coding tasks
_REASONING_PATTERNS = re.compile(
    r"(review|verify|check|validate|critique|analyze|explain why|prove|logic|debug why|"
    r"find the bug|what.s wrong|security audit|architecture review)",
    re.IGNORECASE,
)

_CODING_PATTERNS = re.compile(
    r"(write|implement|create|build|refactor|add feature|migrate|generate|scaffold|"
    r"convert|port|update|fix|patch|modify|change)",
    re.IGNORECASE,
)


def resolve_model_role(model_input: str) -> str:
    """
    Resolve a free-form model string to a role key.
    Falls back to 'coding' if unrecognized.
    """
    normalized = model_input.strip().lower()

    # Direct alias match
    if normalized in _ROLE_ALIASES:
        return _ROLE_ALIASES[normalized]

    # Check if it contains a known model identifier — longest alias first,
    # so a more specific alias always wins over a shorter one that happens
    # to also be a substring (e.g. a full "qwen2.5-coder-32b-instruct" model
    # ID contains both "qwen" (-> coding_fallback) and "coder" (-> coding);
    # without this ordering, whichever alias was inserted first into
    # _ROLE_ALIASES would silently win regardless of which is the better
    # match).
    for alias, role in sorted(_ROLE_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in normalized:
            return role

    logger.warning("model_router.unknown_model", model_input=model_input, fallback="coding")
    return "coding"


def classify_task_to_role(task_description: str) -> str:
    """
    Heuristic: classify a natural-language task description to a model role.
    Used when Keystone Agents auto-selects the model.
    """
    if _REASONING_PATTERNS.search(task_description):
        return "reasoning"
    if _CODING_PATTERNS.search(task_description):
        return "coding"
    return "coding"


# ── Router ────────────────────────────────────────────────────


class ModelRouter:
    """
    Resolves model role → InferenceClient, with health-aware fallback.
    """

    # Ordered fallback chains per role
    FALLBACK_CHAINS: ClassVar[dict[str, list[str]]] = {
        "coding": ["coding", "coding_fallback", "reasoning"],
        "coding_fallback": ["coding_fallback", "coding", "reasoning"],
        "reasoning": ["reasoning", "coding", "coding_fallback"],
    }

    async def get_client(
        self,
        model_input: str,
        task_description: str | None = None,
    ) -> tuple[InferenceClient, str]:
        """
        Returns (client, resolved_role).

        Tries the requested role first, then walks the fallback chain
        if the primary endpoint is unhealthy.
        """
        if task_description and model_input in ("auto", ""):
            role = classify_task_to_role(task_description)
        else:
            role = resolve_model_role(model_input)

        chain = self.FALLBACK_CHAINS.get(role, [role])

        for candidate_role in chain:
            try:
                client = get_inference_client(candidate_role)
                healthy = await client.health()
                if healthy:
                    if candidate_role != role:
                        logger.info(
                            "model_router.fallback",
                            requested=role,
                            serving=candidate_role,
                        )
                    return client, candidate_role
            except Exception as exc:
                logger.warning(
                    "model_router.health_check_failed",
                    role=candidate_role,
                    error=str(exc),
                )

        # Last resort — return the originally requested client anyway
        logger.error("model_router.all_unhealthy", role=role)
        return get_inference_client(role), role


# Singleton
_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
