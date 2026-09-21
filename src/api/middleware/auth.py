"""
Keystone — API Key authentication.

OpenAI-compatible: keys are sent as `Authorization: Bearer ks-{prefix}-{secret}`.
  - prefix: 8 random hex chars (stored, used for lookup)
  - secret: 48 random hex chars (only the SHA-256 hash is stored)
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.connection import get_db
from src.db.models import APIKey, APIKeyStatus, Tenant, User, UserRole

_ROLE_RANK = {UserRole.DEVELOPER: 0, UserRole.LEAD: 1, UserRole.ADMIN: 2}

_bearer_scheme = HTTPBearer(auto_error=False, description="OpenAI-style Bearer API key: ks-{prefix}-{secret}")
_root_admin_scheme = HTTPBearer(auto_error=False, description="Root admin bootstrap token")

API_KEY_PREFIX = "ks-"


# ── Key generation ────────────────────────────────────────────


def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (full_key, prefix, key_hash).
    The full key is shown once; only prefix + hash are persisted.
    """
    prefix = secrets.token_hex(4)  # 8 chars
    secret = secrets.token_hex(24)  # 48 chars
    full_key = f"{API_KEY_PREFIX}{prefix}-{secret}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, f"{API_KEY_PREFIX}{prefix}", key_hash


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ── Dependency ────────────────────────────────────────────────


async def _resolve_api_key(
    raw_key: str | None,
    db: AsyncSession,
) -> tuple[APIKey, Tenant, User | None]:
    """Look up key by hash, validate status & expiry, return key + tenant."""
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer header",
        )

    if not raw_key.startswith(API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid key format — keys start with '{API_KEY_PREFIX}'",
        )

    key_hash = hash_key(raw_key)
    stmt = select(APIKey).where(APIKey.key_hash == key_hash).options()
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if api_key.status == APIKeyStatus.REVOKED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
        )

    if api_key.status == APIKeyStatus.EXPIRED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
        await db.execute(update(APIKey).where(APIKey.id == api_key.id).values(status=APIKeyStatus.EXPIRED))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    # Load tenant
    tenant_stmt = select(Tenant).where(Tenant.id == api_key.tenant_id)
    tenant_result = await db.execute(tenant_stmt)
    tenant = tenant_result.scalar_one_or_none()

    if tenant is None or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is disabled",
        )

    # Touch last_used_at
    await db.execute(update(APIKey).where(APIKey.id == api_key.id).values(last_used_at=datetime.now(UTC)))

    user: User | None = None
    if api_key.user_id is not None:
        user_result = await db.execute(select(User).where(User.id == api_key.user_id))
        user = user_result.scalar_one_or_none()

    return api_key, tenant, user


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> tuple[APIKey, Tenant, User | None]:
    """
    FastAPI dependency.  Validates the OpenAI-style Bearer API key and attaches
    (api_key, tenant, user) to request.state for downstream use. `user` is
    None for a key not yet linked to a real person (e.g. a pre-Phase-4 key,
    or a service/system key) — every route must treat it as optional.
    """
    raw_key = credentials.credentials if credentials else None
    api_key, tenant, user = await _resolve_api_key(raw_key, db)
    request.state.api_key = api_key
    request.state.tenant = tenant
    request.state.user = user
    return api_key, tenant, user


def require_scope(scope: str):
    """Factory for scope-checking dependencies."""

    async def _check(
        auth: tuple[APIKey, Tenant, User | None] = Depends(require_auth),
    ) -> tuple[APIKey, Tenant, User | None]:
        api_key, tenant, user = auth
        if scope not in (api_key.scopes or []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks required scope: {scope}",
            )
        return api_key, tenant, user

    return _check


def require_role(min_role: UserRole):
    """Factory for a role-gated dependency, layered on top of `require_scope("agent")`.

    Requires the calling key to be linked to a real `User` (src/db/models.py)
    whose role ranks at or above `min_role` (developer < lead < admin). A key
    with no linked user is rejected — role gates are meaningless without a
    real person behind the key, so this fails closed rather than treating
    "no user" as "any role."
    """

    async def _check(
        auth: tuple[APIKey, Tenant, User | None] = Depends(require_scope("agent")),
    ) -> tuple[APIKey, Tenant, User]:
        api_key, tenant, user = auth
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This action requires a user-linked API key (see keystone users) — this key isn't linked to one",
            )
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")
        if _ROLE_RANK[user.role] < _ROLE_RANK[min_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the '{min_role.value}' role or higher; you have '{user.role.value}'",
            )
        return api_key, tenant, user

    return _check


# ── Root admin (bootstraps tenants/API keys — src/api/routes/keys.py) ──────


async def require_root_admin(
    credentials: HTTPAuthorizationCredentials | None = Security(_root_admin_scheme),
) -> None:
    """
    FastAPI dependency guarding tenant/API-key bootstrap endpoints.

    These endpoints can't be gated by a tenant-scoped API key (a tenant has none
    yet when it's being created), so they're gated by a separate root admin
    bootstrap token instead, sourced from secrets (OpenBao in production, env
    var in dev) — never a tenant API key.
    """
    settings = get_settings()
    configured = settings.keystone_root_admin_token
    presented = credentials.credentials if credentials else None

    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Root admin token is not configured on this server",
        )

    if not presented or not hmac.compare_digest(presented, configured.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing root admin token",
        )
