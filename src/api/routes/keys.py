# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""API key and tenant management routes — root-admin only (bootstrap surface)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import generate_api_key, require_root_admin
from src.api.middleware.rate_limiter import get_budget_status
from src.api.models.requests import (
    CreateAPIKeyRequest,
    CreateTenantRequest,
    CreateUserRequest,
    LinkAPIKeyToUserRequest,
    RevokeAPIKeyRequest,
)
from src.api.models.responses import (
    APIKeyCreatedResponse,
    APIKeyInfoResponse,
    TenantResponse,
    TokenBudgetResponse,
    UserResponse,
)
from src.config import get_settings
from src.db.connection import get_db
from src.db.models import APIKey, APIKeyStatus, AuditLog, Tenant, TenantTier, User, UserRole

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_root_admin)])


async def _audit(db: AsyncSession, request: Request, action: str, target_type: str, target_id: str, **metadata):
    db.add(
        AuditLog(
            actor="root_admin",
            action=action,
            target_type=target_type,
            target_id=str(target_id),
            metadata_=metadata,
            source_ip=request.client.host if request.client else None,
        )
    )


@router.post("/tenants", response_model=TenantResponse, status_code=201)
async def create_tenant(req: CreateTenantRequest, request: Request, db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    existing = await db.execute(select(Tenant).where(Tenant.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Tenant with this email already exists")

    tenant = Tenant(
        name=req.name,
        email=req.email,
        tier=TenantTier(req.tier),
        daily_token_limit=req.daily_token_limit or settings.default_daily_token_limit,
        monthly_token_limit=req.monthly_token_limit or settings.default_monthly_token_limit,
    )
    db.add(tenant)
    await db.flush()
    await _audit(db, request, "tenant.create", "tenant", tenant.id, name=tenant.name, tier=tenant.tier.value)
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        email=tenant.email,
        tier=tenant.tier.value,
        daily_token_limit=tenant.daily_token_limit,
        monthly_token_limit=tenant.monthly_token_limit,
        max_concurrent_agents=tenant.max_concurrent_agents,
        is_active=tenant.is_active,
        created_at=tenant.created_at,
    )


@router.post("/tenants/{tenant_id}/keys", response_model=APIKeyCreatedResponse, status_code=201)
async def create_api_key(
    tenant_id: UUID,
    req: CreateAPIKeyRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    full_key, prefix, key_hash = generate_api_key()
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.now(UTC) + timedelta(days=req.expires_in_days)

    api_key = APIKey(
        tenant_id=tenant_id,
        name=req.name,
        key_prefix=prefix,
        key_hash=key_hash,
        scopes=req.scopes,
        expires_at=expires_at,
        daily_token_limit_override=req.daily_token_limit_override,
        rate_limit_override=req.rate_limit_override,
    )
    db.add(api_key)
    await db.flush()
    await _audit(db, request, "api_key.create", "api_key", api_key.id, tenant_id=str(tenant_id), prefix=prefix)

    return APIKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key=full_key,
        key_prefix=prefix,
        scopes=api_key.scopes,
        expires_at=expires_at,
        created_at=api_key.created_at,
    )


@router.get("/tenants/{tenant_id}/keys", response_model=list[APIKeyInfoResponse])
async def list_api_keys(tenant_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(APIKey).where(APIKey.tenant_id == tenant_id).order_by(APIKey.created_at.desc()))
    keys = result.scalars().all()
    return [
        APIKeyInfoResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            status=k.status.value,
            scopes=k.scopes or [],
            user_id=k.user_id,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.post("/tenants/{tenant_id}/users", response_model=UserResponse, status_code=201)
async def create_user(
    tenant_id: UUID,
    req: CreateUserRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A user with this email already exists")

    user = User(tenant_id=tenant_id, name=req.name, email=req.email, role=UserRole(req.role))
    db.add(user)
    await db.flush()
    await _audit(db, request, "user.create", "user", user.id, tenant_id=str(tenant_id), role=user.role.value)
    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        name=user.name,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/tenants/{tenant_id}/users", response_model=list[UserResponse])
async def list_users(tenant_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.tenant_id == tenant_id).order_by(User.created_at.desc()))
    return [
        UserResponse(
            id=u.id,
            tenant_id=u.tenant_id,
            name=u.name,
            email=u.email,
            role=u.role.value,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in result.scalars().all()
    ]


@router.post("/tenants/{tenant_id}/keys/{key_id}/link-user", response_model=APIKeyInfoResponse)
async def link_api_key_to_user(
    tenant_id: UUID,
    key_id: UUID,
    req: LinkAPIKeyToUserRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Link an existing API key to a real user, so tasks/memories it creates
    are attributed to that person and role-gated actions (e.g. approving a
    tenant-wide memory) become available to it."""
    key = await db.get(APIKey, key_id)
    if not key or key.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="API key not found for this tenant")

    user = await db.get(User, UUID(req.user_id))
    if not user or user.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="User not found for this tenant")

    key.user_id = user.id
    await _audit(db, request, "api_key.link_user", "api_key", key.id, tenant_id=str(tenant_id), user_id=str(user.id))

    return APIKeyInfoResponse(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        status=key.status.value,
        scopes=key.scopes or [],
        user_id=key.user_id,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        created_at=key.created_at,
    )


@router.post("/tenants/{tenant_id}/keys/revoke")
async def revoke_api_key(
    tenant_id: UUID,
    req: RevokeAPIKeyRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(
            APIKey.tenant_id == tenant_id,
            APIKey.key_prefix == req.key_prefix,
            APIKey.status == APIKeyStatus.ACTIVE,
        )
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="Active key with this prefix not found")
    key.status = APIKeyStatus.REVOKED
    await _audit(db, request, "api_key.revoke", "api_key", key.id, tenant_id=str(tenant_id), prefix=req.key_prefix)
    return {"status": "revoked", "key_prefix": req.key_prefix}


@router.get("/tenants/{tenant_id}/budget", response_model=TokenBudgetResponse)
async def get_tenant_budget(tenant_id: UUID, db: AsyncSession = Depends(get_db)):
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    budget = await get_budget_status(tenant.id, tenant.daily_token_limit, tenant.monthly_token_limit)
    return TokenBudgetResponse(**budget, tenant_id=tenant.id)
