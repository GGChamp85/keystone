#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

"""CLI tool to create tenants and API keys."""

import asyncio
import sys

sys.path.insert(0, "/app")

from sqlalchemy import select

from src.api.middleware.auth import generate_api_key
from src.db.connection import async_session_factory, init_db
from src.db.models import APIKey, Tenant, TenantTier


async def main():
    await init_db()

    print("=" * 60)
    print("Keystone — API Key Generator")
    print("=" * 60)

    async with async_session_factory() as db:
        # List existing tenants
        result = await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
        tenants = result.scalars().all()

        tenant = None
        if tenants:
            print("\nExisting tenants:")
            for i, t in enumerate(tenants, 1):
                print(f"  {i}. {t.name} ({t.email}) — {t.tier.value}")
            choice = input("\nSelect tenant number or press Enter to create new: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(tenants):
                tenant = tenants[int(choice) - 1]

        if tenant is None:
            name = input("Tenant name: ").strip()
            email = input("Tenant email: ").strip()
            tier = input("Tier (free/starter/pro/enterprise) [starter]: ").strip() or "starter"
            tenant = Tenant(name=name, email=email, tier=TenantTier(tier))
            db.add(tenant)
            await db.flush()
            print(f"\nCreated tenant: {tenant.name} (ID: {tenant.id})")

        # Create API key
        key_name = input("\nAPI key name [default]: ").strip() or "default"
        scopes_input = input("Scopes (comma-separated) [inference,agent]: ").strip() or "inference,agent"
        scopes = [s.strip() for s in scopes_input.split(",")]

        full_key, prefix, key_hash = generate_api_key()
        api_key = APIKey(
            tenant_id=tenant.id,
            name=key_name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=scopes,
        )
        db.add(api_key)
        await db.commit()

        print("\n" + "=" * 60)
        print("API KEY CREATED — SAVE THIS, IT WON'T BE SHOWN AGAIN!")
        print("=" * 60)
        print(f"  Tenant:  {tenant.name}")
        print(f"  Name:    {key_name}")
        print(f"  Key:     {full_key}")
        print(f"  Prefix:  {prefix}")
        print(f"  Scopes:  {scopes}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
