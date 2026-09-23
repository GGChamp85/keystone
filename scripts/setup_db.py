#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

"""Initialize the Keystone database schema."""

import asyncio
import sys

sys.path.insert(0, "/app")

from src.db.connection import engine
from src.db.models import Base


async def main():
    async with engine.begin() as conn:
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("Database schema initialized successfully.")
        print(f"Tables: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
