# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `GET /v1/keystone/models`: the Model Library (src/inference/library.py) for this tenant —
serving roles with their live state, this tenant's adapters, and the deployable catalog — the data
behind the web UI's Model Library and Playground and an IDE's model picker.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.middleware.auth import require_scope
from src.db.models import Tenant
from src.inference.library import build_model_library

router = APIRouter(prefix="/v1/keystone", tags=["keystone"])


@router.get("/models")
async def model_library(auth: tuple = Depends(require_scope("inference"))):
    tenant: Tenant = auth[1]
    return await build_model_library(tenant.id)
