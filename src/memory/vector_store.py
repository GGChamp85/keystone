# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Qdrant vector store for codebase RAG.

One real Qdrant *collection* per tenant (`{settings.qdrant_collection}_{tenant_id}`),
not one shared global collection filtered by a `tenant_id` payload field —
the real cross-tenant leak that filter approach had: every call site had to
remember to pass `tenant_id` for isolation to hold, and `search()`'s
`tenant_id` used to be optional, so a caller (today, or added later) that
omitted it would silently search every tenant's code at once. With a
separate collection per tenant, there's no shared physical store to leak
across in the first place — `tenant_id` is now a required parameter
everywhere precisely so that can't regress.
"""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import get_settings
from src.memory.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            https=settings.qdrant_use_tls,
            timeout=30,
        )
        self._collection_prefix = settings.qdrant_collection
        self._embedding_dim = settings.embedding_dimension
        self._embedder = EmbeddingService()
        # Collections this process has already confirmed exist — avoids a
        # get_collections()+create_collection() round trip on every single
        # search/upsert once a tenant's collection is known to be there.
        self._ensured_collections: set[str] = set()

    def _collection_name(self, tenant_id: str) -> str:
        # Qdrant collection names are alphanumeric/underscore/hyphen-safe in
        # practice, but a UUID's hyphens read oddly next to the prefix's own
        # underscore-separated words — normalize to underscores for a
        # consistent, greppable name (e.g. "keystone_code_memory_<tenant>").
        return f"{self._collection_prefix}_{tenant_id.replace('-', '_')}"

    async def ensure_collection(self, tenant_id: str) -> None:
        name = self._collection_name(tenant_id)
        if name in self._ensured_collections:
            return
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if name not in existing:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )
            for field in ("file_path", "language", "repository"):
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema="keyword",
                )
            logger.info("vector_store.collection_created", name=name)
        self._ensured_collections.add(name)

    async def upsert_chunks(self, tenant_id: str, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        await self.ensure_collection(tenant_id)
        collection = self._collection_name(tenant_id)
        texts = [c["content"] for c in chunks]
        embeddings = await self._embedder.embed_batch(texts)
        points = [
            PointStruct(
                id=chunk["id"],
                vector=emb,
                payload={
                    "content": chunk["content"],
                    "file_path": chunk["file_path"],
                    "language": chunk["language"],
                    "repository": chunk["repository"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "file_hash": chunk.get("file_hash", ""),
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        for i in range(0, len(points), 100):
            await self._client.upsert(collection_name=collection, points=points[i : i + 100])
        logger.info("vector_store.upserted", collection=collection, count=len(points))
        return len(points)

    async def search(
        self,
        tenant_id: str,
        query: str,
        repository: str | None = None,
        language: str | None = None,
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection(tenant_id)
        collection = self._collection_name(tenant_id)
        query_embedding = await self._embedder.embed(query)
        conditions: list[Condition] = []
        if repository:
            conditions.append(FieldCondition(key="repository", match=MatchValue(value=repository)))
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        search_filter = Filter(must=conditions) if conditions else None
        # `AsyncQdrantClient.search()` was removed in qdrant-client 1.10+ in
        # favor of `query_points()` (confirmed against the actually-installed
        # 1.19.1: `search` doesn't exist as an attribute at all). This was a
        # real, previously undetected bug — `search()` silently threw
        # AttributeError on every call, always caught by engine.py's
        # `except Exception` around RAG retrieval, so it never surfaced as
        # anything louder than a logged "keystone.rag_failed" warning; no
        # test exercised a real Qdrant search before this file's tests.
        response = await self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            {
                "content": h.payload.get("content", ""),
                "file_path": h.payload.get("file_path", ""),
                "language": h.payload.get("language", ""),
                "repository": h.payload.get("repository", ""),
                "chunk_index": h.payload.get("chunk_index", 0),
                "score": h.score,
            }
            for h in response.points
        ]

    async def delete_by_ids(self, tenant_id: str, ids: list[str]) -> None:
        """Deletes specific chunk ids within one tenant's collection — used by
        incremental ingestion (src/memory/ingestion.py) to remove a shrunk or
        deleted file's now-stale chunks that a plain upsert would never touch."""
        if not ids:
            return
        await self._client.delete(collection_name=self._collection_name(tenant_id), points_selector=ids)

    async def delete_by_tenant(self, tenant_id: str) -> None:
        """Drops the tenant's entire collection — trivial and fast now that
        isolation is structural, not a filtered delete over shared storage."""
        name = self._collection_name(tenant_id)
        await self._client.delete_collection(collection_name=name)
        self._ensured_collections.discard(name)

    async def delete_by_repository(self, tenant_id: str, repository: str) -> None:
        await self._client.delete(
            collection_name=self._collection_name(tenant_id),
            points_selector=Filter(must=[FieldCondition(key="repository", match=MatchValue(value=repository))]),
        )

    async def close(self) -> None:
        await self._client.close()
