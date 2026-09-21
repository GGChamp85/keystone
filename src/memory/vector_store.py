# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Qdrant vector store for codebase RAG.
"""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
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
        self._collection = settings.qdrant_collection
        self._embedding_dim = settings.embedding_dimension
        self._embedder = EmbeddingService()

    async def ensure_collection(self) -> None:
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )
            for field in ("tenant_id", "file_path", "language", "repository"):
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema="keyword",
                )
            logger.info("vector_store.collection_created", name=self._collection)

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
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
                    "tenant_id": chunk["tenant_id"],
                    "repository": chunk["repository"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "file_hash": chunk.get("file_hash", ""),
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        for i in range(0, len(points), 100):
            await self._client.upsert(collection_name=self._collection, points=points[i : i + 100])
        logger.info("vector_store.upserted", count=len(points))
        return len(points)

    async def search(
        self,
        query: str,
        tenant_id: str | None = None,
        repository: str | None = None,
        language: str | None = None,
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embedder.embed(query)
        conditions = []
        if tenant_id:
            conditions.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))
        if repository:
            conditions.append(FieldCondition(key="repository", match=MatchValue(value=repository)))
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        search_filter = Filter(must=conditions) if conditions else None
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
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
            for h in results
        ]

    async def delete_by_tenant(self, tenant_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]),
        )

    async def delete_by_repository(self, tenant_id: str, repository: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    FieldCondition(key="repository", match=MatchValue(value=repository)),
                ]
            ),
        )

    async def close(self) -> None:
        await self._client.close()
