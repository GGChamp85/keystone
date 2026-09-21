# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Embedding service using sentence-transformers (BGE model).
Runs on CPU or GPU. Batched for throughput.
"""

from __future__ import annotations

import asyncio

import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            self._model = SentenceTransformer(
                settings.embedding_model_name,
                trust_remote_code=True,
            )
            logger.info("embedding.model_loaded", model=settings.embedding_model_name)

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embed_sync, texts, batch_size)

    def _embed_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self._load_model()
        # Prefix for BGE models: "Represent this code snippet: "
        prefixed = [f"Represent this code snippet: {t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]
