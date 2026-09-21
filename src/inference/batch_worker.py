# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Batch worker process.
Drains src.inference.batch's Redis-queued jobs against whichever
embedding/vLLM pool has spare capacity. Deployed as its own process/
container (see helm/keystone/templates/ for the Deployment), scaled
independently of the interactive API tier.

Start: python -m src.inference.batch_worker
"""

from __future__ import annotations

import asyncio
import signal
import socket

import structlog

from src.api.middleware.rate_limiter import get_redis
from src.inference.batch import process_one

logger = structlog.get_logger(__name__)


async def run_worker(concurrency: int = 4) -> None:
    consumer_base = socket.gethostname()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("batch_worker.starting", concurrency=concurrency, consumer_base=consumer_base)

    async def _consumer_loop(consumer_id: int) -> None:
        r = await get_redis()
        consumer_name = f"{consumer_base}-{consumer_id}"
        while not stop_event.is_set():
            try:
                processed = await process_one(r, consumer_name, block_ms=5000)
                if not processed:
                    continue
            except Exception as exc:
                logger.warning("batch_worker.consumer_error", consumer=consumer_name, error=str(exc))
                await asyncio.sleep(2)

    tasks = [asyncio.create_task(_consumer_loop(i)) for i in range(concurrency)]
    await stop_event.wait()
    logger.info("batch_worker.stopping")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("batch_worker.stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
