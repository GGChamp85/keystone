# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Temporal Worker
Runs the Temporal worker that processes coding agent workflows.
Deployed as its own process/container (never colocated with the API
server) — see the `worker` service in docker-compose.yml / Helm.
Start separately:
  python -m src.temporal.worker
"""

from __future__ import annotations

import asyncio
import signal

import structlog
from temporalio.client import Client
from temporalio.worker import Worker

from src.config import get_settings
from src.temporal.activities import run_agent_task
from src.temporal.finetune_activities import run_finetune_job_activity
from src.temporal.finetune_workflow import FineTuneJobWorkflow
from src.temporal.workflows import CodingAgentWorkflow

logger = structlog.get_logger(__name__)


async def run_worker() -> None:
    settings = get_settings()

    logger.info(
        "temporal_worker_starting",
        host=settings.temporal_host,
        namespace=settings.temporal_namespace,
        task_queue=settings.temporal_task_queue,
    )

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[CodingAgentWorkflow, FineTuneJobWorkflow],
        activities=[run_agent_task, run_finetune_job_activity],
        max_concurrent_activities=5,
        max_concurrent_workflow_tasks=10,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("temporal_worker_running")
    async with worker:
        await stop_event.wait()
    logger.info("temporal_worker_stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
