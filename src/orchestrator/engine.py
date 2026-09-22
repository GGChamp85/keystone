# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Orchestrator engine.

Launches agent tasks, persists state to PostgreSQL, and runs them through
Temporal for durable execution — an in-flight task survives an API pod
crash/restart, because Temporal (not an in-process asyncio.Task) owns the
execution. Falls back to a local asyncio task only when Temporal is
unreachable (e.g. a bare `python -m src.main` dev loop with no Temporal
server running) so local development doesn't require standing up the full
stack — that fallback is never used in production (VS_ENV=production).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import structlog
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from temporalio.client import Client, WorkflowHandle

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AgentTask, TaskStatus, Tenant
from src.inference.model_router import classify_task_complexity, classify_task_to_role
from src.memory.vector_store import VectorStore
from src.orchestrator.circuit_breaker import CircuitBreakerConfig, CircuitBreakerTripped
from src.orchestrator.concurrency import (
    ConcurrencyLimitExceeded,
    per_user_fair_share,
    release_task_slot,
    try_acquire_task_slot,
)
from src.orchestrator.events import publish_task_event
from src.orchestrator.graph import HeartbeatCallback, _state_to_dict, build_agent_graph
from src.orchestrator.state import AgentPhase, AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import get_sandbox_manager

logger = structlog.get_logger(__name__)

# asyncio.create_task() only holds a weak reference to the task it schedules —
# without keeping our own strong reference somewhere, the task can be
# garbage-collected mid-execution. This set exists purely to hold that
# reference for the (dev-only) non-Temporal fallback path.
_background_tasks: set[asyncio.Task] = set()


class KeystoneEngine:
    """
    The main entry point for running Keystone Agents agent tasks.
    """

    def __init__(self):
        self._vector_store: VectorStore | None = None
        self._temporal_client: Client | None = None
        self._temporal_unavailable = False  # sticky, so we don't retry-connect every request

    async def _get_vector_store(self) -> VectorStore:
        # One VectorStore (one Qdrant client connection) reused across every
        # tenant's tasks — it now holds a real per-tenant Qdrant *collection*
        # internally (src/memory/vector_store.py), lazily ensured per tenant
        # inside search()/upsert_chunks() itself, since this getter has no
        # single tenant_id to ensure ahead of time.
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    async def _get_temporal_client(self) -> Client | None:
        if self._temporal_unavailable:
            return None
        if self._temporal_client is None:
            settings = get_settings()
            try:
                self._temporal_client = await Client.connect(
                    settings.temporal_host,
                    namespace=settings.temporal_namespace,
                )
            except Exception as exc:
                logger.warning("keystone.temporal_unreachable", error=str(exc))
                self._temporal_unavailable = True
                return None
        return self._temporal_client

    async def submit_task(
        self,
        tenant_id: UUID,
        api_key_id: UUID,
        task_description: str,
        repository_url: str | None = None,
        branch: str = "main",
        file_paths: list[str] | None = None,
        model: str = "coding",
        max_iterations: int = 15,
        enable_reasoning_review: bool = True,
        enable_sandbox_testing: bool = True,
        context_files: dict[str, str] | None = None,
        user_id: UUID | None = None,
        user_slug: str | None = None,
    ) -> UUID:
        """
        Create a new agent task and start execution.
        Returns the task ID immediately; execution runs in background.

        Raises ConcurrencyLimitExceeded (caught by the API route -> 429) if
        the tenant, or the calling user's fair share of it, is already at
        Tenant.max_concurrent_agents — checked before any task row is
        created, so a rejected submission never leaves behind an orphaned
        PENDING task that will never run.
        """
        task_id = uuid4()
        settings = get_settings()

        if model == "auto":
            resolved_role = classify_task_to_role(task_description)
            if resolved_role == "coding" and settings.task_complexity_routing_enabled:
                complexity = classify_task_complexity(task_description)
                if complexity == "simple":
                    resolved_role = "coding_fallback"
            logger.info(
                "engine.auto_model_resolved",
                task_id=str(task_id),
                resolved_role=resolved_role,
            )
            model = resolved_role

        async with get_db_context() as db:
            tenant = await db.get(Tenant, tenant_id)
        tenant_max = tenant.max_concurrent_agents if tenant is not None else 3
        if not await try_acquire_task_slot(tenant_id, task_id, tenant_max, user_id=user_id):
            raise ConcurrencyLimitExceeded(tenant_max, per_user_fair_share(tenant_max) if user_id else None)

        # Persist task to database
        async with get_db_context() as db:
            task = AgentTask(
                id=task_id,
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                user_id=user_id,
                task_description=task_description,
                repository_url=repository_url,
                branch=branch,
                file_paths=file_paths or [],
                status=TaskStatus.PENDING,
                max_steps=max_iterations,
                model_role=model,
            )
            db.add(task)

        async def _publish_event(node_name: str, state: dict[str, Any]) -> None:
            await publish_task_event(task_id, node_name, state)

        task_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "api_key_id": api_key_id,
            "task_description": task_description,
            "repository_url": repository_url,
            "branch": branch,
            "file_paths": file_paths or [],
            "model": model,
            "max_iterations": max_iterations,
            "enable_reasoning_review": enable_reasoning_review,
            "enable_sandbox_testing": enable_sandbox_testing,
            "context_files": context_files or {},
            "user_id": user_id,
            "user_slug": user_slug,
            "heartbeat_callback": _publish_event,
        }

        client = await self._get_temporal_client()
        if client is not None:
            from src.temporal.workflows import AgentWorkflowInput, CodingAgentWorkflow

            workflow_id = f"keystone-task-{task_id}"
            handle = await client.start_workflow(
                CodingAgentWorkflow.run,
                AgentWorkflowInput(
                    task_id=str(task_id),
                    tenant_id=str(tenant_id),
                    api_key_id=str(api_key_id),
                    instruction=task_description,
                    repository_url=repository_url,
                    branch=branch,
                    target_files=file_paths or [],
                    context_files=context_files or {},
                    preferred_model=model,
                    max_iterations=max_iterations,
                    enable_reasoning_review=enable_reasoning_review,
                    enable_sandbox_testing=enable_sandbox_testing,
                    user_id=str(user_id) if user_id else None,
                    user_slug=user_slug,
                ),
                id=workflow_id,
                task_queue=settings.temporal_task_queue,
            )
            async with get_db_context() as db:
                await db.execute(
                    update(AgentTask)
                    .where(AgentTask.id == task_id)
                    .values(temporal_workflow_id=handle.id, temporal_run_id=handle.result_run_id)
                )
            logger.info(
                "keystone.task_submitted",
                task_id=str(task_id),
                tenant_id=str(tenant_id),
                model=model,
                execution="temporal",
                workflow_id=handle.id,
            )
        else:
            if settings.is_production:
                logger.error(
                    "keystone.temporal_unavailable_in_production",
                    task_id=str(task_id),
                    detail="Falling back to non-durable asyncio execution — task will be LOST on restart",
                )
            bg_task = asyncio.create_task(self._execute_task(**task_kwargs))
            _background_tasks.add(bg_task)
            bg_task.add_done_callback(_background_tasks.discard)
            logger.info(
                "keystone.task_submitted",
                task_id=str(task_id),
                tenant_id=str(tenant_id),
                model=model,
                execution="asyncio_fallback",
            )

        return task_id

    async def _execute_task(
        self,
        task_id: UUID,
        tenant_id: UUID,
        api_key_id: UUID,
        task_description: str,
        repository_url: str | None,
        branch: str,
        file_paths: list[str],
        model: str,
        max_iterations: int,
        enable_reasoning_review: bool,
        enable_sandbox_testing: bool,
        context_files: dict[str, str],
        user_id: UUID | None = None,
        user_slug: str | None = None,
        heartbeat_callback: HeartbeatCallback | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full agent graph for a task.
        """
        settings = get_settings()
        t0 = time.monotonic()

        # Mark as running
        async with get_db_context() as db:
            await db.execute(
                update(AgentTask)
                .where(AgentTask.id == task_id)
                .values(status=TaskStatus.RUNNING, started_at=datetime.now(UTC))
            )

        # Build initial state
        state = AgentState(
            task_id=task_id,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            task_description=task_description,
            repository_url=repository_url,
            branch=branch,
            target_files=file_paths,
            context_files=context_files,
            primary_model=model,
            max_iterations=max_iterations,
            enable_reasoning_review=enable_reasoning_review,
            enable_sandbox_testing=enable_sandbox_testing,
            user_slug=user_slug,
        )

        # Retrieve RAG context from Qdrant
        try:
            vs = await self._get_vector_store()
            rag_results = await vs.search(
                query=task_description,
                tenant_id=str(tenant_id),
                limit=5,
            )
            if rag_results:
                state.rag_context = "\n\n---\n\n".join(
                    f"File: {r['file_path']}\n```\n{r['content']}\n```" for r in rag_results
                )
        except Exception as exc:
            logger.warning("keystone.rag_failed", error=str(exc))

        # Recall relevant memory (src/memory/store.py) — repo-scope layered
        # over tenant-scope, once at task start (mirroring rag_context
        # above); every node that builds a prompt reads state.memory_context
        # rather than re-querying per node.
        try:
            from src.memory.store import recall, render_memories_for_prompt

            memories = await recall(tenant_id, repository_url, task_description)
            state.memory_context = render_memories_for_prompt(memories)
        except Exception as exc:
            logger.warning("keystone.memory_recall_failed", error=str(exc))

        # Build and run graph
        breaker_config = CircuitBreakerConfig(
            max_iterations=max_iterations,
            max_tokens_per_task=2_000_000,
            max_consecutive_test_failures=3,
            max_consecutive_review_failures=3,
            max_consecutive_quality_failures=3,
            max_wall_clock_seconds=1800,
            tenant_daily_limit=settings.default_daily_token_limit,
            tenant_monthly_limit=settings.default_monthly_token_limit,
        )

        graph = build_agent_graph(breaker_config, on_iteration=heartbeat_callback)
        state_dict = _state_to_dict(state)
        final_state: dict[str, Any] | None = None

        try:
            # Run the graph
            final_state = await graph.ainvoke(state_dict)

            # Determine final status
            final_phase = final_state.get("phase", "failed")
            if isinstance(final_phase, AgentPhase):
                final_phase = final_phase.value

            if final_phase == "complete":
                status = TaskStatus.COMPLETED
            elif final_phase == "cancelled":
                status = TaskStatus.CANCELLED
            else:
                status = TaskStatus.FAILED

            git_result: dict[str, Any] = {}
            if status == TaskStatus.COMPLETED and final_state.get("repo_cloned"):
                git_result = await self._finalize_git_workflow(task_id, final_state)

            if status == TaskStatus.COMPLETED:
                try:
                    from src.memory.extract import propose_memories_from_task

                    await propose_memories_from_task(
                        tenant_id,
                        final_state.get("repository_url"),
                        task_description,
                        root_cause_notes=final_state.get("root_cause_notes", []),
                        quality_findings=final_state.get("quality_findings", []),
                        review_comments=final_state.get("review_comments", []),
                    )
                except Exception as exc:
                    logger.warning("keystone.memory_extraction_failed", task_id=str(task_id), error=str(exc))

            # Persist final state
            elapsed = time.monotonic() - t0
            async with get_db_context() as db:
                await db.execute(
                    update(AgentTask)
                    .where(AgentTask.id == task_id)
                    .values(
                        status=status,
                        current_step=final_state.get("iteration", 0),
                        result_summary=final_state.get("result_summary", ""),
                        error_message=final_state.get("error_message"),
                        total_prompt_tokens=final_state.get("total_prompt_tokens", 0),
                        total_completion_tokens=final_state.get("total_completion_tokens", 0),
                        iteration_count=final_state.get("iteration", 0),
                        execution_trace=final_state.get("trace", []),
                        output_files=final_state.get("files_touched")
                        or [fc["path"] for fc in final_state.get("file_changes", [])],
                        output_diff=git_result.get("output_diff"),
                        branch_name=git_result.get("branch_name"),
                        commit_sha=git_result.get("commit_sha"),
                        pr_url=git_result.get("pr_url"),
                        pr_number=git_result.get("pr_number"),
                        completed_at=datetime.now(UTC),
                    )
                )

            # Record token usage to Redis
            total_tokens = final_state.get("total_prompt_tokens", 0) + final_state.get("total_completion_tokens", 0)
            if total_tokens > 0:
                from src.api.middleware.rate_limiter import record_token_usage

                await record_token_usage(tenant_id, total_tokens)

            logger.info(
                "keystone.task_complete",
                task_id=str(task_id),
                status=status.value,
                iterations=final_state.get("iteration", 0),
                total_tokens=total_tokens,
                duration_s=round(elapsed, 1),
            )

            return {
                "task_id": str(task_id),
                "status": status.value,
                "iterations": final_state.get("iteration", 0),
                "total_tokens": total_tokens,
                "result_summary": final_state.get("result_summary", ""),
            }

        except CircuitBreakerTripped as exc:
            async with get_db_context() as db:
                await db.execute(
                    update(AgentTask)
                    .where(AgentTask.id == task_id)
                    .values(
                        status=TaskStatus.FAILED,
                        error_message=f"Circuit breaker: {exc.reason}",
                        completed_at=datetime.now(UTC),
                    )
                )
            logger.error("keystone.circuit_breaker", task_id=str(task_id), reason=exc.reason)
            return {"task_id": str(task_id), "status": TaskStatus.FAILED.value, "error": exc.reason}

        except Exception as exc:
            async with get_db_context() as db:
                await db.execute(
                    update(AgentTask)
                    .where(AgentTask.id == task_id)
                    .values(
                        status=TaskStatus.FAILED,
                        error_message=str(exc),
                        completed_at=datetime.now(UTC),
                    )
                )
            logger.error("keystone.unhandled_error", task_id=str(task_id), error=str(exc))
            return {"task_id": str(task_id), "status": TaskStatus.FAILED.value, "error": str(exc)}

        finally:
            # The repo-mode sandbox (src/orchestrator/workspace.py) is kept
            # alive across every planning/coding/review/testing/fixing
            # iteration deliberately — it's destroyed exactly once, here,
            # after the graph has fully finished (success, failure, or a
            # circuit-breaker trip) and _finalize_git_workflow (if it ran)
            # has already read whatever it needed from the working tree.
            if final_state is not None and final_state.get("repo_cloned"):
                try:
                    await Workspace(get_sandbox_manager(), task_id=str(task_id)).close()
                except Exception as exc:
                    logger.warning("keystone.repo_sandbox_cleanup_failed", task_id=str(task_id), error=str(exc))

            # Concurrency slot released unconditionally, on every exit path
            # (success, failure, circuit-breaker trip, or an unhandled
            # exception) — a task that never releases its slot would
            # permanently shrink the tenant's real capacity.
            try:
                await release_task_slot(tenant_id, task_id, user_id=user_id)
            except Exception as exc:
                logger.warning("keystone.concurrency_slot_release_failed", task_id=str(task_id), error=str(exc))

    async def _finalize_git_workflow(self, task_id: UUID, final_state: dict[str, Any]) -> dict[str, Any]:
        """
        Runs once, after the graph reaches COMPLETE with a cloned repo:
        commit whatever's in the working tree, push the agent's branch, and
        open a PR if a git host token is configured. Never raises — a
        failure here (e.g. push rejected, PR API unreachable) degrades to
        "task completed, work is on an unpushed/unopened branch" rather
        than turning a successful COMPLETE back into a FAILED task; the
        reason is logged and surfaced in the returned dict for the caller
        to persist alongside the rest of the task's result.
        """
        settings = get_settings()
        fallback_slug = final_state.get("user_slug") or "agent"
        working_branch = final_state.get("working_branch") or f"keystone/{fallback_slug}/{task_id}"
        ws = Workspace(get_sandbox_manager(), task_id=str(task_id))
        result: dict[str, Any] = {"branch_name": working_branch}

        try:
            diff = await ws.diff()
            if not diff.strip():
                logger.info("keystone.git_workflow_no_changes", task_id=str(task_id))
                return result
            result["output_diff"] = diff[:200_000]  # cap — a huge diff shouldn't blow out the DB row

            commit_sha = await ws.commit_all(f"Keystone Agents: {final_state.get('task_description', '')[:200]}")
            result["commit_sha"] = commit_sha

            await ws.push(working_branch)

            if not settings.git_host_token:
                logger.info("keystone.git_workflow_pushed_no_token", task_id=str(task_id), branch=working_branch)
                return result

            from src.git.host import get_git_host, parse_owner_repo

            repository_url = final_state.get("repository_url")
            if not repository_url:
                logger.info("keystone.git_workflow_pushed_no_repository_url", task_id=str(task_id))
                return result
            owner, repo = parse_owner_repo(repository_url)
            host = get_git_host()
            base_branch = final_state.get("branch") or await host.get_default_branch(owner, repo)
            pr = await host.open_pull_request(
                owner,
                repo,
                head=working_branch,
                base=base_branch,
                title=f"Keystone Agents: {final_state.get('task_description', '')[:200]}",
                body=final_state.get("result_summary", "") or "Opened automatically by Keystone Agents.",
            )
            result["pr_url"] = pr.html_url
            result["pr_number"] = pr.number
            logger.info("keystone.git_workflow_pr_opened", task_id=str(task_id), pr_url=pr.html_url)

        except Exception as exc:
            logger.warning("keystone.git_workflow_failed", task_id=str(task_id), error=str(exc))

        return result

    async def get_task_status(self, task_id: UUID) -> dict | None:
        """Retrieve current task status from the database."""
        async with get_db_context() as db:
            from sqlalchemy import select

            stmt = select(AgentTask).where(AgentTask.id == task_id)
            result = await db.execute(stmt)
            task = result.scalar_one_or_none()
            if task is None:
                return None
            return {
                "id": task.id,
                "user_id": task.user_id,
                "status": task.status.value,
                "task_description": task.task_description,
                "current_step": task.current_step,
                "max_steps": task.max_steps,
                "model_role": task.model_role.value if task.model_role else "coding",
                "iteration_count": task.iteration_count,
                "total_prompt_tokens": task.total_prompt_tokens,
                "total_completion_tokens": task.total_completion_tokens,
                "result_summary": task.result_summary,
                "output_diff": task.output_diff,
                "output_files": task.output_files or [],
                "error_message": task.error_message,
                "execution_trace": task.execution_trace or [],
                "sandbox_id": task.sandbox_id,
                "temporal_workflow_id": task.temporal_workflow_id,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "created_at": task.created_at,
            }

    async def list_tasks(
        self,
        tenant_id: UUID,
        user_id: UUID | None = None,
        repository_url: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Team task list — who's working on what, and its PR status —
        backing the web UI's team view and `GET /v1/keystone/tasks`."""
        from sqlalchemy import select

        async with get_db_context() as db:
            stmt = select(AgentTask).where(AgentTask.tenant_id == tenant_id)
            if user_id is not None:
                stmt = stmt.where(AgentTask.user_id == user_id)
            if repository_url is not None:
                stmt = stmt.where(AgentTask.repository_url == repository_url)
            stmt = stmt.order_by(AgentTask.created_at.desc()).limit(limit)
            result = await db.execute(stmt)
            tasks = result.scalars().all()
            return [
                {
                    "id": t.id,
                    "user_id": t.user_id,
                    "status": t.status.value,
                    "task_description": t.task_description,
                    "repository_url": t.repository_url,
                    "branch": t.branch,
                    "branch_name": t.branch_name,
                    "pr_url": t.pr_url,
                    "pr_number": t.pr_number,
                    "model_role": t.model_role.value if t.model_role else "coding",
                    "error_message": t.error_message,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                    "created_at": t.created_at,
                }
                for t in tasks
            ]

    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a running task — propagates real cancellation to the Temporal
        workflow (which delivers it into the running activity) when the task
        is Temporal-backed, not just a DB status flip."""
        from sqlalchemy import select

        async with get_db_context() as db:
            row = await db.execute(select(AgentTask).where(AgentTask.id == task_id))
            task = row.scalar_one_or_none()
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            workflow_id = task.temporal_workflow_id

            result = await db.execute(
                update(AgentTask)
                .where(AgentTask.id == task_id, AgentTask.status == TaskStatus.RUNNING)
                .values(
                    status=TaskStatus.CANCELLED,
                    error_message="Cancelled by user",
                    completed_at=datetime.now(UTC),
                )
            )
            cancelled = cast("CursorResult[Any]", result).rowcount > 0

        if cancelled and workflow_id:
            client = await self._get_temporal_client()
            if client is not None:
                try:
                    handle: WorkflowHandle = client.get_workflow_handle(workflow_id)
                    await handle.cancel()
                except Exception as exc:
                    logger.warning("keystone.temporal_cancel_failed", task_id=str(task_id), error=str(exc))

        return cancelled


# Singleton
_engine: KeystoneEngine | None = None


def get_keystone_engine() -> KeystoneEngine:
    global _engine
    if _engine is None:
        _engine = KeystoneEngine()
    return _engine
