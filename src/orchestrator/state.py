# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Agent state schema for the LangGraph orchestrator.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


class AgentPhase(enum.StrEnum):
    PLANNING = "planning"
    CODING = "coding"
    QUALITY = "quality"
    REVIEW = "review"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileChange:
    path: str
    original_content: str | None = None
    new_content: str = ""
    language: str = ""
    action: str = "create"  # create | modify | delete


@dataclass
class TestResult:
    test_name: str
    passed: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0


@dataclass
class ReviewComment:
    file_path: str
    line: int | None = None
    severity: str = "info"  # info | warning | error | critical
    message: str = ""
    suggestion: str | None = None


@dataclass
class QualityFinding:
    tool: str  # ruff | mypy | bandit | eslint | tsc | go vet | cargo clippy | ...
    path: str
    line: int | None = None
    severity: str = "warning"  # info | warning | error
    code: str = ""
    message: str = ""


@dataclass
class IterationRecord:
    """One full loop through the agent graph."""

    iteration: int
    phase: str
    model_role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    files_changed: list[str] = field(default_factory=list)
    test_results: list[dict] = field(default_factory=list)
    review_comments: list[dict] = field(default_factory=list)
    quality_findings: list[dict] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0


@dataclass
class AgentState:
    """
    Complete mutable state carried through the LangGraph execution.
    Persisted to PostgreSQL via the orchestrator engine after each step.
    """

    # ── Identity ──────────────────────────────────────────────
    task_id: UUID = field(default_factory=uuid4)
    tenant_id: UUID | None = None
    api_key_id: UUID | None = None
    # A git-ref-safe slug for the real user behind this task (src/db/models.py's
    # User, resolved at submission — src/orchestrator/nodes/_shared.py's
    # slugify_for_branch()), used in the working branch name
    # (keystone/<user_slug>/<task_id>) so concurrent developers' branches
    # don't collide and a PR's author is obvious from its branch alone.
    # None for a key not yet linked to a user — branch naming falls back to
    # "agent" in that case, same as before Phase 4.
    user_slug: str | None = None

    # ── Task definition ───────────────────────────────────────
    task_description: str = ""
    repository_url: str | None = None
    branch: str = "main"
    target_files: list[str] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)

    # ── Execution control ─────────────────────────────────────
    phase: AgentPhase = AgentPhase.PLANNING
    iteration: int = 0
    max_iterations: int = 15
    primary_model: str = "coding"
    enable_reasoning_review: bool = True
    enable_sandbox_testing: bool = True
    enable_quality_gates: bool = True

    # ── Plan ──────────────────────────────────────────────────
    plan: str = ""
    plan_steps: list[str] = field(default_factory=list)
    current_plan_step: int = 0

    # ── Code changes ──────────────────────────────────────────
    file_changes: list[FileChange] = field(default_factory=list)

    # ── Review ────────────────────────────────────────────────
    # Retry limits live in CircuitBreakerConfig (circuit_breaker.py), the
    # one place check() actually reads them — not duplicated here.
    review_comments: list[ReviewComment] = field(default_factory=list)
    review_passed: bool = False
    consecutive_review_failures: int = 0

    # ── Testing ───────────────────────────────────────────────
    test_results: list[TestResult] = field(default_factory=list)
    tests_passed: bool = False
    consecutive_test_failures: int = 0

    # ── Quality gates (nodes/quality.py) ─────────────────────────
    # Runs between CODING and REVIEW, repo-mode only: the repo's own real
    # lint/typecheck/security tools (repo_profile.py's lint_cmds/typecheck_cmd
    # — already baked into the sandbox image, previously detected but never
    # actually run). Bandit/mypy findings block (FIXING); everything else is
    # informational only for now, pending repo-memory-driven strictness.
    quality_findings: list[QualityFinding] = field(default_factory=list)
    consecutive_quality_failures: int = 0
    # Which tools' findings block (Settings.quality_gate_blocking_tools, per-task override
    # via AgentTaskRequest.quality_blocking_tools). bandit blocks on high/medium only.
    quality_blocking_tools: list[str] = field(default_factory=lambda: ["bandit", "mypy"])

    # ── Sandbox ───────────────────────────────────────────────
    sandbox_id: str | None = None
    sandbox_output: str = ""

    # ── Git workflow (src/orchestrator/workspace.py) ───────────
    # `branch` above is the SOURCE branch cloned from; `working_branch` is
    # the agent's own branch its commits land on. `repo_cloned` guards
    # testing_node from re-cloning on every iteration — the same sandbox
    # (and its working tree) persists across the whole task via
    # src.sandbox.manager.get_sandbox_manager()'s shared handle cache.
    repo_cloned: bool = False
    working_branch: str = ""
    # The repo's own install command (repo_profile.install_cmd) runs once,
    # right after the clone (nodes/_shared.py's install_dependencies), so
    # the real test suite has its real dependencies. `deps_installed` means
    # attempted; a non-zero exit is kept in `deps_install_error` and shown
    # to the coding model so a failing import is attributed correctly.
    deps_installed: bool = False
    deps_install_error: str = ""
    commit_sha: str | None = None
    pr_url: str | None = None
    pr_number: int | None = None

    # ── Agentic tool-use loop (src/orchestrator/tools/) ─────────
    # `max_tool_steps` bounds one coding_node call's inner LLM<->tool
    # round-trip loop (a per-call sub-budget, independent of the graph's
    # own `max_iterations`). `files_touched` is the union of paths any
    # apply_patch call has successfully changed across the whole task —
    # display/summary only; the real diff comes from `Workspace.diff()`.
    max_tool_steps: int = 25
    tool_protocol: str = "native"
    files_touched: list[str] = field(default_factory=list)
    # The ranked symbol outline (src/orchestrator/repo_map.py), built once
    # right after the clone and given to both the planner and the coding
    # loop so neither starts from a blind `list_dir`. Empty for standalone
    # (no-repository) tasks or if the map could not be built.
    repo_map: str = ""
    # Real token budget for the loop's own conversation (src/orchestrator/context.py's
    # trim_turns_to_budget), independent of `max_tokens_per_task` (the whole task's
    # spend cap). Conservative default with headroom below a 32K context model for the
    # per-turn max_tokens=8192 completion plus system/task overhead; raise per-role once
    # VLLMConfig exposes each model's real max_model_len (Phase 0).
    max_context_tokens: int = 24_000

    # ── Root-cause / re-plan (nodes/tool_execution.py) ──────────
    # Populated when fixing_node's root-cause step decides the current
    # plan itself is wrong (not just the execution) after repeated
    # consecutive test/review failures — carried into planning_node's
    # prompt on the resulting re-plan so it doesn't repeat the same
    # mistake, and never cleared, so later root-cause runs see the full
    # history of what's already been tried and ruled out.
    root_cause_notes: list[str] = field(default_factory=list)

    # ── Token accounting ──────────────────────────────────────
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    max_tokens_per_task: int = 0  # 0 = unlimited (Settings.max_tokens_per_task)

    # ── Execution trace ───────────────────────────────────────
    trace: list[IterationRecord] = field(default_factory=list)

    # ── Messages (conversation history for the LLM) ───────────
    messages: list[dict[str, str]] = field(default_factory=list)

    # ── Result ────────────────────────────────────────────────
    result_summary: str = ""
    error_message: str | None = None

    # ── RAG context retrieved from Qdrant ─────────────────────
    rag_context: str = ""

    # ── Recalled memory (src/memory/store.py) — repo-over-tenant ─
    memory_context: str = ""

    # ── Helpers ───────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def is_over_token_budget(self) -> bool:
        return self.total_tokens >= self.max_tokens_per_task

    @property
    def is_over_iteration_budget(self) -> bool:
        return self.iteration >= self.max_iterations

    @property
    def should_stop(self) -> bool:
        return (
            self.phase in (AgentPhase.COMPLETE, AgentPhase.FAILED, AgentPhase.CANCELLED)
            or self.is_over_token_budget
            or self.is_over_iteration_budget
        )

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion

    def record_iteration(self, record: IterationRecord) -> None:
        self.trace.append(record)
        self.iteration += 1

    def to_db_dict(self) -> dict[str, Any]:
        """Serialize for PostgreSQL JSONB storage."""
        return {
            "task_id": str(self.task_id),
            "phase": self.phase.value,
            "iteration": self.iteration,
            "plan": self.plan,
            "plan_steps": self.plan_steps,
            "current_plan_step": self.current_plan_step,
            "file_changes": [
                {"path": fc.path, "action": fc.action, "language": fc.language} for fc in self.file_changes
            ],
            "review_passed": self.review_passed,
            "tests_passed": self.tests_passed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "trace": [
                {
                    "iteration": r.iteration,
                    "phase": r.phase,
                    "model_role": r.model_role,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "files_changed": r.files_changed,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self.trace
            ],
        }
