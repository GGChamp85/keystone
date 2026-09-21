# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Planning node.
Generates a structured plan before the agent starts coding.
"""

from __future__ import annotations

import json
import time

import structlog

from src.inference.client import get_inference_client
from src.inference.model_router import resolve_model_name_for_client
from src.orchestrator.state import AgentPhase, AgentState, IterationRecord

logger = structlog.get_logger(__name__)

PLANNING_SYSTEM_PROMPT = """\
You are the planning module of Keystone Agents, an autonomous coding agent by Keystone.

Your job is to analyze a coding task and produce a precise, step-by-step execution plan.

Rules:
1. Break the task into concrete, atomic steps (max 10 steps).
2. Each step must specify which files to create or modify.
3. Consider dependencies between steps — order matters.
4. If context files or RAG context are provided, reference them.
5. Identify potential risks or edge cases.

Respond in JSON:
{
  "plan_summary": "One-line summary of the approach",
  "steps": [
    {
      "step_number": 1,
      "description": "What to do",
      "files": ["path/to/file.py"],
      "action": "create | modify | delete",
      "dependencies": []
    }
  ],
  "risks": ["potential issue 1", "..."],
  "estimated_complexity": "low | medium | high"
}
"""


async def planning_node(state: AgentState) -> AgentState:
    """
    Generate a structured plan from the task description.
    Uses the primary coding model for planning.
    """
    t0 = time.monotonic()
    client = get_inference_client(state.primary_model)
    model_name = await resolve_model_name_for_client(client, state.tenant_id)

    # Build user prompt
    user_parts = [f"## Task\n{state.task_description}"]

    if state.repository_url:
        user_parts.append(f"\n## Repository\n{state.repository_url} (branch: {state.branch})")

    if state.root_cause_notes:
        # Set when nodes/tool_execution.py's root-cause step decided a
        # PREVIOUS plan itself was wrong (not just its execution) — this is
        # a re-plan, not a first attempt.
        notes = "\n".join(f"- {n}" for n in state.root_cause_notes)
        user_parts.append(
            f"\n## This Is A Re-Plan\nAn earlier plan for this task failed repeatedly. Root-cause analysis "
            f"found:\n{notes}\nProduce a genuinely different plan that avoids repeating these mistakes."
        )

    if state.target_files:
        user_parts.append("\n## Target Files\n" + "\n".join(f"- {f}" for f in state.target_files))

    if state.memory_context:
        user_parts.append(f"\n{state.memory_context}")

    if state.rag_context:
        user_parts.append(f"\n## Codebase Context (from vector search)\n{state.rag_context}")

    if state.context_files:
        for fname, content in state.context_files.items():
            user_parts.append(f"\n## Context File: {fname}\n```\n{content}\n```")

    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    try:
        response = await client.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            model_override=model_name,
        )

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        state.add_tokens(prompt_tokens, completion_tokens)

        # Parse the plan
        content = response["choices"][0]["message"]["content"]

        # Strip markdown fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan_data = json.loads(content.strip())

        state.plan = plan_data.get("plan_summary", "")
        state.plan_steps = [step["description"] for step in plan_data.get("steps", [])]
        state.current_plan_step = 0
        state.phase = AgentPhase.CODING

        # Store messages for continuity
        state.messages = [*messages, {"role": "assistant", "content": json.dumps(plan_data, indent=2)}]

        state.record_iteration(
            IterationRecord(
                iteration=state.iteration,
                phase="planning",
                model_role=state.primary_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )

        logger.info(
            "planning.complete",
            task_id=str(state.task_id),
            steps=len(state.plan_steps),
            tokens=prompt_tokens + completion_tokens,
        )

    except json.JSONDecodeError as exc:
        logger.error("planning.json_parse_error", error=str(exc))
        # Fall back to a simple single-step plan
        state.plan = state.task_description
        state.plan_steps = [state.task_description]
        state.current_plan_step = 0
        state.phase = AgentPhase.CODING

    except Exception as exc:
        logger.error("planning.failed", error=str(exc))
        state.phase = AgentPhase.FAILED
        state.error_message = f"Planning failed: {exc}"

    return state
