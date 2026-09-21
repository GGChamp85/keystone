# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — token-budgeted prompt assembly.

The agentic tool loop (`nodes/coding.py`) can run up to `max_tool_steps`
turns, each appending an assistant message plus its tool results to the
conversation — with no trimming, that grows unboundedly and eventually
either blows past the model's context window or silently burns an
enormous amount of tokens on stale tool output the model doesn't need
anymore. `trim_turns_to_budget` keeps the conversation within a real token
budget by dropping the OLDEST whole turns first, never a partial turn —
splitting an assistant message with `tool_calls` from its matching
`role: tool` results would produce a malformed request against a
strict OpenAI-compatible backend, so trimming always operates on whole
turns, not individual messages.

Real token counting via `tiktoken` (cached per-encoding), not a len//4
guess — falling back to that guess only if tiktoken's encoding data isn't
available (e.g. a fully air-gapped box with no cached encoding file), so
the loop degrades gracefully instead of crashing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_ENCODING = "cl100k_base"
_FALLBACK_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD_TOKENS = 4  # OpenAI's own rule-of-thumb per-message framing overhead


@lru_cache(maxsize=4)
def _get_encoding(name: str = DEFAULT_ENCODING) -> Any:
    import tiktoken

    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    if not text:
        return 0
    try:
        encoding = _get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as exc:
        logger.warning("context.tiktoken_unavailable", error=str(exc))
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)


def count_message_tokens(message: dict[str, Any], encoding_name: str = DEFAULT_ENCODING) -> int:
    total = _PER_MESSAGE_OVERHEAD_TOKENS
    content = message.get("content")
    if content:
        total += count_tokens(content, encoding_name)
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        total += count_tokens(fn.get("name", ""), encoding_name)
        arguments = fn.get("arguments", "")
        total += count_tokens(arguments if isinstance(arguments, str) else str(arguments), encoding_name)
    return total


def count_messages_tokens(messages: list[dict[str, Any]], encoding_name: str = DEFAULT_ENCODING) -> int:
    return sum(count_message_tokens(m, encoding_name) for m in messages)


def trim_turns_to_budget(
    turns: list[list[dict[str, Any]]],
    *,
    max_tokens: int,
    keep_head_turns: int = 1,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    """
    Flattens `turns` (each a list of one or more messages that belong
    together — e.g. one assistant tool-calls message plus its tool
    results) into a single message list for `InferenceClient.complete()`,
    dropping the oldest turns after the first `keep_head_turns` (the
    system + initial task-context turn, always kept) until the total fits
    `max_tokens`. Always keeps at least the most recent turn, even if it
    alone exceeds the budget — the request may still fail against the
    real model in that case, but the loop has something coherent to
    retry from rather than an empty conversation.
    """
    if len(turns) <= keep_head_turns + 1:
        return [message for turn in turns for message in turn]

    head, tail = turns[:keep_head_turns], list(turns[keep_head_turns:])
    total = sum(count_message_tokens(m, encoding_name) for turn in turns for m in turn)

    dropped_turns = 0
    while len(tail) > 1 and total > max_tokens:
        removed = tail.pop(0)
        total -= sum(count_message_tokens(m, encoding_name) for m in removed)
        dropped_turns += 1

    flat = [message for turn in head for message in turn]
    if dropped_turns:
        flat.append(
            {
                "role": "user",
                "content": f"[{dropped_turns} earlier tool-call turn(s) were dropped to stay within the "
                "context budget — re-read a file if you need to see its content again.]",
            }
        )
    flat.extend(message for turn in tail for message in turn)
    return flat
