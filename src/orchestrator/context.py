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

from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_ENCODING = "cl100k_base"
_FALLBACK_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD_TOKENS = 4  # OpenAI's own rule-of-thumb per-message framing overhead
# No tokenizer here can know a vision backend's real image-patch token cost ahead of time;
# 1000 is a deliberately conservative flat estimate (OpenAI's own high-detail images commonly
# cost several hundred to ~1500 tokens) so the budget never under-counts an attached image.
_PER_IMAGE_TOKEN_ESTIMATE = 1000


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


def count_content_tokens(content: str | list[dict[str, Any]] | None, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Token count for a message's `content`, which is either a plain string or an OpenAI-style
    list of content parts (text + image_url) once a task has an attached image — see
    `nodes/coding.py::_build_agentic_user_context`."""
    if not content:
        return 0
    if isinstance(content, str):
        return count_tokens(content, encoding_name)
    total = 0
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            total += count_tokens(part.get("text", ""), encoding_name)
        elif part_type == "image_url":
            total += _PER_IMAGE_TOKEN_ESTIMATE
    return total


def count_message_tokens(message: dict[str, Any], encoding_name: str = DEFAULT_ENCODING) -> int:
    total = _PER_MESSAGE_OVERHEAD_TOKENS
    total += count_content_tokens(message.get("content"), encoding_name)
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        total += count_tokens(fn.get("name", ""), encoding_name)
        arguments = fn.get("arguments", "")
        total += count_tokens(arguments if isinstance(arguments, str) else str(arguments), encoding_name)
    return total


def count_messages_tokens(messages: list[dict[str, Any]], encoding_name: str = DEFAULT_ENCODING) -> int:
    return sum(count_message_tokens(m, encoding_name) for m in messages)


Turns = list[list[dict[str, Any]]]
# Summarises the messages of the turns being dropped into a short text the model keeps
# instead of them. Async because it is a real model call (nodes/coding.py wires one up).
Summarizer = Callable[[list[dict[str, Any]]], Awaitable[str]]


def split_turns_for_budget(
    turns: Turns, *, max_tokens: int, keep_head_turns: int = 1, encoding_name: str = DEFAULT_ENCODING
) -> tuple[Turns, Turns, Turns]:
    """
    (head, dropped, tail): the first `keep_head_turns` turns are always kept
    (system + task context), the oldest of the rest are moved to `dropped`
    until what remains fits `max_tokens`. At least the most recent turn is
    always kept even if it alone exceeds the budget — the request may still
    fail against the real model in that case, but the loop has something
    coherent to retry from rather than an empty conversation.
    """
    if len(turns) <= keep_head_turns + 1:
        return list(turns), [], []
    head, tail = list(turns[:keep_head_turns]), list(turns[keep_head_turns:])
    total = sum(count_message_tokens(m, encoding_name) for turn in turns for m in turn)
    dropped: Turns = []
    while len(tail) > 1 and total > max_tokens:
        removed = tail.pop(0)
        total -= sum(count_message_tokens(m, encoding_name) for m in removed)
        dropped.append(removed)
    return head, dropped, tail


def _drop_notice(dropped_turns: int) -> str:
    return (
        f"[{dropped_turns} earlier tool-call turn(s) were dropped to stay within the "
        "context budget — re-read a file if you need to see its content again.]"
    )


def _assemble(head: Turns, notice: str | None, tail: Turns) -> list[dict[str, Any]]:
    flat = [message for turn in head for message in turn]
    if notice:
        flat.append({"role": "user", "content": notice})
    flat.extend(message for turn in tail for message in turn)
    return flat


def trim_turns_to_budget(
    turns: Turns,
    *,
    max_tokens: int,
    keep_head_turns: int = 1,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    """
    Flattens `turns` (each a list of one or more messages that belong
    together — e.g. one assistant tool-calls message plus its tool
    results) into a single message list for `InferenceClient.complete()`,
    dropping the oldest turns after the first `keep_head_turns` until the
    total fits `max_tokens` (see `split_turns_for_budget`), and leaving a
    one-line notice where they were.
    """
    head, dropped, tail = split_turns_for_budget(
        turns, max_tokens=max_tokens, keep_head_turns=keep_head_turns, encoding_name=encoding_name
    )
    if not dropped:
        return [message for turn in turns for message in turn]
    return _assemble(head, _drop_notice(len(dropped)), tail)


async def trim_turns_to_budget_async(
    turns: Turns,
    *,
    max_tokens: int,
    keep_head_turns: int = 1,
    summarizer: Summarizer | None = None,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    """
    `trim_turns_to_budget`, but the dropped turns are summarised by
    `summarizer` (a real model call) into a "Context summary" message the
    model keeps — what it learned from files it read, edits it already
    applied, results it saw — instead of a bare "N turns were dropped".
    Falls back to the plain notice if the summariser fails or returns
    nothing; a summary is never allowed to be the reason a task fails.
    """
    head, dropped, tail = split_turns_for_budget(
        turns, max_tokens=max_tokens, keep_head_turns=keep_head_turns, encoding_name=encoding_name
    )
    if not dropped:
        return [message for turn in turns for message in turn]
    notice = _drop_notice(len(dropped))
    if summarizer is not None:
        try:
            summary = (await summarizer([m for turn in dropped for m in turn])).strip()
        except Exception as exc:
            logger.warning("context.summarize_failed", dropped_turns=len(dropped), error=str(exc))
            summary = ""
        if summary:
            notice = (
                f"[Context summary of {len(dropped)} earlier tool-call turn(s), removed to stay within the "
                f"context budget]\n{summary}\n[Re-read a file if you need its exact current content.]"
            )
    return _assemble(head, notice, tail)


def fit_to_tokens(text: str, max_tokens: int, *, keep: str = "head", what: str = "content") -> str:
    """
    The only truncation in the agent, and only for what goes INTO one prompt:
    keep `text` whole when it fits `max_tokens` (tiktoken-counted), otherwise
    keep the head (files, diffs) or the tail (test output — failures and the
    summary line come last) and say exactly how much was omitted and why.
    Nothing persisted is ever cut with this; the full text stays on the
    state, in the trace, and in the sandbox for the model to read by range.
    """
    if max_tokens <= 0 or count_tokens(text) <= max_tokens:
        return text
    encoding = _get_encoding_or_none()
    if encoding is None:
        budget_chars = max_tokens * _FALLBACK_CHARS_PER_TOKEN
        kept = text[:budget_chars] if keep == "head" else text[-budget_chars:]
        omitted = len(text) - budget_chars
        unit = "characters"
    else:
        tokens = encoding.encode(text)
        kept_tokens = tokens[:max_tokens] if keep == "head" else tokens[-max_tokens:]
        kept = encoding.decode(kept_tokens)
        omitted = len(tokens) - max_tokens
        unit = "tokens"
    note = (
        f"\n… [{omitted} more {unit} of {what} omitted to fit the model context; the full text is kept in the task "
        "record — use read_file with a line range, grep, or the trace to see the rest]"
    )
    return (
        kept + note
        if keep == "head"
        else f"[… first {omitted} {unit} of {what} omitted to fit the model context]\n" + kept
    )


def _get_encoding_or_none() -> Any:
    try:
        return _get_encoding()
    except Exception:
        return None
