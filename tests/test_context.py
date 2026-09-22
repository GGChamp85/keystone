# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real unit tests for src/orchestrator/context.py — real tiktoken encoding (no mocking of the tokenizer), pure
trimming logic.
"""

from __future__ import annotations

from src.orchestrator.context import count_message_tokens, count_tokens, trim_turns_to_budget


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_matches_real_tiktoken_encoding():
    import tiktoken

    text = "def greet(name: str) -> str:\n    return f'hello {name}'"
    encoding = tiktoken.get_encoding("cl100k_base")
    assert count_tokens(text) == len(encoding.encode(text))


def test_count_tokens_longer_text_counts_more_tokens():
    short = count_tokens("hello")
    long = count_tokens("hello " * 500)
    assert long > short * 100


def test_count_message_tokens_includes_tool_call_arguments():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ],
    }
    bare = count_message_tokens({"role": "assistant", "content": None})
    with_call = count_message_tokens(message)
    assert with_call > bare


def test_trim_turns_no_trimming_needed_returns_everything_flattened():
    turns = [[{"role": "system", "content": "sys"}], [{"role": "user", "content": "hi"}]]
    result = trim_turns_to_budget(turns, max_tokens=10_000)
    assert result == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def _big_turn(i: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "grep", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": "x " * 3000},
    ]


def test_trim_turns_drops_oldest_turns_first_keeping_head_and_latest():
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    turns = [head, _big_turn(1), _big_turn(2), _big_turn(3), _big_turn(4)]

    result = trim_turns_to_budget(turns, max_tokens=3000, keep_head_turns=1)

    # Head always present.
    assert result[0] == {"role": "system", "content": "sys"}
    assert result[1] == {"role": "user", "content": "task"}
    # The oldest turn(s) were dropped — turn 1's tool_call_id should not appear.
    tool_call_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
    assert "c1" not in tool_call_ids
    # The most recent turn always survives.
    assert "c4" in tool_call_ids
    # A drop-notice message was inserted.
    assert any("dropped" in (m.get("content") or "") for m in result if m["role"] == "user")


def test_trim_turns_always_keeps_at_least_the_most_recent_turn_even_if_over_budget():
    head = [{"role": "system", "content": "sys"}]
    huge_turn = [{"role": "assistant", "content": "x" * 50_000}]
    turns = [head, [{"role": "user", "content": "hi"}], huge_turn]

    result = trim_turns_to_budget(turns, max_tokens=10, keep_head_turns=1)

    assert result[-1] == huge_turn[0]


def test_trim_turns_never_splits_an_assistant_tool_calls_message_from_its_tool_results():
    """A turn that survives trimming must keep its assistant+tool-result messages together — dropping only
    the tool results (or only the assistant message) would produce a malformed request against a strict
    OpenAI-compatible backend."""
    head = [{"role": "system", "content": "sys"}]
    turn = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    turns = [head, turn]

    result = trim_turns_to_budget(turns, max_tokens=100_000, keep_head_turns=1)

    assert result == [*head, *turn]


# ── summarise-on-overflow (trim_turns_to_budget_async) ────────


def _big_turns(n: int) -> list[list[dict]]:
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    return [head] + [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            },
            # ~300 real tokens per turn (distinct words — a run of one character tokenises to almost nothing)
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": f"turn {i} " + " ".join(f"w{i}_{j}" for j in range(150)),
            },
        ]
        for i in range(n)
    ]


async def test_async_trim_replaces_dropped_turns_with_the_summarizers_text():
    from src.orchestrator.context import trim_turns_to_budget_async

    seen: list[list[dict]] = []

    async def summarizer(messages):
        seen.append(messages)
        return "Read app.py (greet at line 1); patched app.py; tests passed."

    messages = await trim_turns_to_budget_async(_big_turns(6), max_tokens=500, keep_head_turns=1, summarizer=summarizer)
    assert messages[0]["role"] == "system" and messages[1]["content"] == "task"  # head kept
    notice = messages[2]
    assert notice["role"] == "user"
    assert notice["content"].startswith("[Context summary of ")
    assert "Read app.py (greet at line 1)" in notice["content"]
    assert seen and all(m["role"] in ("assistant", "tool") for m in seen[0])  # only the dropped turns were summarised
    assert messages[-1]["content"].startswith("turn 5")  # the most recent turn survives


async def test_async_trim_falls_back_to_the_plain_notice_when_the_summarizer_fails_or_is_empty():
    from src.orchestrator.context import trim_turns_to_budget_async

    async def broken(messages):
        raise RuntimeError("model down")

    async def empty(messages):
        return "   "

    for summarizer in (broken, empty, None):
        messages = await trim_turns_to_budget_async(_big_turns(6), max_tokens=500, summarizer=summarizer)
        assert "were dropped to stay within the context budget" in messages[2]["content"]


async def test_async_trim_is_a_no_op_within_budget_and_never_calls_the_summarizer():
    from src.orchestrator.context import trim_turns_to_budget_async

    async def must_not_run(messages):
        raise AssertionError("summarizer called within budget")

    turns = _big_turns(2)
    messages = await trim_turns_to_budget_async(turns, max_tokens=100_000, summarizer=must_not_run)
    assert messages == [m for t in turns for m in t]
