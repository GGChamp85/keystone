# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""The pure Anthropic <-> OpenAI translation (src/inference/anthropic_compat.py), both directions and the
stream translator, on the exact chunk shapes vLLM emits (the same ones tests/fixtures/fake_vllm_server.py
serves)."""

from __future__ import annotations

import json

import pytest

from src.inference.anthropic_compat import (
    AnthropicStreamTranslator,
    UnsupportedContentError,
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
)


def test_system_text_tool_use_and_tool_result_round_trip_into_openai_shapes():
    out = anthropic_messages_to_openai(
        [{"type": "text", "text": "Be terse."}],
        [
            {"role": "user", "content": "Read app.py"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "app.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "print('hi')", "is_error": False},
                    {"type": "text", "text": "Now fix it."},
                ],
            },
        ],
    )
    assert out[0] == {"role": "system", "content": "Be terse."}
    assert out[1] == {"role": "user", "content": "Read app.py"}
    assert out[2]["role"] == "assistant" and out[2]["content"] == "Reading."
    assert out[2]["tool_calls"] == [
        {"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "app.py"}'}}
    ]
    # the tool result precedes the user's text, as OpenAI requires
    assert out[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "print('hi')"}
    assert out[4] == {"role": "user", "content": "Now fix it."}


def test_error_tool_results_are_marked_and_images_are_refused_not_dropped():
    out = anthropic_messages_to_openai(
        None,
        [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": True}],
            }
        ],
    )
    assert out[0]["content"].startswith("[tool error] boom")
    with pytest.raises(UnsupportedContentError, match="image"):
        anthropic_messages_to_openai(
            None,
            [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "..."}}]}],
        )
    with pytest.raises(UnsupportedContentError, match="role"):
        anthropic_messages_to_openai(None, [{"role": "system", "content": "x"}])


def test_tools_and_tool_choice_translate_to_openai_function_calling():
    tools = anthropic_tools_to_openai(
        [{"name": "grep", "description": "search", "input_schema": {"type": "object", "properties": {"q": {}}}}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "search",
                "parameters": {"type": "object", "properties": {"q": {}}},
            },
        }
    ]
    assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "grep"}) == {
        "type": "function",
        "function": {"name": "grep"},
    }
    assert anthropic_tool_choice_to_openai(None) is None
    with pytest.raises(UnsupportedContentError):
        anthropic_tools_to_openai([{"type": "web_search_20250305", "name": "web_search"}])


def test_openai_response_becomes_an_anthropic_message_with_tool_use_blocks():
    msg = openai_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me look.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        "coding",
    )
    assert msg["type"] == "message" and msg["role"] == "assistant" and msg["model"] == "coding"
    assert msg["id"].startswith("msg_")
    assert msg["content"] == [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a"}},
    ]
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {"input_tokens": 10, "output_tokens": 5}

    plain = openai_response_to_anthropic(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}], "usage": {}}, "coding"
    )
    assert plain["stop_reason"] == "max_tokens" and plain["content"] == [{"type": "text", "text": "hi"}]


def _events(frames: list[str]) -> list[tuple[str, dict]]:
    out = []
    for frame in frames:
        event_line, data_line = frame.strip().split("\n")
        assert event_line.startswith("event: ") and data_line.startswith("data: ")
        payload = json.loads(data_line[len("data: ") :])
        assert payload["type"] == event_line[len("event: ") :]
        out.append((payload["type"], payload))
    return out


def test_stream_translator_emits_the_anthropic_event_sequence_for_text():
    t = AnthropicStreamTranslator("coding", estimate_input_tokens=lambda: 7)
    frames: list[str] = []
    frames += t.feed({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    frames += t.feed({"choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}]})
    frames += t.feed({"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]})
    frames += t.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    frames += t.feed({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    frames += t.finish()
    events = _events(frames)
    assert [e for e, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["usage"] == {"input_tokens": 7, "output_tokens": 0} and start["model"] == "coding"
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert [e[1]["delta"]["text"] for e in events[2:4]] == ["hel", "lo"]
    delta = events[5][1]
    assert delta["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert delta["usage"] == {"input_tokens": 10, "output_tokens": 5}  # the backend's real numbers


def test_stream_translator_turns_split_tool_arguments_into_input_json_deltas():
    """The fake vLLM server's shape: id/name and the head of the JSON in one chunk, the tail in another."""
    t = AnthropicStreamTranslator("coding", estimate_input_tokens=lambda: 3)
    args = json.dumps({"path": "app.py"})
    frames: list[str] = []
    frames += t.feed(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_s1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": args[:5]},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    frames += t.feed(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[5:]}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    frames += t.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    frames += t.finish(fallback_usage=lambda: {"input_tokens": 3, "output_tokens": 4})
    events = _events(frames)
    assert [e for e, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "tool_use", "id": "call_s1", "name": "read_file", "input": {}}
    partial = "".join(e[1]["delta"]["partial_json"] for e in events[2:4])
    assert all(e[1]["delta"]["type"] == "input_json_delta" for e in events[2:4])
    assert json.loads(partial) == {"path": "app.py"}
    assert events[5][1]["delta"]["stop_reason"] == "tool_use"
    assert events[5][1]["usage"] == {"input_tokens": 3, "output_tokens": 4}  # the tokenizer fallback was used


def test_stream_translator_closes_a_text_block_before_a_tool_block_and_pads_empty_input():
    t = AnthropicStreamTranslator("coding", estimate_input_tokens=lambda: 1)
    frames: list[str] = []
    frames += t.feed({"choices": [{"index": 0, "delta": {"content": "Sure."}, "finish_reason": None}]})
    frames += t.feed(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "list_dir"}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    frames += t.finish()
    kinds = [e for e, _ in _events(frames)]
    assert kinds == [
        "message_start",
        "content_block_start",  # text
        "content_block_delta",
        "content_block_stop",
        "content_block_start",  # tool_use, index 1
        "content_block_delta",  # the "{}" padding so the SDK parses valid JSON
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    events = _events(frames)
    assert events[4][1]["index"] == 1
    assert events[5][1]["delta"] == {"type": "input_json_delta", "partial_json": "{}"}
