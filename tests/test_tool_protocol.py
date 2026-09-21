# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real unit tests for src/orchestrator/tools/protocol.py — pure parsing/formatting logic, no infra needed. Exercises
both the well-formed path and the malformed-model-output path (bad JSON in a native tool call's arguments, an
unparseable fenced block in text mode) since a real model will produce both.
"""

from __future__ import annotations

from src.orchestrator.tools.protocol import (
    NativeToolProtocol,
    TextToolProtocol,
    ToolCall,
    ToolCallError,
    get_tool_protocol,
)


def test_native_protocol_parses_well_formed_tool_calls():
    proto = NativeToolProtocol()
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ],
    }
    calls = proto.parse_tool_calls(message)
    assert calls == [ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})]


def test_native_protocol_no_tool_calls_returns_empty_list():
    proto = NativeToolProtocol()
    message = {"role": "assistant", "content": "All done."}
    assert proto.parse_tool_calls(message) == []


def test_native_protocol_bad_json_arguments_becomes_tool_call_error_not_a_crash():
    proto = NativeToolProtocol()
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": "{not json"}}],
    }
    calls = proto.parse_tool_calls(message)
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallError)
    assert calls[0].name == "grep"
    assert "not valid JSON" in calls[0].error


def test_native_protocol_request_kwargs_includes_tools_and_auto_choice():
    proto = NativeToolProtocol()
    kwargs = proto.request_kwargs()
    assert kwargs["tool_choice"] == "auto"
    assert any(t["function"]["name"] == "apply_patch" for t in kwargs["tools"])


def test_native_protocol_format_tool_result_carries_call_id():
    proto = NativeToolProtocol()
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    result = proto.format_tool_result(call, "file contents here")
    assert result == [{"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file contents here"}]


def test_text_protocol_system_prompt_lists_all_tools():
    proto = TextToolProtocol()
    suffix = proto.system_prompt_suffix()
    for name in ("read_file", "list_dir", "grep", "apply_patch", "run_command"):
        assert name in suffix


def test_text_protocol_parses_fenced_tool_call_block():
    proto = TextToolProtocol()
    message = {"role": "assistant", "content": '```tool_call\n{"name": "list_dir", "arguments": {"path": "."}}\n```'}
    calls = proto.parse_tool_calls(message)
    assert calls == [ToolCall(id="text-call-0", name="list_dir", arguments={"path": "."})]


def test_text_protocol_no_fenced_block_returns_empty_list():
    proto = TextToolProtocol()
    message = {"role": "assistant", "content": "Here is my final answer, no tool needed."}
    assert proto.parse_tool_calls(message) == []


def test_text_protocol_malformed_json_in_fence_becomes_tool_call_error():
    proto = TextToolProtocol()
    message = {"role": "assistant", "content": "```tool_call\n{not valid json}\n```"}
    calls = proto.parse_tool_calls(message)
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallError)
    assert "not valid JSON" in calls[0].error


def test_text_protocol_missing_name_becomes_tool_call_error():
    proto = TextToolProtocol()
    message = {"role": "assistant", "content": '```tool_call\n{"arguments": {}}\n```'}
    calls = proto.parse_tool_calls(message)
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallError)
    assert "missing `name`" in calls[0].error


def test_get_tool_protocol_rejects_unknown_protocol_name():
    import pytest

    with pytest.raises(ValueError, match="Unknown tool protocol"):
        get_tool_protocol("carrier-pigeon")


def test_get_tool_protocol_returns_matching_implementation():
    assert isinstance(get_tool_protocol("native"), NativeToolProtocol)
    assert isinstance(get_tool_protocol("text"), TextToolProtocol)
