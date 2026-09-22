# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Minimal real HTTP server mimicking vLLM's OpenAI-compatible /chat/completions
API, for testing InferenceClient's real HTTP/retry/payload behavior against a
real server — not a mock of business logic, a real ASGI server with scripted
responses, run as a subprocess by tests/test_inference_client.py.
"""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
state = {"call_count": 0, "fail_with_429_times": 0, "last_payload": None, "last_headers": None}


@app.post("/reset")
async def reset(req: Request):
    body = await req.json()
    state["call_count"] = 0
    state["fail_with_429_times"] = body.get("fail_with_429_times", 0)
    state["last_payload"] = None
    state["last_headers"] = None
    return {"ok": True}


@app.get("/last_payload")
async def last_payload():
    return state["last_payload"] or {}


@app.get("/last_headers")
async def last_headers():
    """The real request headers the last /chat/completions call arrived
    with — lets a test prove a bearer token actually reached the server
    rather than trusting that the client set it."""
    return state["last_headers"] or {}


@app.post("/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    state["call_count"] += 1
    state["last_payload"] = payload
    state["last_headers"] = {k.lower(): v for k, v in req.headers.items()}

    if state["call_count"] <= state["fail_with_429_times"]:
        return JSONResponse(status_code=429, content={"error": "rate limited"})

    if payload.get("stream"):
        return StreamingResponse(_stream_chunks(payload), media_type="text/event-stream")

    if payload.get("tools"):
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": json.dumps({"path": "app.py"})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    if payload.get("response_format", {}).get("type") == "json_schema":
        # First call in a "bad JSON then good JSON" test scenario
        if state["call_count"] == 1 and state.get("simulate_bad_json_once"):
            content = "not valid json {{{"
        else:
            content = json.dumps({"plan_steps": ["step one", "step two"]})
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    return {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _stream_chunks(payload: dict):
    """The real chunk shapes vLLM emits: role first, then content deltas — or a tool call whose id/name
    arrive in one chunk and whose JSON arguments are split across later ones — then finish_reason, then
    (only with stream_options.include_usage) a usage-only chunk with empty choices, then [DONE]."""
    base = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "model": payload.get("model", "fake")}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    if payload.get("tools"):
        args = json.dumps({"path": "app.py"})
        head, tail = args[:5], args[5:]
        yield _sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_s1",
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": head},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
        yield _sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": tail}}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
        yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    else:
        for piece in ("hel", "lo"):
            yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if (payload.get("stream_options") or {}).get("include_usage"):
        yield _sse({**base, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    yield "data: [DONE]\n\n"


@app.post("/set_bad_json_once")
async def set_bad_json_once():
    state["simulate_bad_json_once"] = True
    return {"ok": True}
