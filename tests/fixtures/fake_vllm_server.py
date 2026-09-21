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
from fastapi.responses import JSONResponse

app = FastAPI()
state = {"call_count": 0, "fail_with_429_times": 0, "last_payload": None}


@app.post("/reset")
async def reset(req: Request):
    body = await req.json()
    state["call_count"] = 0
    state["fail_with_429_times"] = body.get("fail_with_429_times", 0)
    state["last_payload"] = None
    return {"ok": True}


@app.get("/last_payload")
async def last_payload():
    return state["last_payload"] or {}


@app.post("/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    state["call_count"] += 1
    state["last_payload"] = payload

    if state["call_count"] <= state["fail_with_429_times"]:
        return JSONResponse(status_code=429, content={"error": "rate limited"})

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


@app.post("/set_bad_json_once")
async def set_bad_json_once():
    state["simulate_bad_json_once"] = True
    return {"ok": True}
