# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
A real HTTP server speaking the pod endpoints of RunPod's REST API as the
published OpenAPI document describes them (tests/fixtures/runpod_openapi_pods.json,
a verbatim subset of https://rest.runpod.io/v1/openapi.json) plus the pod's
proxied status port — so src/finetuning/backends/runpod_pod.py's request
shapes and polling loop run against a server, not a stub of themselves.
Same pattern as fake_vllm_server.py: a scripted ASGI app run as a
subprocess by tests/test_runpod_pod_backend.py.

`POST /pods` rejects a body carrying a key the schema does not declare with
400 — the real API's response to a wrong field — which is what keeps the
payload builder honest. A scenario (`POST /scenario`) scripts what the
pod's status endpoint answers on each poll and when the pod itself changes
`desiredStatus`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

_SPEC = json.loads((Path(__file__).parent / "runpod_openapi_pods.json").read_text())
_POD_CREATE_KEYS = set(_SPEC["components"]["schemas"]["PodCreateInput"]["properties"])
_GPU_TYPE_IDS = set(_SPEC["components"]["schemas"]["PodCreateInput"]["properties"]["gpuTypeIds"]["items"]["enum"])
_CLOUD_TYPES = set(_SPEC["components"]["schemas"]["PodCreateInput"]["properties"]["cloudType"]["enum"])

app = FastAPI()
_ids = itertools.count(1)
state: dict = {
    "pods": {},  # pod_id -> Pod record
    "requests": [],  # (method, path, body) in order
    "status_script": [],  # answers of the pod's /status endpoint per poll; the last one repeats
    "status_polls": 0,
    "pod_status_after": None,  # {"polls": n, "desiredStatus": "EXITED"}: flip the pod on the n-th poll
    "create_error": None,  # {"status": 500, "body": "..."}: fail pod creation
    "shutdown_requests": 0,
}


@app.post("/scenario")
async def scenario(req: Request):
    body = await req.json()
    state["pods"] = {}
    state["requests"] = []
    state["status_script"] = body.get("status_script", [])
    state["status_polls"] = 0
    state["pod_status_after"] = body.get("pod_status_after")
    state["create_error"] = body.get("create_error")
    state["shutdown_requests"] = 0
    return {"ok": True}


@app.get("/recorded")
async def recorded():
    return {
        "requests": state["requests"],
        "pods": state["pods"],
        "status_polls": state["status_polls"],
        "shutdown_requests": state["shutdown_requests"],
    }


# ── RunPod REST: pods ───────────────────────────────────────────


@app.post("/pods", status_code=201)
async def create_pod(req: Request):
    body = await req.json()
    state["requests"].append(("POST", "/pods", body))
    unknown = sorted(set(body) - _POD_CREATE_KEYS)
    if unknown:
        return JSONResponse(status_code=400, content={"error": f"unknown field(s) in PodCreateInput: {unknown}"})
    bad_gpus = sorted(set(body.get("gpuTypeIds", [])) - _GPU_TYPE_IDS)
    if bad_gpus:
        return JSONResponse(status_code=400, content={"error": f"gpuTypeIds not in the enum: {bad_gpus}"})
    if body.get("cloudType") not in _CLOUD_TYPES:
        return JSONResponse(status_code=400, content={"error": f"cloudType not in the enum: {body.get('cloudType')}"})
    if state["create_error"]:
        return Response(status_code=state["create_error"]["status"], content=state["create_error"]["body"])
    pod_id = f"pod{next(_ids):04d}"
    pod = {
        "id": pod_id,
        "name": body.get("name"),
        "image": body.get("imageName"),
        "desiredStatus": "RUNNING",
        "costPerHr": 0.44,
        "gpuCount": body.get("gpuCount", 1),
        "env": body.get("env", {}),
        "ports": body.get("ports", []),
        "networkVolume": {"id": body.get("networkVolumeId")} if body.get("networkVolumeId") else None,
        "volumeMountPath": body.get("volumeMountPath"),
        "dockerStartCmd": body.get("dockerStartCmd"),
    }
    state["pods"][pod_id] = pod
    return pod


@app.get("/pods/{pod_id}")
async def get_pod(pod_id: str):
    state["requests"].append(("GET", f"/pods/{pod_id}", None))
    pod = state["pods"].get(pod_id)
    if pod is None:
        return JSONResponse(status_code=404, content={"error": "pod not found"})
    flip = state["pod_status_after"]
    if flip and state["status_polls"] >= flip["polls"]:
        pod["desiredStatus"] = flip["desiredStatus"]
    return pod


@app.post("/pods/{pod_id}/stop")
async def stop_pod(pod_id: str):
    state["requests"].append(("POST", f"/pods/{pod_id}/stop", None))
    pod = state["pods"].get(pod_id)
    if pod is None:
        return JSONResponse(status_code=404, content={"error": "pod not found"})
    pod["desiredStatus"] = "EXITED"
    return {}


@app.delete("/pods/{pod_id}", status_code=204)
async def delete_pod(pod_id: str):
    state["requests"].append(("DELETE", f"/pods/{pod_id}", None))
    if pod_id not in state["pods"]:
        return JSONResponse(status_code=404, content={"error": "pod not found"})
    del state["pods"][pod_id]
    return Response(status_code=204)


# ── the pod's own status port, as RunPod's proxy would expose it ─


@app.get("/proxy/{pod_id}/status")
async def pod_status(pod_id: str):
    state["requests"].append(("GET", f"/proxy/{pod_id}/status", None))
    script = state["status_script"]
    if pod_id not in state["pods"] or not script:
        return JSONResponse(status_code=502, content={"error": "nothing listening yet"})
    index = min(state["status_polls"], len(script) - 1)
    state["status_polls"] += 1
    answer = script[index]
    if answer is None:  # the container is still pulling the image: the proxy has nothing to reach
        return JSONResponse(status_code=502, content={"error": "nothing listening yet"})
    return answer


@app.post("/proxy/{pod_id}/shutdown")
async def pod_shutdown(pod_id: str):
    state["requests"].append(("POST", f"/proxy/{pod_id}/shutdown", None))
    state["shutdown_requests"] += 1
    return {"ok": True}
