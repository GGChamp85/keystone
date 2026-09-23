# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
A real HTTP server speaking Ray's Jobs REST API on the dashboard — the
endpoints `ray job submit` itself uses: `POST /api/jobs/`,
`GET /api/jobs/{id}`, `GET /api/jobs/{id}/logs`, `POST /api/jobs/{id}/stop`
— with scripted status progressions, so src/finetuning/backends/ray_train.py's
submission and polling loop run against a server rather than a stub.
Run as a subprocess by tests/test_ray_train_backend.py.
"""

from __future__ import annotations

import itertools

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
_ids = itertools.count(1)
state: dict = {
    "jobs": {},  # submission_id -> {"spec": ..., "polls": 0, "stopped": False}
    "statuses": ["PENDING", "RUNNING", "SUCCEEDED"],  # what each successive GET /api/jobs/{id} answers
    "logs_by_poll": [],  # cumulative log text per poll; the last entry repeats
    "message": None,
    "submit_error": None,  # {"status": 400, "body": "..."}
}


@app.post("/scenario")
async def scenario(req: Request):
    body = await req.json()
    state["jobs"] = {}
    state["statuses"] = body.get("statuses", ["PENDING", "RUNNING", "SUCCEEDED"])
    state["logs_by_poll"] = body.get("logs_by_poll", [])
    state["message"] = body.get("message")
    state["submit_error"] = body.get("submit_error")
    return {"ok": True}


@app.get("/recorded")
async def recorded():
    return state["jobs"]


@app.post("/api/jobs/")
async def submit(req: Request):
    spec = await req.json()
    if state["submit_error"]:
        return JSONResponse(
            status_code=state["submit_error"]["status"], content={"detail": state["submit_error"]["body"]}
        )
    submission_id = spec.get("submission_id") or f"raysubmit_{next(_ids)}"
    if submission_id in state["jobs"]:
        # Ray's real answer to a duplicate submission_id.
        return JSONResponse(
            status_code=400, content={"detail": f"Job with submission_id {submission_id} already exists"}
        )
    state["jobs"][submission_id] = {"spec": spec, "polls": 0, "stopped": False}
    return {"submission_id": submission_id}


@app.get("/api/jobs/{submission_id}")
async def status(submission_id: str):
    job = state["jobs"].get(submission_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    index = min(job["polls"], len(state["statuses"]) - 1)
    job["polls"] += 1
    return {
        "type": "SUBMISSION",
        "job_id": None,
        "submission_id": submission_id,
        "status": "STOPPED" if job["stopped"] else state["statuses"][index],
        "entrypoint": job["spec"].get("entrypoint"),
        "message": state["message"],
        "metadata": job["spec"].get("metadata", {}),
        "runtime_env": job["spec"].get("runtime_env", {}),
    }


@app.get("/api/jobs/{submission_id}/logs")
async def logs(submission_id: str):
    job = state["jobs"].get(submission_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    if not state["logs_by_poll"]:
        return {"logs": ""}
    index = min(max(job["polls"] - 1, 0), len(state["logs_by_poll"]) - 1)
    return {"logs": state["logs_by_poll"][index]}


@app.post("/api/jobs/{submission_id}/stop")
async def stop(submission_id: str):
    job = state["jobs"].get(submission_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    job["stopped"] = True
    return {"stopped": True}
