# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
The process a RunPod training pod runs (`python -m src.finetuning.backends.pod_entrypoint`).

It reads the job from the environment the control plane put on the pod
(src/finetuning/backends/runpod_pod.py's `build_pod_payload`), trains
with the same in-process trainer every backend uses, writes the adapter to
the network-volume path, and — because RunPod's REST API exposes a pod's
status but not its stdout — serves its own status over HTTP on
`KEYSTONE_POD_STATUS_PORT` for the control plane to poll through RunPod's
proxy: live progress while training, the final metrics after, or the error.

After completion it keeps answering until the control plane has read the
result and asked it to shut down (`POST /shutdown`), bounded by
`KEYSTONE_POD_LINGER_SECONDS` so a control plane that died cannot leave a
GPU billing forever. stdlib only, so the training image needs nothing extra.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

JOB_ENV = "KEYSTONE_FINETUNE_JOB"
OUTPUT_DIR_ENV = "KEYSTONE_FINETUNE_OUTPUT_DIR"
STATUS_PORT_ENV = "KEYSTONE_POD_STATUS_PORT"
LINGER_ENV = "KEYSTONE_POD_LINGER_SECONDS"
DEFAULT_LINGER_SECONDS = 3600  # a safety bound on idle GPU billing after the result is ready, not a policy limit

METRICS_FILE = "metrics.json"


class PodStatus:
    """The one piece of shared state: what the status endpoint reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "starting"  # starting | running | completed | failed
        self.progress: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.error: str | None = None
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.shutdown = threading.Event()

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)
            self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "progress": dict(self.progress),
                "metrics": dict(self.metrics),
                "error": self.error,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
            }


def make_handler(status: PodStatus) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/status", "/"):
                self._json(200, status.snapshot())
            elif self.path == "/health":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/shutdown":
                status.shutdown.set()
                self._json(200, {"ok": True, "state": status.snapshot()["state"]})
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return  # polling every few seconds would otherwise fill the pod log

    return Handler


def serve_status(status: PodStatus, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(status))
    thread = threading.Thread(target=server.serve_forever, name="keystone-pod-status", daemon=True)
    thread.start()
    return server


def load_job_from_env(environ: dict[str, str]) -> dict[str, Any]:
    raw = environ.get(JOB_ENV)
    if not raw:
        raise SystemExit(f"{JOB_ENV} is not set — this process expects the job the control plane put on the pod")
    job = json.loads(raw)
    for key in ("id", "job_type", "base_model"):
        if key not in job:
            raise SystemExit(f"{JOB_ENV} is missing {key!r}")
    job.setdefault("config", {})
    return job


def train(status: PodStatus, job: dict[str, Any], output_dir: str) -> int:
    """Runs the trainer, records the outcome on `status` and in `<output_dir>/metrics.json`. Returns the
    process exit code."""
    from src.finetuning.backends.inprocess import run_training

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    status.update(state="running")

    def progress(payload: dict[str, Any]) -> None:
        status.update(progress=payload)

    try:
        metrics = run_training(job, progress=progress, output_dir=output_dir)
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    Path(output_dir, METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=str))
    status.update(state="completed", metrics=metrics)
    return 0


def linger(status: PodStatus, seconds: int) -> None:
    """Wait for the control plane's shutdown request, at most `seconds` (0 = until asked)."""
    if seconds <= 0:
        status.shutdown.wait()
    else:
        status.shutdown.wait(timeout=seconds)


def main(environ: dict[str, str] | None = None) -> int:
    env = dict(os.environ if environ is None else environ)
    job = load_job_from_env(env)
    output_dir = env.get(OUTPUT_DIR_ENV) or f"/runpod-volume/adapters/{job['id']}"
    port = int(env.get(STATUS_PORT_ENV) or 8000)
    linger_seconds = int(env.get(LINGER_ENV) or DEFAULT_LINGER_SECONDS)

    status = PodStatus()
    server = serve_status(status, port)
    try:
        code = train(status, job, output_dir)
        linger(status, linger_seconds)
    finally:
        server.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
