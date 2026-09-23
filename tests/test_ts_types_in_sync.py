# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""scripts/check_ts_types_in_sync.py: the task event types in web/src/api.ts and
vscode/extension/src/api.ts are identical today, and the check really fails when
they are not — a field added to one, a union member removed from the other, a
declaration missing altogether."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import check_ts_types_in_sync as sync

ROOT = Path(__file__).resolve().parent.parent


def test_the_two_clients_declare_identical_event_types():
    assert sync.compare() == []


def test_the_script_exits_zero_on_the_real_files():
    proc = subprocess.run(
        [sys.executable, "scripts/check_ts_types_in_sync.py"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "identical" in proc.stdout


def test_extract_declaration_handles_interfaces_and_multiline_type_aliases():
    web = sync.WEB_API.read_text()
    node = sync.extract_declaration(web, "NodeEvent")
    assert node.startswith("export interface NodeEvent {") and node.endswith("}")
    assert "pr_number?: number | null" in node
    step_type = sync.extract_declaration(web, "StepEventType")
    assert step_type.startswith("export type StepEventType =")
    assert "| 'pr'" in step_type and "export interface StepEvent" not in step_type
    task = sync.extract_declaration(web, "TaskEvent")
    assert task == "export type TaskEvent = NodeEvent | StepEvent"


def test_a_drift_is_reported_with_a_diff(tmp_path: Path):
    web = sync.WEB_API.read_text()
    drifted = web.replace("  pr_number?: number | null\n", "  pr_number?: number | null\n  pr_state?: string\n", 1)
    assert drifted != web
    drifted = drifted.replace("  | 'pr'\n", "", 1)  # a union member removed too
    other = tmp_path / "api.ts"
    other.write_text(drifted)
    problems = sync.compare(sync.WEB_API, other)
    names = [p.split(" ", 1)[0] for p in problems]
    assert names == ["NodeEvent", "StepEventType"]
    assert "+pr_state?: string" in problems[0]
    assert "-| 'pr'" in problems[1]

    proc = subprocess.run(  # noqa: S603 — our own script with a pytest tmp_path, not untrusted input
        [sys.executable, "scripts/check_ts_types_in_sync.py", "--extension", str(other)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "NodeEvent differs" in proc.stderr and "2 problem(s)" in proc.stderr


def test_a_missing_declaration_is_a_failure(tmp_path: Path):
    other = tmp_path / "api.ts"
    other.write_text("export interface NodeEvent { node: string }\n")
    problems = sync.compare(sync.WEB_API, other)
    assert any(
        p.startswith("StepEventType: extension") and "no `export interface|type StepEventType`" in p for p in problems
    )
    assert any(p.startswith("NodeEvent differs") for p in problems)
