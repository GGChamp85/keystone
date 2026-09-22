# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `keystone up`/`keystone bundle` real command construction.

Pure, testable functions building the real subprocess argv for each
operation (docker compose up/exec, the sandbox image build, the airgap
bundle scripts) — kept separate from the Typer commands in src/cli/main.py
so the command-construction logic is testable without actually invoking
Docker or downloading anything. The Typer commands run these argv lists
via subprocess with stdout/stderr inherited (not captured), so the user
sees the same real, live output `make up`/the airgap scripts already
produce — this is a thin, honest wrapper, not a reimplementation.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walks up from `start` (default: CWD) looking for docker-compose.yml
    — the same "must be run from inside a checkout" contract
    src/cli/init.py's find_env_template already establishes, generalized
    to whatever file marks the repo root for a given operation."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "docker-compose.yml").exists():
            return candidate
    return None


def build_up_command(compose_file: Path, *, dev_only: bool = False) -> list[str]:
    cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if dev_only:
        cmd += ["postgres", "redis", "qdrant"]
    return cmd


def build_down_command(compose_file: Path) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), "down"]


def build_sandbox_images_command(repo_root: Path) -> list[str]:
    dockerfile = repo_root / "docker" / "sandbox-runtimes" / "python.Dockerfile"
    return ["docker", "build", "-f", str(dockerfile), "-t", "keystone-sandbox-python:latest", str(repo_root)]


def build_migrate_command(compose_file: Path) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), "exec", "app", "alembic", "upgrade", "head"]


def build_bundle_download_models_command(repo_root: Path, output_dir: str) -> list[str]:
    return [str(repo_root / "airgap" / "download_models.sh"), output_dir]


def build_bundle_build_images_command(repo_root: Path, output_dir: str) -> list[str]:
    return [str(repo_root / "airgap" / "build_image_bundle.sh"), output_dir]


def build_bundle_import_command(repo_root: Path, bundle_dir: str | None = None, *, push: bool = False) -> list[str]:
    cmd = [str(repo_root / "airgap" / "import_bundle.sh")]
    if bundle_dir:
        cmd.append(bundle_dir)
    if push:
        cmd.append("--push")
    return cmd
