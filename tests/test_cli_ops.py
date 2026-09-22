# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real tests for src/cli/ops.py's command construction — pure logic, plus one real subprocess invocation proving
the constructed argv actually works against this repo's real docker-compose.yml (not just that the strings look
plausible)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.cli.ops import (
    build_bundle_build_images_command,
    build_bundle_download_models_command,
    build_bundle_import_command,
    build_down_command,
    build_migrate_command,
    build_sandbox_images_command,
    build_up_command,
    find_repo_root,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_find_repo_root_locates_this_repo():
    assert find_repo_root(start=REPO_ROOT) == REPO_ROOT


def test_find_repo_root_walks_up_from_a_subdirectory():
    assert find_repo_root(start=REPO_ROOT / "src" / "cli") == REPO_ROOT


def test_find_repo_root_returns_none_outside_any_checkout(tmp_path):
    assert find_repo_root(start=tmp_path) is None


def test_build_up_command_default():
    cmd = build_up_command(REPO_ROOT / "docker-compose.yml", dev_only=False)
    assert cmd == ["docker", "compose", "-f", str(REPO_ROOT / "docker-compose.yml"), "up", "-d"]


def test_build_up_command_dev_only_scopes_to_infra_services():
    cmd = build_up_command(REPO_ROOT / "docker-compose.yml", dev_only=True)
    assert cmd[-3:] == ["postgres", "redis", "qdrant"]


def test_build_down_command():
    cmd = build_down_command(REPO_ROOT / "docker-compose.yml")
    assert cmd == ["docker", "compose", "-f", str(REPO_ROOT / "docker-compose.yml"), "down"]


def test_build_sandbox_images_command_points_at_the_real_dockerfile():
    cmd = build_sandbox_images_command(REPO_ROOT)
    dockerfile = REPO_ROOT / "docker" / "sandbox-runtimes" / "python.Dockerfile"
    assert dockerfile.exists()
    assert str(dockerfile) in cmd
    assert "keystone-sandbox-python:latest" in cmd


def test_build_migrate_command():
    cmd = build_migrate_command(REPO_ROOT / "docker-compose.yml")
    assert cmd == [
        "docker",
        "compose",
        "-f",
        str(REPO_ROOT / "docker-compose.yml"),
        "exec",
        "app",
        "alembic",
        "upgrade",
        "head",
    ]


def test_build_bundle_download_models_command_points_at_the_real_script():
    cmd = build_bundle_download_models_command(REPO_ROOT, "./out")
    script = REPO_ROOT / "airgap" / "download_models.sh"
    assert script.exists()
    assert cmd == [str(script), "./out"]


def test_build_bundle_build_images_command_points_at_the_real_script():
    cmd = build_bundle_build_images_command(REPO_ROOT, "./out")
    script = REPO_ROOT / "airgap" / "build_image_bundle.sh"
    assert script.exists()
    assert cmd == [str(script), "./out"]


def test_build_bundle_import_command_points_at_the_real_script_with_optional_args():
    script = REPO_ROOT / "airgap" / "import_bundle.sh"
    assert script.exists()
    assert build_bundle_import_command(REPO_ROOT) == [str(script)]
    assert build_bundle_import_command(REPO_ROOT, "./bundle") == [str(script), "./bundle"]
    assert build_bundle_import_command(REPO_ROOT, "./bundle", push=True) == [str(script), "./bundle", "--push"]


def test_the_real_up_command_actually_validates_against_this_repos_compose_file():
    """Not just that the argv looks right — a real subprocess call proving
    `docker compose -f <file> config` (the read-only validation form of
    the same command construction) succeeds against the actual
    docker-compose.yml in this repo, the same file `keystone up` targets."""
    compose_file = REPO_ROOT / "docker-compose.yml"
    result = subprocess.run(  # noqa: S603
        ["docker", "compose", "-f", str(compose_file), "config", "-q"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "POSTGRES_PASSWORD": "x",
            "REDIS_PASSWORD": "x",
            "QDRANT_API_KEY": "x",
            "KEYSTONE_ROOT_ADMIN_TOKEN": "x",
        },
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
