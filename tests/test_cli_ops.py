# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real tests for src/cli/ops.py's command construction — pure logic, plus one real subprocess invocation proving
the constructed argv actually works against this repo's real docker-compose.yml (not just that the strings look
plausible)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.cli.ops import (
    CLOUDS,
    build_bundle_build_images_command,
    build_bundle_download_models_command,
    build_bundle_import_command,
    build_down_command,
    build_helm_install_command,
    build_migrate_command,
    build_sandbox_images_command,
    build_tofu_command,
    build_tofu_output_command,
    build_up_command,
    cloud_env_dir,
    default_cloud_values_files,
    find_iac_binary,
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


def test_every_sandbox_runtime_image_has_a_real_dockerfile_and_a_build_command():
    from src.cli.ops import build_sandbox_image_commands

    cmds = build_sandbox_image_commands(REPO_ROOT)
    tags = [c[c.index("-t") + 1] for c in cmds]
    assert tags == ["keystone-sandbox-python:latest", "keystone-sandbox-node:latest", "keystone-sandbox-go:latest"]
    for c in cmds:
        assert Path(c[c.index("-f") + 1]).exists()


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


# ---------------------------------------------------------------------------
# `keystone deploy cloud` — OpenTofu/Terraform + Helm argv
# ---------------------------------------------------------------------------

ENV_DIR = REPO_ROOT / "infra" / "opentofu" / "environments" / "aws-pilot"


def test_every_cloud_has_a_pilot_environment_and_an_overlay():
    for cloud in CLOUDS:
        env_dir = cloud_env_dir(REPO_ROOT, cloud, "pilot")
        assert (env_dir / "main.tf").is_file(), env_dir
        assert (env_dir / "terraform.tfvars.example").is_file(), env_dir
        for values in default_cloud_values_files(cloud, REPO_ROOT / "helm" / "keystone"):
            assert values.is_file(), values


def test_cloud_env_dir_rejects_unknown_cloud():
    with pytest.raises(ValueError, match="cloud must be one of"):
        cloud_env_dir(REPO_ROOT, "digitalocean", "pilot")


def test_build_tofu_command_init_plan_apply_destroy_exact_argv():
    assert build_tofu_command(ENV_DIR, "init") == ["tofu", f"-chdir={ENV_DIR}", "init"]
    assert build_tofu_command(ENV_DIR, "plan") == ["tofu", f"-chdir={ENV_DIR}", "plan"]
    assert build_tofu_command(ENV_DIR, "apply") == ["tofu", f"-chdir={ENV_DIR}", "apply"]
    assert build_tofu_command(ENV_DIR, "destroy") == ["tofu", f"-chdir={ENV_DIR}", "destroy"]
    assert build_tofu_command(ENV_DIR, "validate", binary="terraform") == [
        "terraform",
        f"-chdir={ENV_DIR}",
        "validate",
    ]


def test_build_tofu_command_never_auto_approves():
    for action in ("apply", "destroy"):
        assert "-auto-approve" not in build_tofu_command(ENV_DIR, action)


def test_build_tofu_command_var_file_only_where_it_applies():
    var_file = ENV_DIR / "terraform.tfvars.example"
    assert build_tofu_command(ENV_DIR, "plan", var_file) == [
        "tofu",
        f"-chdir={ENV_DIR}",
        "plan",
        f"-var-file={var_file}",
    ]
    with pytest.raises(ValueError, match="-var-file does not apply"):
        build_tofu_command(ENV_DIR, "init", var_file)
    with pytest.raises(ValueError, match="action must be one of"):
        build_tofu_command(ENV_DIR, "console")


def test_build_tofu_output_command():
    assert build_tofu_output_command(ENV_DIR, "kubeconfig_command") == [
        "tofu",
        f"-chdir={ENV_DIR}",
        "output",
        "-raw",
        "kubeconfig_command",
    ]


def test_build_helm_install_command_default_overlay_stack_exact_argv():
    assert build_helm_install_command("gcp", "keystone", "keystone") == [
        "helm",
        "upgrade",
        "--install",
        "keystone",
        "helm/keystone",
        "--namespace",
        "keystone",
        "--create-namespace",
        "-f",
        "helm/keystone/values-client-vpc.yaml",
        "-f",
        "helm/keystone/values-gcp.yaml",
        "--wait",
        "--timeout",
        "15m",
    ]


def test_build_helm_install_command_explicit_values_files_in_order():
    chart = REPO_ROOT / "helm" / "keystone"
    files = [chart / "values-client-vpc.yaml", chart / "values-azure.yaml", Path("/srv/overrides/extra.yaml")]
    cmd = build_helm_install_command("azure", "ks", "team-a", files, chart_dir=chart)
    assert cmd[:8] == ["helm", "upgrade", "--install", "ks", str(chart), "--namespace", "team-a", "--create-namespace"]
    assert cmd[8:14] == ["-f", str(files[0]), "-f", str(files[1]), "-f", "/srv/overrides/extra.yaml"]
    assert cmd[14:] == ["--wait", "--timeout", "15m"]
    with pytest.raises(ValueError, match="cloud must be one of"):
        build_helm_install_command("oci", "ks", "ns")


def test_find_iac_binary_prefers_tofu_then_terraform_then_none(tmp_path):
    """Real lookups against a real directory: two executables named tofu/terraform, then one, then none."""

    def make(name: str) -> Path:
        exe = tmp_path / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        return exe

    assert find_iac_binary(path=str(tmp_path)) is None
    terraform = make("terraform")
    assert find_iac_binary(path=str(tmp_path)) == str(terraform)
    tofu = make("tofu")
    assert find_iac_binary(path=str(tmp_path)) == str(tofu)
    assert find_iac_binary(path=str(tmp_path / "empty")) is None


@pytest.mark.skipif(find_iac_binary() is None, reason="neither tofu nor terraform installed")
def test_the_real_tofu_validate_command_succeeds_against_every_pilot_environment():
    """Not just that the argv looks right — the constructed `init -backend=false` + `validate` argv
    really runs against each environment root (no cloud credentials involved; providers are
    downloaded), the same check CI's iac-validate job performs."""
    binary = find_iac_binary()
    assert binary is not None
    for cloud in CLOUDS:
        env_dir = cloud_env_dir(REPO_ROOT, cloud, "pilot")
        init = [*build_tofu_command(env_dir, "init", binary=binary), "-backend=false", "-input=false", "-no-color"]
        result = subprocess.run(init, capture_output=True, text=True, timeout=600)  # noqa: S603
        assert result.returncode == 0, result.stderr
        validate = [*build_tofu_command(env_dir, "validate", binary=binary), "-no-color"]
        result = subprocess.run(validate, capture_output=True, text=True, timeout=120)  # noqa: S603
        assert result.returncode == 0, result.stdout + result.stderr
