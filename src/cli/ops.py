# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `keystone up`/`keystone bundle`/`keystone deploy cloud` real
command construction.

Pure, testable functions building the real subprocess argv for each
operation (docker compose up/exec, the sandbox image build, the airgap
bundle scripts, OpenTofu/Terraform against infra/opentofu/environments,
the Helm install for a cloud overlay) — kept separate from the Typer
commands in src/cli/main.py so the command-construction logic is testable
without actually invoking Docker, a cloud provider, or downloading
anything. The Typer commands run these argv lists via subprocess with
stdout/stderr inherited (not captured), so the user sees the same real,
live output `make up`/the airgap scripts/`tofu apply` already produce —
this is a thin, honest wrapper, not a reimplementation.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

CLOUDS: tuple[str, ...] = ("aws", "azure", "gcp")
"""Clouds with an infra/opentofu/modules/<cloud>-*-gpu module, an environments/<cloud>-pilot root
and a helm/keystone/values-<cloud>.yaml overlay."""

IAC_ACTIONS: tuple[str, ...] = ("init", "validate", "plan", "apply", "destroy")
"""OpenTofu/Terraform subcommands `keystone deploy cloud` is allowed to run."""

IAC_BINARIES: tuple[str, ...] = ("tofu", "terraform")
"""Preferred first. The HCL under infra/opentofu has no provider-specific syntax, so either works."""


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


def build_up_command(compose_file: Path, *, dev_only: bool = False, with_demo_model: bool = False) -> list[str]:
    """`with_demo_model` adds the `demo-model` compose profile (llama.cpp serving Qwen2.5-Coder-0.5B on CPU —
    a real OpenAI-compatible backend with no GPU) to whatever else comes up."""
    cmd = ["docker", "compose", "-f", str(compose_file)]
    if with_demo_model:
        cmd += ["--profile", "demo-model"]
    cmd += ["up", "-d"]
    if dev_only:
        cmd += ["postgres", "redis", "qdrant"]
        if with_demo_model:
            cmd.append("demo-model")
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


# ---------------------------------------------------------------------------
# `keystone deploy cloud` — OpenTofu/Terraform + Helm for the AWS/Azure/GCP GPU tiers
# ---------------------------------------------------------------------------


def find_iac_binary(path: str | None = None) -> str | None:
    """Absolute path of `tofu` if installed, else `terraform`, else None. `path` overrides the
    PATH searched (shutil.which's own parameter) so the lookup is testable against a real
    directory rather than whatever the developer's machine happens to have."""
    for name in IAC_BINARIES:
        found = shutil.which(name, path=path)
        if found:
            return found
    return None


def cloud_env_dir(repo_root: Path, cloud: str, env: str) -> Path:
    """infra/opentofu/environments/<cloud>-<env> — the root module `keystone deploy cloud` runs in."""
    if cloud not in CLOUDS:
        raise ValueError(f"cloud must be one of {', '.join(CLOUDS)}, not {cloud!r}")
    return repo_root / "infra" / "opentofu" / "environments" / f"{cloud}-{env}"


def build_tofu_command(env_dir: Path, action: str, var_file: Path | None = None, *, binary: str = "tofu") -> list[str]:
    """`tofu -chdir=<env_dir> <action> [-var-file=<var_file>]`.

    -chdir keeps the state file, .terraform/ and terraform.tfvars auto-loading inside the
    environment directory exactly as if the user had cd'd there. `apply`/`destroy` are left
    interactive on purpose (no -auto-approve): the confirmation prompt is the one real
    safety net between a typo and a deleted cluster. Pass an absolute `var_file` — with
    -chdir, relative paths resolve against env_dir, not the caller's cwd."""
    if action not in IAC_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(IAC_ACTIONS)}, not {action!r}")
    cmd = [binary, f"-chdir={env_dir}", action]
    if var_file is not None:
        if action in ("init", "validate"):
            raise ValueError(f"-var-file does not apply to `{action}`")
        cmd.append(f"-var-file={var_file}")
    return cmd


def build_tofu_output_command(env_dir: Path, name: str, *, binary: str = "tofu") -> list[str]:
    """`tofu -chdir=<env_dir> output -raw <name>` — how the CLI reads a single string output
    (kubeconfig_command, helm_install_command) back after an apply."""
    return [binary, f"-chdir={env_dir}", "output", "-raw", name]


def default_cloud_values_files(cloud: str, chart_dir: Path = Path("helm/keystone")) -> list[Path]:
    """The overlay stack every cloud pilot installs with: the production client-VPC values, then the
    cloud file that fills in its StorageClass / node-label TODOs."""
    if cloud not in CLOUDS:
        raise ValueError(f"cloud must be one of {', '.join(CLOUDS)}, not {cloud!r}")
    return [chart_dir / "values-client-vpc.yaml", chart_dir / f"values-{cloud}.yaml"]


def build_helm_install_command(
    cloud: str,
    release: str,
    namespace: str,
    values_files: Sequence[Path] | None = None,
    *,
    chart_dir: Path = Path("helm/keystone"),
) -> list[str]:
    """`helm upgrade --install <release> helm/keystone -n <namespace> --create-namespace -f ... --wait`.

    `values_files` None means the cloud's default stack (default_cloud_values_files); an explicit
    list is used verbatim, in order — Helm applies later files over earlier ones."""
    files = list(values_files) if values_files is not None else default_cloud_values_files(cloud, chart_dir)
    cmd = ["helm", "upgrade", "--install", release, str(chart_dir), "--namespace", namespace, "--create-namespace"]
    for values in files:
        cmd += ["-f", str(values)]
    cmd += ["--wait", "--timeout", "15m"]
    return cmd
