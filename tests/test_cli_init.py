# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real tests for src/cli/init.py's env-file generation — pure logic, no infra needed, but verified against this
repo's own real .env.example so a template drift breaks this test rather than shipping silently broken."""

from __future__ import annotations

from pathlib import Path

from src.cli.init import default_overrides, find_env_template, generate_secret, render_env_file
from src.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_generate_secret_is_a_real_random_64_char_hex_string():
    a, b = generate_secret(), generate_secret()
    assert a != b
    assert len(a) == 64
    int(a, 16)  # raises ValueError if not real hex


def test_render_env_file_overrides_matching_keys_and_preserves_everything_else():
    template = "# comment\nFOO=old\nBAR=unrelated\n\nBAZ=old2\n"
    result = render_env_file(template, {"FOO": "new", "BAZ": "new2"})
    lines = result.splitlines()
    assert "# comment" in lines
    assert "FOO=new" in lines
    assert "BAR=unrelated" in lines
    assert "BAZ=new2" in lines


def test_render_env_file_appends_override_keys_with_no_matching_template_line():
    result = render_env_file("EXISTING=1\n", {"EXISTING": "2", "BRAND_NEW_KEY": "value"})
    assert "EXISTING=2" in result
    assert "BRAND_NEW_KEY=value" in result


def test_default_overrides_keeps_postgres_password_and_database_url_consistent():
    overrides = default_overrides()
    password = overrides["POSTGRES_PASSWORD"]
    assert password in overrides["DATABASE_URL"]
    assert overrides["DATABASE_URL"].startswith("postgresql+asyncpg://keystone:")


def test_default_overrides_keeps_redis_password_and_redis_url_consistent():
    overrides = default_overrides()
    password = overrides["REDIS_PASSWORD"]
    assert password in overrides["REDIS_URL"]


def test_default_overrides_empty_when_auto_secrets_is_false():
    assert default_overrides(auto_secrets=False) == {}


def test_find_env_template_locates_this_repos_real_env_example():
    found = find_env_template(start=REPO_ROOT)
    assert found is not None
    assert found == REPO_ROOT / ".env.example"


def test_find_env_template_walks_up_from_a_subdirectory():
    found = find_env_template(start=REPO_ROOT / "src" / "cli")
    assert found == REPO_ROOT / ".env.example"


def test_find_env_template_returns_none_outside_any_keystone_checkout(tmp_path):
    assert find_env_template(start=tmp_path) is None


def test_rendering_this_repos_real_env_example_with_default_overrides_produces_a_settings_loadable_file(
    tmp_path, monkeypatch
):
    """End-to-end proof, not just unit-level: the actual .env.example this
    repo ships, real generated overrides, written to a real file, loaded
    by the real Settings class — the exact path `keystone init` drives.
    Clears DATABASE_URL/REDIS_URL from the real ambient environment first
    — pydantic-settings' env vars take priority over _env_file by design,
    and this test suite's own other integration tests already export
    both, which would otherwise mask what this .env file actually says."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    template_path = find_env_template(start=REPO_ROOT)
    assert template_path is not None

    content = render_env_file(template_path.read_text(), default_overrides())
    env_file = tmp_path / ".env"
    env_file.write_text(content)

    settings = Settings(_env_file=str(env_file))
    assert settings.database_url.startswith("postgresql+asyncpg://keystone:")
    assert settings.redis_url.startswith("redis://:")
    assert settings.git_allowed_hosts == ["gitea.internal.keystone.local"]


def test_the_real_keystone_init_command_writes_a_settings_loadable_env_file(tmp_path, monkeypatch):
    """The actual wired-up Typer command (src/cli/main.py's `init`), run
    non-interactively — not just init.py's underlying functions in
    isolation."""
    from typer.testing import CliRunner

    from src.cli.main import app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(REPO_ROOT)
    output_path = tmp_path / ".env"
    result = CliRunner().invoke(app, ["init", "--yes", "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    settings = Settings(_env_file=str(output_path))
    generated_password = output_path.read_text().split("POSTGRES_PASSWORD=")[1].split("\n")[0]
    assert settings.database_url == f"postgresql+asyncpg://keystone:{generated_password}@postgres:5432/keystone"


def test_keystone_init_refuses_to_silently_overwrite_an_existing_env_file(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from src.cli.main import app

    monkeypatch.chdir(REPO_ROOT)
    output_path = tmp_path / ".env"
    output_path.write_text("EXISTING=1\n")

    result = CliRunner().invoke(app, ["init", "--output", str(output_path)], input="n\n")

    assert result.exit_code == 1
    assert output_path.read_text() == "EXISTING=1\n"
