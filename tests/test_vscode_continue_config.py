# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
`keystone ide continue-config` (src/cli/ide.py): the rendered YAML is
parsed back and checked field by field against Continue's config-yaml
schema (continuedev/continue, packages/config-yaml/src/schemas): model
entries carry provider/model/apiBase/apiKey/roles with roles from the
schema's enum; the MCP server is the streamable-http form with url and
requestOptions.headers. A wrong field name would be silently ignored by
Continue, so this is what keeps the generated file honest.
"""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from src.cli.ide import ContinueConfigInputs, render_continue_config
from src.cli.main import app

_ROLES = {"chat", "autocomplete", "embed", "rerank", "edit", "apply", "summarize", "subagent"}  # schema enum


def _rendered(**overrides) -> dict:
    inputs = ContinueConfigInputs(base_url="https://keystone.internal:8080/", api_key="ks-test-key", **overrides)
    return yaml.safe_load(render_continue_config(inputs))


def test_models_point_every_role_at_the_gateway_with_the_key():
    cfg = _rendered()
    assert (cfg["name"], cfg["schema"]) == ("Keystone", "v1")
    coding, autocomplete = cfg["models"]
    for model in (coding, autocomplete):
        assert model["provider"] == "openai"
        assert model["apiBase"] == "https://keystone.internal:8080/v1"  # trailing slash normalised, /v1 appended once
        assert model["apiKey"] == "ks-test-key"
        assert set(model["roles"]) <= _ROLES
    assert coding["model"] == "coding" and set(coding["roles"]) == {"chat", "edit", "apply", "summarize"}
    assert coding["capabilities"] == ["tool_use"]
    assert autocomplete["model"] == "coding_fallback" and autocomplete["roles"] == ["autocomplete"]
    assert autocomplete["defaultCompletionOptions"]["maxTokens"] == 256


def test_mcp_server_is_the_streamable_http_form_with_bearer_auth():
    cfg = _rendered()
    (mcp,) = cfg["mcpServers"]
    assert mcp["type"] == "streamable-http"
    assert mcp["url"] == "https://keystone.internal:8080/v1/keystone/mcp"
    assert mcp["requestOptions"]["headers"]["Authorization"] == "Bearer ks-test-key"


def test_ca_bundle_is_threaded_into_every_request_options():
    cfg = _rendered(ca_bundle_path="/etc/keystone/pki/ca.crt")
    for model in cfg["models"]:
        assert model["requestOptions"]["caBundlePath"] == "/etc/keystone/pki/ca.crt"
    assert cfg["mcpServers"][0]["requestOptions"]["caBundlePath"] == "/etc/keystone/pki/ca.crt"


def test_cli_writes_the_file_and_reads_the_key_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYSTONE_API_KEY", "ks-from-env")
    out = tmp_path / "config.yaml"
    result = CliRunner().invoke(
        app, ["ide", "continue-config", "--base-url", "http://localhost:8080", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(out.read_text())
    assert cfg["models"][0]["apiKey"] == "ks-from-env"
    assert cfg["models"][0]["apiBase"] == "http://localhost:8080/v1"


def test_checked_in_example_matches_the_renderer():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "vscode" / "continue" / "config.example.yaml"
    expected = render_continue_config(
        ContinueConfigInputs(base_url="https://keystone.internal:8080", api_key="ks-XXXX-XXXXXXXX")
    )
    assert example.read_text() == expected, (
        "regenerate: keystone ide continue-config ... > vscode/continue/config.example.yaml"
    )
