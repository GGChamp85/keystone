# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""The truth-in-docs gates: the generated configuration reference is complete and current, and the claims
checker really fails on a claim whose text or test does not exist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from scripts import check_claims, gen_config_reference
from src.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_configuration_reference_lists_every_setting_with_a_description():
    fields = gen_config_reference.collect_fields()
    names = {f["name"] for f in fields}
    assert names == set(Settings.model_fields)  # nothing skipped, nothing invented
    undocumented = sorted(f["env"] for f in fields if not f["description"])
    assert undocumented == [], f"settings without a description in src/config.py: {undocumented}"
    secrets = [f for f in fields if "secret" in f["type"]]
    assert secrets and all(f["default"] in ("(generated at startup)", "(secret; set it)", "unset") for f in secrets)
    page = gen_config_reference.render(fields)
    assert "| `MAX_TOKENS_PER_REQUEST` |" in page and "0 = the model's own limit" in page


def test_configuration_reference_page_is_current():
    proc = subprocess.run(
        [sys.executable, "scripts/gen_config_reference.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_claims_file_checks_out_against_the_collected_suite(tmp_path: Path):
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    saved = tmp_path / "collected.txt"
    saved.write_text(collected)
    assert check_claims.check(saved_collection=saved) == []

    # a claim whose text is not in the doc, and one naming a test that does not exist, are both caught
    bad = tmp_path / "claims.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "claims": [
                    {"text": "this sentence is nowhere", "docs": ["README.md"], "tests": ["tests/test_docs_gates.py"]},
                    {
                        "text": "Keystone",
                        "docs": ["README.md"],
                        "tests": ["tests/test_docs_gates.py::test_that_does_not_exist"],
                    },
                ]
            }
        )
    )
    problems = check_claims.check(claims_path=bad, saved_collection=saved)
    assert len(problems) == 2
    assert "text not found verbatim in README.md" in problems[0]
    assert "is not collected by pytest" in problems[1]


def test_cli_and_api_reference_pages_are_current():
    proc = subprocess.run(
        [sys.executable, "scripts/gen_cli_reference.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    cli = (ROOT / "docs" / "reference" / "cli.md").read_text()
    api = (ROOT / "docs" / "reference" / "api.md").read_text()
    assert "## keystone finetune promote" in cli and "--force" in cli
    assert (
        "| `POST` | `/v1/messages` |" in api
        and "| `POST` | `/v1/admin/tenants/{tenant_id}/keys/{key_prefix}/rotate` |" in api
    )
