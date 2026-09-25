# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real unit tests for the vision-capable catalog entry added for image-attached tasks
(src/inference/catalog.py) — pure, no I/O.
"""

from __future__ import annotations

import pytest

from src.inference.catalog import CATALOG, ROLE_MODELS, CatalogEntry, find


def test_catalog_entry_defaults_to_not_vision_capable():
    entry = CatalogEntry("some/model", 7.0, "Apache-2.0", 32_768)
    assert entry.vision is False


def test_qwen_coder_catalog_entries_are_not_vision_capable():
    assert all(not entry.vision for entry in CATALOG)


def test_qwen_vl_role_model_is_vision_capable_and_findable():
    entry = find("Qwen/Qwen2.5-VL-7B-Instruct")
    assert entry.vision is True
    assert entry in ROLE_MODELS


def test_qwen_vl_is_not_a_fine_tune_catalog_entry():
    """Vision entries are served-only — they must live in ROLE_MODELS, not the fine-tune CATALOG,
    per the file's own rule that the fine-tune catalog is scoped to the Qwen2.5-Coder family."""
    assert "Qwen/Qwen2.5-VL-7B-Instruct" not in {e.hf_id for e in CATALOG}


def test_existing_role_models_stay_not_vision_capable():
    glm = find("zai-org/GLM-5.3-Flash")
    deepseek = find("deepseek-ai/DeepSeek-R1")
    assert glm.vision is False
    assert deepseek.vision is False


def test_to_dict_includes_vision_field():
    entry = find("Qwen/Qwen2.5-VL-7B-Instruct")
    assert entry.to_dict()["vision"] is True


def test_find_raises_for_an_unknown_model_id():
    with pytest.raises(KeyError):
        find("not/a-real-model")
