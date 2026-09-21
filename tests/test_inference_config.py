# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Unit tests for src/inference/config.py's VLLMConfig CLI arg generation — pure logic, no infra needed."""

from __future__ import annotations

from src.inference.config import (
    coding_fallback_model_config,
    coding_model_config,
    reasoning_model_config,
)


def test_coding_model_is_glm53_flash():
    cfg = coding_model_config()
    assert cfg.model == "zai-org/GLM-5.3-Flash"
    assert cfg.served_model_name == "glm-5.3-flash"


def test_coding_model_does_not_pass_quantization_flag():
    """GLM-5.3-Flash ships natively FP8-quantized — vLLM auto-detects this from the checkpoint's own
    quantization_config, so --quantization must NOT be passed (that flag is for quantizing a BF16 checkpoint
    at load time)."""
    cfg = coding_model_config()
    args = cfg.to_cli_args()
    assert "--quantization" not in args


def test_coding_model_emits_glm47_tool_parser():
    cfg = coding_model_config()
    args = cfg.to_cli_args()
    assert "--enable-auto-tool-choice" in args
    idx = args.index("--tool-call-parser")
    assert args[idx + 1] == "glm47"


def test_coding_fallback_model_is_qwen():
    cfg = coding_fallback_model_config()
    assert cfg.model == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_coding_fallback_model_emits_hermes_tool_parser():
    cfg = coding_fallback_model_config()
    args = cfg.to_cli_args()
    idx = args.index("--tool-call-parser")
    assert args[idx + 1] == "hermes"


def test_tool_parser_flags_omitted_when_tool_parser_is_none():
    cfg = reasoning_model_config()
    args = cfg.to_cli_args()
    assert "--tool-call-parser" not in args
    assert "--enable-auto-tool-choice" not in args


def test_tool_parser_flags_omitted_for_text_protocol_even_if_tool_parser_set():
    cfg = coding_model_config()
    cfg.tool_protocol = "text"
    args = cfg.to_cli_args()
    assert "--tool-call-parser" not in args


def test_gpu_count_controls_tensor_parallel_size():
    cfg = coding_model_config(gpu_count=8)
    args = cfg.to_cli_args()
    idx = args.index("--tensor-parallel-size")
    assert args[idx + 1] == "8"


def test_to_command_includes_model_flag():
    cfg = coding_model_config()
    command = cfg.to_command()
    assert "--model zai-org/GLM-5.3-Flash" in command
