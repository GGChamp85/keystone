# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Unit tests for src/inference/config.py's VLLMConfig CLI arg generation — pure logic, no infra needed."""

from __future__ import annotations

from src.inference.config import (
    VLLMConfig,
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


def test_lora_disabled_by_default_emits_no_lora_flags():
    """No deployment should get --enable-lora it didn't ask for — flags must
    stay absent by default, not just default to empty/zero values."""
    cfg = coding_model_config()
    args = cfg.to_cli_args()
    assert "--enable-lora" not in args
    assert "--max-lora-rank" not in args
    assert "--max-loras" not in args
    assert "--lora-modules" not in args


def test_enabling_lora_emits_the_real_vllm_flags():
    """Flag names/syntax verified against vLLM's real current docs
    (docs.vllm.ai/en/latest/features/lora.html): --enable-lora,
    --max-lora-rank N, --max-loras N, --lora-modules name=path ..."""
    cfg = VLLMConfig(model="zai-org/GLM-5.3-Flash", enable_lora=True, max_lora_rank=32, max_loras=2)
    args = cfg.to_cli_args()
    assert "--enable-lora" in args
    rank_idx = args.index("--max-lora-rank")
    assert args[rank_idx + 1] == "32"
    loras_idx = args.index("--max-loras")
    assert args[loras_idx + 1] == "2"


def test_lora_modules_render_as_name_equals_path_pairs():
    cfg = VLLMConfig(
        model="zai-org/GLM-5.3-Flash",
        enable_lora=True,
        lora_modules=[("tenant-a-lora", "/data/adapters/tenant-a"), ("tenant-b-lora", "/data/adapters/tenant-b")],
    )
    args = cfg.to_cli_args()
    idx = args.index("--lora-modules")
    assert args[idx + 1 : idx + 3] == ["tenant-a-lora=/data/adapters/tenant-a", "tenant-b-lora=/data/adapters/tenant-b"]


def test_lora_modules_omitted_flag_when_enabled_but_no_modules_given():
    """--enable-lora with no --lora-modules is a real, valid vLLM startup
    (adapters can be registered dynamically later via the load-adapter API)
    — the flag must simply be absent, not emitted with an empty value."""
    cfg = VLLMConfig(model="zai-org/GLM-5.3-Flash", enable_lora=True)
    args = cfg.to_cli_args()
    assert "--enable-lora" in args
    assert "--lora-modules" not in args
