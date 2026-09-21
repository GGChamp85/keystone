# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — vLLM launch configurations for each GPU node.

These produce the exact CLI args for `python -m vllm.entrypoints.openai.api_server`.
Optimizations applied:
  - Automatic Prefix Caching (APC) for shared system prompts
  - Chunked prefill for long-context coding tasks
  - FP8 quantization on H100/L40S (falls back to FP16 on A100)
  - PagedAttention with tuned block sizes
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VLLMConfig:
    model: str
    served_model_name: str | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.92
    max_num_seqs: int = 64
    max_num_batched_tokens: int | None = None

    # Prefix caching & chunked prefill
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_chunked_prefill_tokens: int = 8192

    # Quantization
    quantization: str | None = None  # "fp8", "awq", "gptq", None
    dtype: str = "auto"

    # KV cache
    block_size: int = 16
    swap_space_gb: int = 4

    # Serving
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    disable_log_requests: bool = False
    enforce_eager: bool = False

    # Trust remote code (needed for some models)
    trust_remote_code: bool = True

    # Tool-calling — see src/orchestrator/tools/protocol.py. `tool_protocol`
    # picks which protocol nodes/coding.py uses for this role at the
    # application layer (native tool_calls vs a text-fenced fallback for a
    # role with no working vLLM tool-parser); `tool_parser` is the real
    # `--tool-call-parser` value vLLM needs when `tool_protocol="native"`
    # (None means don't pass the flag — e.g. a reasoning-only role that
    # never receives `tools=`).
    tool_protocol: str = "native"  # "native" | "text"
    tool_parser: str | None = None

    # LoRA adapter serving (src/db/models.py's ModelAdapter registry) — flags
    # verified against vLLM's real current docs (docs.vllm.ai/en/latest/
    # features/lora.html): `--enable-lora`, `--max-lora-rank N`,
    # `--max-loras N`, `--lora-modules name=path name2=path2 ...`.
    # `lora_modules` is `(served_name, path)` pairs — src/inference/
    # model_router.py builds this list from each tenant's *promoted*
    # ModelAdapter rows, so a served name always corresponds to a real,
    # currently-promoted adapter, never a stale or candidate one.
    enable_lora: bool = False
    max_lora_rank: int = 64
    max_loras: int = 4
    lora_modules: list[tuple[str, str]] = field(default_factory=list)

    def to_cli_args(self) -> list[str]:
        args = [
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--block-size",
            str(self.block_size),
            "--swap-space",
            str(self.swap_space_gb),
            "--dtype",
            self.dtype,
        ]

        if self.served_model_name:
            args += ["--served-model-name", self.served_model_name]

        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
            args += ["--max-num-batched-tokens", str(self.max_chunked_prefill_tokens)]
        elif self.max_num_batched_tokens:
            args += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]

        if self.quantization:
            args += ["--quantization", self.quantization]

        if self.api_key:
            args += ["--api-key", self.api_key]

        if self.trust_remote_code:
            args.append("--trust-remote-code")

        if self.disable_log_requests:
            args.append("--disable-log-requests")

        if self.enforce_eager:
            args.append("--enforce-eager")

        if self.tool_protocol == "native" and self.tool_parser:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", self.tool_parser]

        if self.enable_lora:
            args.append("--enable-lora")
            args += ["--max-lora-rank", str(self.max_lora_rank)]
            args += ["--max-loras", str(self.max_loras)]
            if self.lora_modules:
                args.append("--lora-modules")
                args += [f"{name}={path}" for name, path in self.lora_modules]

        return args

    def to_command(self) -> str:
        args = self.to_cli_args()
        return f"python -m vllm.entrypoints.openai.api_server {' '.join(args)}"


# ── Pre-built configs for the three model roles ───────────────


def coding_model_config(
    gpu_count: int = 8,
    max_context: int = 131072,
) -> VLLMConfig:
    """
    GPU Node A: GLM-5.3-Flash (zai-org, MIT) — primary coding model.

    Real footprint, verified against the model's own published config
    (huggingface.co/zai-org/GLM-5.3-Flash): a 288-expert MoE, ~321B total
    parameters, shipped natively FP8-quantized on disk (~314B params in
    FP8 + ~6.9B in BF16 ≈ ~328GB of weights) — vLLM auto-detects the
    checkpoint's own `quantization_config`, so no `--quantization` flag is
    passed here (that flag is for quantizing a BF16 checkpoint at load
    time, not a model that's already FP8 on disk). 8x GPU tensor-parallel
    is the realistic floor for the weights alone (8x80GB A100/H100 =
    640GB, or 4x141GB H200 with a tighter `gpu_memory_utilization`);
    hybrid attention (34 of 45 layers use gated linear attention, only 11
    are full attention) plus MLA (kv_lora_rank=512) makes its real
    1,048,576-token max context tractable, but this default stays at 128K
    — the same practical ceiling the rest of Keystone already assumes —
    since serving the full 1M context multiplies KV-cache memory well
    beyond what an 8-GPU node budgets for by default; raise `max_context`
    per-deployment once real KV-cache headroom is confirmed on real
    hardware.
    """
    return VLLMConfig(
        model="zai-org/GLM-5.3-Flash",
        served_model_name="glm-5.3-flash",
        tensor_parallel_size=gpu_count,
        max_model_len=max_context,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_chunked_prefill_tokens=16384,
        quantization=None,  # already FP8 on disk — see docstring
        block_size=16,
        swap_space_gb=8,
        tool_protocol="native",
        # No dedicated "glm5"/"glm53" parser exists in vLLM's tool-parser
        # registry (verified against vllm/tool_parsers/__init__.py) — GLM-5.3's
        # own chat template emits the identical <tool_call>name<arg_key>...
        # XML format as GLM-4.5/4.7 MoE (verified against the model's real
        # chat_template.jinja), which is why "glm45"/"glm47" are registered
        # as aliases of the same parser class and no glm5-specific one was
        # needed. Reconfirm with scripts/smoke_glm_toolcall.py on the first
        # real GPU endpoint before relying on this in production.
        tool_parser="glm47",
    )


def coding_fallback_model_config(
    gpu_count: int = 4,
    max_context: int = 131072,
    use_fp8: bool = False,
) -> VLLMConfig:
    """
    GPU Node A2: Qwen2.5-Coder-32B — coding fallback, used when GLM-5.3-Flash
    is unhealthy or unavailable (src/inference/model_router.py's
    FALLBACK_CHAINS). 4x A100 80GB tensor-parallel, 128K context window.
    """
    return VLLMConfig(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        served_model_name="qwen-coder-32b",
        tensor_parallel_size=gpu_count,
        max_model_len=max_context,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_chunked_prefill_tokens=16384,
        quantization="fp8" if use_fp8 else None,
        block_size=16,
        swap_space_gb=8,
        tool_protocol="native",
        tool_parser="hermes",
    )


def reasoning_model_config(
    gpu_count: int = 2,
    max_context: int = 65536,
) -> VLLMConfig:
    """
    GPU Node B: DeepSeek-R1 — reasoning critic for code review.
    2x A100 80GB tensor-parallel, 64K context.
    """
    return VLLMConfig(
        model="deepseek-ai/DeepSeek-R1",
        served_model_name="deepseek-r1",
        tensor_parallel_size=gpu_count,
        max_model_len=max_context,
        gpu_memory_utilization=0.90,
        max_num_seqs=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_chunked_prefill_tokens=8192,
        block_size=16,
        swap_space_gb=4,
    )
