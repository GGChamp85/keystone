# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the small-language-model catalog.

The models an enterprise admin can pick for a fine-tune or a deployment
without reading a model card: id, size, license, context, and what they
cost in GPU memory to serve or to train. Launch set is the Qwen2.5-Coder
family (Apache-2.0 end to end, one architecture, one tool-call parser) —
another family joins only after a real training run on it in this repo.

VRAM numbers are estimates from parameter counts, stated as such and kept
deliberately conservative so a plan that says "fits" fits:

- serve (bf16 weights + KV cache headroom): params x 2 bytes x 1.25
- QLoRA (4-bit base + LoRA + optimizer + activations): params x 0.5 bytes + 0.35 x params x 2 bytes + 2 GB
- LoRA on bf16 (bf16 base frozen + adapter + optimizer + activations): params x 2 bytes x 1.6 + 2 GB

The catalog is data, not behaviour; src/finetuning/planner.py turns an
entry plus detected hardware into a plan.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOOL_PARSER = "hermes"  # vLLM's parser for Qwen2.5's tool-call format


@dataclass(frozen=True)
class CatalogEntry:
    hf_id: str
    params_b: float
    license: str
    context: int
    tool_parser: str | None = DEFAULT_TOOL_PARSER
    moe: bool = False
    default_lora_r: int = 8
    notes: str = ""
    vision: bool = False

    @property
    def short_name(self) -> str:
        return self.hf_id.split("/", 1)[-1]

    @property
    def vram_serve_gb(self) -> float:
        return round(self.params_b * 2 * 1.25, 1)

    @property
    def vram_qlora_gb(self) -> float:
        return round(self.params_b * 0.5 + 0.35 * self.params_b * 2 + 2, 1)

    @property
    def vram_lora_bf16_gb(self) -> float:
        return round(self.params_b * 2 * 1.6 + 2, 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "hf_id": self.hf_id,
            "short_name": self.short_name,
            "params_b": self.params_b,
            "license": self.license,
            "context": self.context,
            "tool_parser": self.tool_parser,
            "moe": self.moe,
            "default_lora_r": self.default_lora_r,
            "vram_serve_gb": self.vram_serve_gb,
            "vram_qlora_gb": self.vram_qlora_gb,
            "vram_lora_bf16_gb": self.vram_lora_bf16_gb,
            "notes": self.notes,
            "vision": self.vision,
        }


_QWEN_CODER = "Apache-2.0"

CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct", 0.5, _QWEN_CODER, 32_768, notes="CI/laptop smoke model; not for quality"
    ),
    CatalogEntry("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, _QWEN_CODER, 32_768, notes="autocomplete-class"),
    CatalogEntry("Qwen/Qwen2.5-Coder-3B-Instruct", 3.0, _QWEN_CODER, 32_768),
    CatalogEntry(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        7.0,
        _QWEN_CODER,
        131_072,
        notes="default SLM: one 24 GB GPU serves it and trains a QLoRA on it",
    ),
    CatalogEntry(
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        14.0,
        _QWEN_CODER,
        131_072,
        notes="one 24 GB GPU: AWQ serving or QLoRA training only",
    ),
    CatalogEntry(
        "Qwen/Qwen2.5-Coder-32B-Instruct", 32.0, _QWEN_CODER, 131_072, notes="coding_fallback role model; one 80 GB GPU"
    ),
)

# The role models a deployment may already serve — listed so the Model Library can show them
# alongside the SLMs; they are not fine-tune defaults. Qwen2.5-VL is the one vision-capable entry
# here: an operator can point a role's endpoint at it to accept image-attached tasks (see
# AgentTaskRequest.images), but it is a different architecture from the Qwen2.5-Coder family the
# fine-tune catalog is scoped to, so it is served-only, never a fine-tune target, until a real
# training run against this exact family happens in this repo (same rule CATALOG's docstring
# states for any new family).
ROLE_MODELS: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "zai-org/GLM-5.3-Flash",
        321.0,
        "MIT",
        1_048_576,
        tool_parser="glm45",
        moe=True,
        notes="primary coding role; 8x80 GB",
    ),
    CatalogEntry("deepseek-ai/DeepSeek-R1", 671.0, "MIT", 131_072, tool_parser=None, moe=True, notes="reasoning role"),
    CatalogEntry(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        7.0,
        _QWEN_CODER,
        32_768,
        tool_parser=None,
        vision=True,
        notes="optional vision-capable role for image-attached tasks (screenshots, mockups, "
        "diagrams); served-only, not a fine-tune target",
    ),
)


def default_slm() -> CatalogEntry:
    """The model a guided fine-tune picks when asked for "auto": the 7B coder — the largest that both serves
    and trains (QLoRA) on a single 24 GB GPU."""
    return find("Qwen/Qwen2.5-Coder-7B-Instruct")


def find(hf_id: str) -> CatalogEntry:
    for entry in (*CATALOG, *ROLE_MODELS):
        if entry.hf_id == hf_id:
            return entry
    raise KeyError(f"{hf_id!r} is not in the model catalog; known: {[e.hf_id for e in CATALOG]}")


def is_known(hf_id: str) -> bool:
    return any(e.hf_id == hf_id for e in (*CATALOG, *ROLE_MODELS))


def largest_that_trains_on(vram_gb: float) -> CatalogEntry | None:
    """The biggest catalog SLM whose QLoRA footprint fits `vram_gb` on one GPU, or None."""
    fitting = [e for e in CATALOG if e.vram_qlora_gb <= vram_gb]
    return max(fitting, key=lambda e: e.params_b) if fitting else None
