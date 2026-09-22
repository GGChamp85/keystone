# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Shared trainer plumbing for src/finetuning/*_train.py — the pieces that
depend on which transformers/trl versions are actually installed, resolved
by inspecting the real classes rather than by pinning the ecosystem back.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any


@cache
def _accepts(config_cls: type, parameter: str) -> bool:
    return parameter in inspect.signature(config_cls).parameters


def warmup_kwargs(ratio: float, config_cls: type) -> dict[str, Any]:
    """
    The warmup argument `config_cls` (TrainingArguments, or a trl subclass
    such as SFTConfig/DPOConfig) actually accepts for a warmup *ratio*.

    transformers 4.x has a separate `warmup_ratio`; 5.x removed it and
    made `warmup_steps` a float where a value in [0, 1) means "ratio of
    total training steps" — the same semantics under a different name.
    Passing `warmup_ratio` to 5.x is a TypeError, which is how the first
    real trainer run in this repository failed; this resolves the right
    name from the installed class's real signature.
    """
    if _accepts(config_cls, "warmup_ratio"):
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}
