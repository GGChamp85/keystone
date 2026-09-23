# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the promotion verdict: did the adapter measurably beat the base model?

A fine-tune is only worth serving if it is better than what was already
there. The trainer (src/finetuning/lora_train.py) evaluates the untouched
base model on the held-out split BEFORE training and the adapter on the
same split AFTER, so every completed job carries `base_eval_loss` and
`eval_loss` for identical data. The verdict is the comparison:

- **pass** — the adapter's held-out loss is lower than the base model's by
  at least `MIN_RELATIVE_IMPROVEMENT` (1 %), so the gain is not noise.
- **fail** — it is not.
- **unknown** — the job has no held-out evaluation (no holdout split, or
  an older job trained before the base eval existed).

`promote_job` (src/api/routes/finetune.py) refuses a `fail` or `unknown`
verdict unless an admin forces it, and records the force in the audit log.
Pure and unit-tested; the numbers come from the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_RELATIVE_IMPROVEMENT = 0.01


@dataclass(frozen=True)
class Verdict:
    status: str  # "pass" | "fail" | "unknown"
    reason: str
    base_eval_loss: float | None
    eval_loss: float | None
    relative_improvement: float | None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "base_eval_loss": self.base_eval_loss,
            "eval_loss": self.eval_loss,
            "relative_improvement": self.relative_improvement,
            "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        }


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def compute_verdict(metrics: dict[str, Any] | None) -> Verdict:
    metrics = metrics or {}
    base = _as_float(metrics.get("base_eval_loss"))
    adapter = _as_float(metrics.get("eval_loss"))
    if base is None or adapter is None:
        missing = "base_eval_loss" if base is None else "eval_loss"
        return Verdict(
            "unknown",
            f"no held-out comparison: {missing} is missing (train with a holdout split so the base model and the "
            "adapter are evaluated on the same data)",
            base,
            adapter,
            None,
        )
    if base <= 0:
        return Verdict("unknown", "base_eval_loss is not a positive number", base, adapter, None)
    improvement = (base - adapter) / base
    if improvement >= MIN_RELATIVE_IMPROVEMENT:
        return Verdict(
            "pass",
            f"held-out loss {adapter:.4f} vs base {base:.4f}: {improvement:.1%} better "
            f"(threshold {MIN_RELATIVE_IMPROVEMENT:.0%})",
            base,
            adapter,
            round(improvement, 4),
        )
    return Verdict(
        "fail",
        f"held-out loss {adapter:.4f} vs base {base:.4f}: {improvement:.1%} — not better than the base model by "
        f"{MIN_RELATIVE_IMPROVEMENT:.0%}",
        base,
        adapter,
        round(improvement, 4),
    )
