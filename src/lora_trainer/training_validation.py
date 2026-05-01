"""Utilities for post-training effectiveness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingEffectivenessReport:
    """Result of post-training effectiveness evaluation."""

    passed: bool
    reasons: list[str]


def evaluate_training_effectiveness(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> TrainingEffectivenessReport:
    """Evaluate whether a training run meets effectiveness thresholds."""
    validation_cfg = config.get("validation", {})

    min_effective_steps = int(validation_cfg.get("min_effective_steps", 100))
    min_lora_delta_l2 = float(validation_cfg.get("min_lora_delta_l2", 1e-6))
    max_loss_ratio = float(validation_cfg.get("max_loss_ratio", 1.2))
    require_loss_drop = bool(validation_cfg.get("require_loss_drop", False))

    reasons: list[str] = []

    total_steps = int(metrics.get("total_steps", 0) or 0)
    if total_steps < min_effective_steps:
        reasons.append(
            f"total_steps={total_steps} is lower than min_effective_steps={min_effective_steps}"
        )

    lora_delta_l2 = metrics.get("lora_delta_l2")
    if lora_delta_l2 is None:
        reasons.append("lora_delta_l2 is unavailable; cannot verify parameter updates")
    elif float(lora_delta_l2) < min_lora_delta_l2:
        reasons.append(
            "lora_delta_l2={:.3e} is lower than min_lora_delta_l2={:.3e}".format(
                float(lora_delta_l2),
                min_lora_delta_l2,
            )
        )

    first_loss = metrics.get("first_loss")
    final_loss = metrics.get("final_loss")
    loss_ratio = metrics.get("loss_ratio")

    if require_loss_drop:
        if first_loss is None or final_loss is None:
            reasons.append("first_loss/final_loss unavailable while require_loss_drop=true")
        elif float(final_loss) >= float(first_loss):
            reasons.append(
                "final_loss={:.6f} did not drop below first_loss={:.6f}".format(
                    float(final_loss),
                    float(first_loss),
                )
            )

    if first_loss is not None and final_loss is not None and float(first_loss) > 0:
        current_ratio = float(final_loss) / float(first_loss)
        if current_ratio > max_loss_ratio:
            reasons.append(
                "loss_ratio={:.6f} exceeded max_loss_ratio={:.6f}".format(
                    current_ratio,
                    max_loss_ratio,
                )
            )
    elif loss_ratio is not None and float(loss_ratio) > max_loss_ratio:
        reasons.append(
            "loss_ratio={:.6f} exceeded max_loss_ratio={:.6f}".format(
                float(loss_ratio),
                max_loss_ratio,
            )
        )

    return TrainingEffectivenessReport(passed=len(reasons) == 0, reasons=reasons)
