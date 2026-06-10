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

    # -- Visual diff criteria (only checked when present in metrics) ----------

    min_pixel_mae = validation_cfg.get("min_pixel_mae")
    if min_pixel_mae is not None:
        pixel_mae = metrics.get("mean_pixel_mae")
        if pixel_mae is not None and float(pixel_mae) < float(min_pixel_mae):
            reasons.append(
                "mean_pixel_mae={:.4f} is below min_pixel_mae={:.4f}; "
                "LoRA may not have changed model output".format(
                    float(pixel_mae), float(min_pixel_mae)
                )
            )

    min_delta_clip = validation_cfg.get("min_delta_clip")
    if min_delta_clip is not None:
        delta_clip = metrics.get("delta_clip")
        if delta_clip is not None and float(delta_clip) < float(min_delta_clip):
            reasons.append(
                "delta_clip={:.4f} is below min_delta_clip={:.4f}; "
                "LoRA did not move outputs closer to the target concept".format(
                    float(delta_clip), float(min_delta_clip)
                )
            )

    return TrainingEffectivenessReport(passed=len(reasons) == 0, reasons=reasons)
