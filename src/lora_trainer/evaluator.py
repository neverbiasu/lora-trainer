# src/lora_trainer/evaluator.py
"""TrainingEvaluator — pixel diff, CLIPScore, and comparison sheet generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _list_images(directory: Path) -> dict[str, Path]:
    """Return {filename: path} for image files in directory."""
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


@dataclass
class EvaluationReport:
    """Complete evaluation report for a training run."""

    mean_pixel_mae: float
    mean_pixel_mse: float
    baseline_clip_sim: float | None = None
    lora_clip_sim: float | None = None
    delta_clip: float | None = None
    comparison_sheet_path: Path | None = None
    per_image_details: list[dict[str, Any]] = field(default_factory=list)


class TrainingEvaluator:
    """Evaluate training effectiveness via pixel diff and CLIPScore."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip_model: Any = None
        self._clip_processor: Any = None

    # -- Pixel diff ----------------------------------------------------------

    def compute_pixel_diff(
        self, baseline_dir: Path, final_dir: Path
    ) -> tuple[float, float]:
        """Compute mean MAE and MSE between matched images in two directories.

        Returns (mean_mae, mean_mse). Only files with the same name are compared.
        Returns (0.0, 0.0) if no matching files exist.
        """
        baseline_images = _list_images(baseline_dir)
        final_images = _list_images(final_dir)
        names = sorted(set(baseline_images) & set(final_images))

        if not names:
            return 0.0, 0.0

        mae_values: list[float] = []
        mse_values: list[float] = []

        for name in names:
            ref = np.asarray(
                Image.open(baseline_images[name]).convert("RGB"), dtype=np.float32
            )
            cand = np.asarray(
                Image.open(final_images[name]).convert("RGB"), dtype=np.float32
            )
            # Handle size mismatch by cropping to the smaller dimensions
            min_h = min(ref.shape[0], cand.shape[0])
            min_w = min(ref.shape[1], cand.shape[1])
            ref = ref[:min_h, :min_w, :]
            cand = cand[:min_h, :min_w, :]

            diff = ref - cand
            mae_values.append(float(np.mean(np.abs(diff))))
            mse_values.append(float(np.mean(diff * diff)))

        return float(np.mean(mae_values)), float(np.mean(mse_values))

    # -- Comparison sheet ----------------------------------------------------

    def create_comparison_sheet(
        self,
        baseline_dir: Path,
        final_dir: Path,
        output_path: Path,
        max_pairs: int = 9,
    ) -> Path:
        """Create a side-by-side baseline vs final comparison grid.

        Raises FileNotFoundError if no matching filenames exist.
        """
        baseline_images = _list_images(baseline_dir)
        final_images = _list_images(final_dir)
        names = sorted(set(baseline_images) & set(final_images))[:max_pairs]

        if not names:
            raise FileNotFoundError(
                "No matching image filenames between baseline and final directories"
            )

        panels: list[Image.Image] = []
        for name in names:
            ref = Image.open(baseline_images[name]).convert("RGB")
            cand = Image.open(final_images[name]).convert("RGB")
            # Resize to match if needed
            if ref.size != cand.size:
                size = (min(ref.width, cand.width), min(ref.height, cand.height))
                ref = ref.resize(size, Image.LANCZOS)
                cand = cand.resize(size, Image.LANCZOS)

            label_height = 28
            width = ref.width + cand.width
            height = max(ref.height, cand.height) + label_height
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(ref, (0, label_height))
            canvas.paste(cand, (ref.width, label_height))
            panels.append(canvas)

        sheet_width = max(p.width for p in panels)
        sheet_height = sum(p.height for p in panels)
        sheet = Image.new("RGB", (sheet_width, sheet_height), "#f5f5f5")

        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        logger.info("Saved comparison sheet: %s (%d pairs)", output_path, len(names))
        return output_path
