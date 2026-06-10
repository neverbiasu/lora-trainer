# tests/test_evaluator.py
"""Tests for TrainingEvaluator."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.lora_trainer.evaluator import TrainingEvaluator


def _create_test_image(path: Path, color: tuple[int, int, int], size: int = 64) -> Path:
    """Helper: create a solid-color PNG."""
    img = Image.new("RGB", (size, size), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def test_pixel_diff_identical_images(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    final_dir = tmp_path / "final"
    _create_test_image(baseline_dir / "p0_seed42.png", (128, 0, 0))
    _create_test_image(final_dir / "p0_seed42.png", (128, 0, 0))

    evaluator = TrainingEvaluator(device="cpu")
    mae, mse = evaluator.compute_pixel_diff(baseline_dir, final_dir)
    assert mae == pytest.approx(0.0, abs=1e-6)
    assert mse == pytest.approx(0.0, abs=1e-6)


def test_pixel_diff_different_images(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    final_dir = tmp_path / "final"
    _create_test_image(baseline_dir / "p0_seed42.png", (0, 0, 0))
    _create_test_image(final_dir / "p0_seed42.png", (255, 255, 255))

    evaluator = TrainingEvaluator(device="cpu")
    mae, mse = evaluator.compute_pixel_diff(baseline_dir, final_dir)
    assert mae == pytest.approx(255.0, abs=1.0)
    assert mse > 60000  # 255^2 = 65025


def test_comparison_sheet_created(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    final_dir = tmp_path / "final"
    _create_test_image(baseline_dir / "p0_seed42.png", (128, 0, 0))
    _create_test_image(final_dir / "p0_seed42.png", (0, 0, 128))

    evaluator = TrainingEvaluator(device="cpu")
    out_path = tmp_path / "comparison.png"
    result = evaluator.create_comparison_sheet(baseline_dir, final_dir, out_path)
    assert result.exists()
    img = Image.open(result)
    assert img.width > 0 and img.height > 0


def test_comparison_sheet_no_matching_files(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    final_dir = tmp_path / "final"
    _create_test_image(baseline_dir / "a.png", (128, 0, 0))
    _create_test_image(final_dir / "b.png", (0, 0, 128))

    evaluator = TrainingEvaluator(device="cpu")
    with pytest.raises(FileNotFoundError, match="No matching"):
        evaluator.create_comparison_sheet(baseline_dir, final_dir, tmp_path / "out.png")
