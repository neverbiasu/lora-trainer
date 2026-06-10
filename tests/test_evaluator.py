# tests/test_evaluator.py
"""Tests for TrainingEvaluator."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
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


def test_clip_similarity_returns_float(tmp_path: Path):
    """Test CLIPScore computation with mocked CLIP model."""
    gen_dir = tmp_path / "generated"
    ref_dir = tmp_path / "reference"
    _create_test_image(gen_dir / "p0_seed42.png", (128, 0, 0))
    _create_test_image(ref_dir / "img1.png", (128, 0, 0))
    _create_test_image(ref_dir / "img2.png", (130, 5, 5))

    evaluator = TrainingEvaluator(device="cpu")

    # Mock the CLIP model to return deterministic embeddings
    mock_features = torch.randn(1, 512)
    mock_features = mock_features / mock_features.norm(dim=-1, keepdim=True)

    with patch.object(evaluator, "_get_clip_image_embedding", return_value=mock_features[0]):
        sim = evaluator.compute_clip_similarity(
            generated_dir=gen_dir, reference_dir=ref_dir
        )

    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_evaluate_full_report(tmp_path: Path):
    """Test the full evaluate() pipeline produces an EvaluationReport."""
    baseline_dir = tmp_path / "baseline"
    final_dir = tmp_path / "final"
    dataset_dir = tmp_path / "dataset"
    _create_test_image(baseline_dir / "p0_seed42.png", (0, 0, 0))
    _create_test_image(final_dir / "p0_seed42.png", (255, 255, 255))
    _create_test_image(dataset_dir / "img1.png", (200, 200, 200))

    evaluator = TrainingEvaluator(device="cpu")

    mock_embedding = torch.randn(512)
    mock_embedding = mock_embedding / mock_embedding.norm()

    with patch.object(evaluator, "_get_clip_image_embedding", return_value=mock_embedding):
        report = evaluator.evaluate(
            baseline_dir=baseline_dir,
            final_dir=final_dir,
            dataset_path=str(dataset_dir),
            output_dir=tmp_path / "results",
        )

    assert report.mean_pixel_mae > 200  # black vs white
    assert report.baseline_clip_sim is not None
    assert report.lora_clip_sim is not None
    assert report.delta_clip is not None
    assert report.comparison_sheet_path is not None
    assert report.comparison_sheet_path.exists()
