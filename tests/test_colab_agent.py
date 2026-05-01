"""Tests for Colab automation agent helpers."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.lora_trainer.colab_agent import (
    apply_trigger_token,
    compare_image_dirs,
    create_comparison_sheet,
    extract_log_highlights,
    validate_image_caption_pairs,
)


def test_validate_image_caption_pairs_detects_missing_caption(tmp_path: Path) -> None:
    """Validation should count images and detect missing .txt pair."""
    (tmp_path / "a.png").write_bytes(b"fake")
    (tmp_path / "a.txt").write_text("caption", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"fake")

    result = validate_image_caption_pairs(tmp_path)

    assert result.image_count == 2
    assert result.missing_caption_count == 1
    assert any(path.endswith("b.jpg") for path in result.missing_caption_files)


def test_apply_trigger_token_updates_only_non_prefixed_files(tmp_path: Path) -> None:
    """Trigger token should be prepended once per caption file."""
    (tmp_path / "x.txt").write_text("1girl", encoding="utf-8")
    (tmp_path / "y.txt").write_text("f3rn_char, portrait", encoding="utf-8")

    updated = apply_trigger_token(tmp_path, "f3rn_char")

    assert updated == 1
    assert (tmp_path / "x.txt").read_text(encoding="utf-8").startswith("f3rn_char")
    assert (tmp_path / "y.txt").read_text(encoding="utf-8") == "f3rn_char, portrait"


def test_compare_image_dirs_returns_metrics(tmp_path: Path) -> None:
    """Image comparison should return non-zero pair count and finite metrics."""
    ref_dir = tmp_path / "ref"
    cand_dir = tmp_path / "cand"
    ref_dir.mkdir()
    cand_dir.mkdir()

    ref_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cand_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cand_img[:, :, 0] = 10

    Image.fromarray(ref_img).save(ref_dir / "sample.png")
    Image.fromarray(cand_img).save(cand_dir / "sample.png")

    summary = compare_image_dirs(ref_dir, cand_dir)

    assert summary.matched_pairs == 1
    assert summary.mean_mae > 0
    assert summary.mean_mse > 0


def test_create_comparison_sheet_builds_image(tmp_path: Path) -> None:
    """Comparison sheet should be written when matching images exist."""
    ref_dir = tmp_path / "ref"
    cand_dir = tmp_path / "cand"
    out_path = tmp_path / "comparison.png"
    ref_dir.mkdir()
    cand_dir.mkdir()

    ref_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cand_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cand_img[:, :, 1] = 255
    Image.fromarray(ref_img).save(ref_dir / "sample.png")
    Image.fromarray(cand_img).save(cand_dir / "sample.png")

    result = create_comparison_sheet(ref_dir, cand_dir, out_path, max_pairs=4)

    assert result == out_path
    assert out_path.exists()


def test_extract_log_highlights_picks_relevant_lines(tmp_path: Path) -> None:
    """Log highlight extraction should keep the useful training lines."""
    log_path = tmp_path / "train.log"
    log_path.write_text(
        """
2026-04-26 - src.lora_trainer.trainer - INFO - Precision config: mixed_precision=fp16 autocast=True amp_dtype=torch.float16 scaler=True device=cuda
2026-04-26 - src.lora_trainer.trainer - INFO - Dataset ready: path=/tmp/data batch_size=1 resolution=512 batches=27
2026-04-26 - src.lora_trainer.trainer - INFO - Training summary: steps=1000 first_loss=0.9 final_loss=0.4 loss_ratio=0.44 lora_delta_l2=1.2 lora_delta_mean_abs=0.0001 effectiveness=True
2026-04-26 - src.lora_trainer.trainer - WARNING - Training effectiveness reasons: []
""".strip(),
        encoding="utf-8",
    )

    highlights = extract_log_highlights(log_path)

    assert "Precision config" in highlights
    assert "Training summary" in highlights
