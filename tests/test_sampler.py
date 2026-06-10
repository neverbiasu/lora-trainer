# tests/test_sampler.py
"""Tests for SampleGenerator."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.lora_trainer.sampler import SampleGenerator, SampleGrid


def test_sample_grid_defaults():
    grid = SampleGrid(prompts=["a cat"], seeds=[42])
    assert grid.num_inference_steps == 20
    assert grid.guidance_scale == 7.5
    assert grid.width == 512
    assert grid.height == 512


def test_generate_grid_creates_expected_files(tmp_path: Path):
    mock_adapter = MagicMock()
    # generate() returns a [3, 512, 512] tensor in [0, 1]
    mock_adapter.generate.return_value = torch.rand(3, 512, 512)

    grid = SampleGrid(
        prompts=["prompt one", "prompt two"],
        seeds=[42, 123],
    )
    gen = SampleGenerator(model_adapter=mock_adapter, grid=grid)
    paths = gen.generate_grid(tmp_path / "out")

    assert len(paths) == 4  # 2 prompts × 2 seeds
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"

    # Verify the adapter was called with correct args
    assert mock_adapter.generate.call_count == 4
    first_call = mock_adapter.generate.call_args_list[0]
    assert first_call.kwargs["prompt"] == "prompt one"
    assert first_call.kwargs["seed"] == 42


def test_generate_grid_empty_prompts_raises():
    mock_adapter = MagicMock()
    grid = SampleGrid(prompts=[], seeds=[42])
    gen = SampleGenerator(model_adapter=mock_adapter, grid=grid)
    with pytest.raises(ValueError, match="at least one prompt"):
        gen.generate_grid(Path("/tmp/unused"))
