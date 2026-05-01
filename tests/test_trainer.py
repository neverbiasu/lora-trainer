"""Trainer lifecycle and orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.lora_trainer.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(output_dir: Path | None = None, **overrides: Any) -> dict:
    cfg: dict = {
        "model": {"base_model": "sd15"},
        "lora": {"rank": 4, "alpha": 4.0, "apply_text_encoder": False},
        "training": {
            "seed": 42,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "max_train_steps": 2,
            "save_every_n_steps": 100,
            "gradient_accumulation": 1,
            "lr_scheduler": "constant",
        },
        "data": {"dataset_path": "/fake/dataset", "resolution": 64},
        "export": {"output_dir": str(output_dir or "/tmp/output")},
        "_config_path": None,
        "_config_version": "0.1",
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _make_fake_batch() -> tuple[torch.Tensor, list[str]]:
    return torch.zeros(1, 3, 64, 64), ["test caption"]


# Fake data loader that yields a fixed number of batches
class _FakeDataLoader:
    def __init__(self, num_batches: int = 3) -> None:
        self._n = num_batches

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        for _ in range(self._n):
            yield _make_fake_batch()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model_adapter():
    adapter = MagicMock()
    adapter.load_models.return_value = (
        MagicMock(spec=nn.Module),  # vae
        MagicMock(spec=nn.Module),  # unet
        MagicMock(spec=nn.Module),  # text_encoder
    )
    latents = torch.zeros(1, 4, 8, 8)
    adapter.encode_image.return_value = latents
    adapter.encode_prompt.return_value = torch.zeros(1, 77, 768)
    # unet forward — adapter.unet will be set on model_adapter
    return adapter


@pytest.fixture()
def mock_lora_adapter():
    adapter = MagicMock()
    adapter.apply_to.return_value = {
        "injected_count": 3,
        "skipped_count": 0,
        "injected_modules": [],
        "skipped_modules": [],
    }
    adapter.get_trainable_params.return_value = [torch.zeros(4, 4, requires_grad=True)]
    adapter.state_dict.return_value = {"lora_down": torch.zeros(4, 4)}
    return adapter


@pytest.fixture()
def mock_run_manager(tmp_path):
    manager = MagicMock()
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "export").mkdir()
    (run_dir / "samples").mkdir()
    manager.start.return_value = run_dir
    manager.run_dir = run_dir
    manager.save_checkpoint.return_value = run_dir / "checkpoints" / "step_1.safetensors"
    return manager


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_trainer_init_sets_defaults() -> None:
    """Trainer __init__ should populate fields with correct types."""
    cfg = _base_config()
    trainer = Trainer(cfg)

    assert trainer.global_step == 0
    assert trainer.last_loss == 0.0
    assert trainer.model_adapter is None
    assert trainer.run_manager is None


def test_configure_precision_enables_fp16_autocast_on_cuda() -> None:
    """fp16 config should enable autocast settings when CUDA is requested."""
    cfg = _base_config()
    cfg["training"]["mixed_precision"] = "fp16"
    trainer = Trainer(cfg)
    trainer.device = torch.device("cuda")

    trainer._configure_precision()

    assert trainer.mixed_precision_mode == "fp16"
    assert trainer.amp_autocast_enabled is True
    assert trainer.amp_dtype == torch.float16


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def test_start_raises_on_unsupported_base_model() -> None:
    """start() should raise ValueError for unknown base_model."""
    cfg = _base_config()
    cfg["model"] = {"base_model": "flux999"}
    trainer = Trainer(cfg)

    with pytest.raises((ValueError, Exception)):
        trainer.start()


# ---------------------------------------------------------------------------
# end() — guard before start
# ---------------------------------------------------------------------------


def test_end_raises_before_start() -> None:
    """end() must raise RuntimeError if start() was never called."""
    trainer = Trainer(_base_config())
    with pytest.raises(RuntimeError, match="call start\\(\\) first"):
        trainer.end()


def test_validate_raises_before_start() -> None:
    """validate() must raise RuntimeError if start() was never called."""
    cfg = _base_config(validation={"prompt": "test", "seed": 0, "every_n_steps": 10})
    trainer = Trainer(cfg)
    with pytest.raises(RuntimeError, match="call start\\(\\) first"):
        trainer.validate(0)


# ---------------------------------------------------------------------------
# validate() — early return without config
# ---------------------------------------------------------------------------


def test_validate_skips_when_no_validation_config() -> None:
    """validate() should return immediately when 'validation' key is absent."""
    trainer = Trainer(_base_config())
    trainer.model_adapter = MagicMock()
    trainer.run_manager = MagicMock()

    trainer.validate(0)

    trainer.model_adapter.generate.assert_not_called()


# ---------------------------------------------------------------------------
# save_checkpoint()
# ---------------------------------------------------------------------------


def test_save_checkpoint_writes_json_sidecar(
    tmp_path: Path,
    mock_lora_adapter,
    mock_run_manager,
) -> None:
    """save_checkpoint() should write a JSON sidecar with step/rank/alpha."""
    cfg = _base_config(tmp_path)
    trainer = Trainer(cfg)
    trainer.lora_adapter = mock_lora_adapter
    trainer.run_manager = mock_run_manager
    # Patch out safetensors write
    mock_run_manager.save_checkpoint.return_value = (
        tmp_path / "run_test" / "checkpoints" / "step_5.safetensors"
    )

    with patch("src.lora_trainer.trainer.json.dumps", wraps=json.dumps):
        trainer.save_checkpoint(5)

    json_path = tmp_path / "run_test" / "checkpoints" / "step_5.json"
    assert json_path.exists()
    meta = json.loads(json_path.read_text())
    assert meta["global_step"] == 5
    assert meta["rank"] == 4
    assert meta["alpha"] == 4.0


# ---------------------------------------------------------------------------
# full train() lifecycle with mocks
# ---------------------------------------------------------------------------


@patch("src.lora_trainer.trainer.create_data_loader")
@patch("src.lora_trainer.trainer.LoRAAdapter")
@patch("src.lora_trainer.trainer.SD15ModelAdapter")
@patch("src.lora_trainer.trainer.RunManager")
def test_train_lifecycle_calls_end(
    mock_run_manager_cls,
    mock_sd15_cls,
    mock_lora_cls,
    mock_create_dl,
    tmp_path: Path,
) -> None:
    """train() must always call end() via try/finally."""
    run_dir = tmp_path / "run_xyz"
    run_dir.mkdir(parents=True)
    (run_dir / "export").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()

    # RunManager mock
    rm = MagicMock()
    rm.start.return_value = run_dir
    rm.run_dir = run_dir
    rm.save_checkpoint.return_value = run_dir / "checkpoints" / "step_0.safetensors"
    mock_run_manager_cls.return_value = rm

    # SD15 adapter mock
    sd = MagicMock()
    unet_mock = MagicMock(spec=nn.Module)
    unet_sample = MagicMock()
    unet_sample.sample = torch.zeros(1, 4, 8, 8)
    unet_mock.return_value = unet_sample
    sd.load_models.return_value = (MagicMock(spec=nn.Module), unet_mock, MagicMock(spec=nn.Module))
    sd.encode_image.return_value = torch.zeros(1, 4, 8, 8)
    sd.encode_prompt.return_value = torch.zeros(1, 77, 768)
    mock_sd15_cls.return_value = sd

    # LoRA adapter mock
    lora = MagicMock()
    lora.apply_to.return_value = {
        "injected_count": 3,
        "skipped_count": 0,
        "injected_modules": [],
        "skipped_modules": [],
    }
    lora.get_trainable_params.return_value = [torch.zeros(4, 4, requires_grad=True)]
    lora.state_dict.return_value = {}
    mock_lora_cls.return_value = lora

    # DataLoader mock
    mock_create_dl.return_value = _FakeDataLoader(num_batches=3)

    cfg = _base_config(tmp_path)
    trainer = Trainer(cfg)

    # patch train_step to avoid real gradient computation
    with patch.object(trainer, "train_step", return_value=0.5):
        trainer.train()

    # end() should have been called → run_manager.end() called
    rm.end.assert_called_once()
    # lora.export_weights called once in end()
    lora.export_weights.assert_called_once()
    # export_weights was called with metadata kwarg
    _, kwargs = lora.export_weights.call_args
    assert "metadata" in kwargs
    assert kwargs["metadata"].get("rank") == "4"
