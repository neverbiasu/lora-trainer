"""RunManager tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.lora_trainer.run_manager import RunManager


def _build_config(output_dir: Path) -> dict:
    return {
        "model": {"base_model": "sd15"},
        "lora": {"algorithm": "lora"},
        "training": {"seed": 42},
        "export": {"output_dir": str(output_dir)},
        "_config_path": "examples/config_basic.yaml",
        "_config_version": "0.1.0",
    }


def test_start_creates_run_directory_and_snapshot(tmp_path: Path) -> None:
    """Start should create run folders and config snapshot."""
    manager = RunManager(output_dir=tmp_path)
    config = _build_config(output_dir=tmp_path)

    run_dir = manager.start(config)

    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    assert (run_dir / "logs").exists()
    assert (run_dir / "checkpoints").exists()
    assert (run_dir / "samples").exists()
    assert (run_dir / "export").exists()

    snapshot_path = run_dir / "config_snapshot.yaml"
    assert snapshot_path.exists()

    snapshot = yaml.safe_load(snapshot_path.read_text())
    assert snapshot["model"]["base_model"] == "sd15"
    assert snapshot["_snapshot_meta"]["original_config_path"] == "examples/config_basic.yaml"
    assert snapshot["_snapshot_meta"]["config_version"] == "0.1.0"


def test_save_config_snapshot_requires_start(tmp_path: Path) -> None:
    """Snapshot save should fail before start initializes run_dir."""
    manager = RunManager(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match=r"call start\(\) first"):
        manager.save_config_snapshot(_build_config(output_dir=tmp_path))


def test_end_writes_metadata_json(tmp_path: Path) -> None:
    """End should write metadata file with training metrics."""
    manager = RunManager(output_dir=tmp_path)
    run_dir = manager.start(_build_config(output_dir=tmp_path))

    manager.end({"loss": 0.1})

    metadata_path = run_dir / "metadata.json"
    assert metadata_path.exists()
    metadata = metadata_path.read_text()
    assert '"training_metrics": {' in metadata
    assert '"loss": 0.1' in metadata
