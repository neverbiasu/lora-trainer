"""ConfigManager tests."""

from __future__ import annotations

import argparse

import pytest

from src.lora_trainer.config_manager import ConfigManager
from src.lora_trainer.errors import InvalidConfigError, MissingRequiredFieldError


def test_config_manager_init() -> None:
    """ConfigManager exposes default config version."""
    manager = ConfigManager()
    assert manager.config_version == "0.1.0"


def test_resolve_merges_defaults_yaml_and_cli(tmp_path) -> None:
    """Resolved config follows DEFAULTS < YAML < CLI precedence."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
config_version: "0.1.0"
model:
    base_model: sd15
data:
    dataset_path: ./dataset
training:
    learning_rate: 1e-4
""".strip()
    )

    args = argparse.Namespace(
        dataset=tmp_path / "cli_dataset",
        learning_rate=2e-4,
        no_bucketing=True,
    )

    manager = ConfigManager()
    config = manager.resolve(config_path=config_path, args=args)

    assert config["training"]["learning_rate"] == pytest.approx(2e-4)
    assert config["data"]["dataset_path"] == str(tmp_path / "cli_dataset")
    assert config["data"]["enable_bucketing"] is False
    assert config["training"]["batch_size"] == 4


def test_resolve_normalizes_numeric_and_path_types(tmp_path) -> None:
    """Resolve converts numeric-like strings and Path values."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
config_version: "0.1.0"
model:
    base_model: sd15
data:
    dataset_path: ./dataset
    resolution: "768"
lora:
    rank: "16"
training:
    learning_rate: "1e-4"
    max_train_steps: "1200"
output:
    output_dir: ./output
""".strip()
    )

    args = argparse.Namespace(dataset=tmp_path / "images")
    manager = ConfigManager()
    config = manager.resolve(config_path=config_path, args=args)

    assert isinstance(config["training"]["learning_rate"], float)
    assert config["training"]["learning_rate"] == pytest.approx(1e-4)
    assert isinstance(config["training"]["max_train_steps"], int)
    assert isinstance(config["lora"]["rank"], int)
    assert config["data"]["dataset_path"] == str(tmp_path / "images")


def test_validate_or_raise_missing_required_field() -> None:
    """Missing required fields raise MissingRequiredFieldError."""
    manager = ConfigManager()
    config = manager.resolve()

    with pytest.raises(MissingRequiredFieldError):
        manager.validate_or_raise(config)


def test_validate_or_raise_invalid_value(tmp_path) -> None:
    """Invalid value raises InvalidConfigError."""
    manager = ConfigManager()
    config = manager.resolve(
        args=argparse.Namespace(
            dataset=tmp_path,
            base_model="sd15",
            lr_scheduler="bad_scheduler",
        )
    )

    with pytest.raises(InvalidConfigError):
        manager.validate_or_raise(config)
