"""Config manager - handles YAML config loading, validation, merging"""
import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from lora_trainer.errors import InvalidConfigError, MissingRequiredFieldError

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "model": {
        "base_model": None,
    },
    "data": {
        "dataset_path": None,
        "resolution": 512,
        "cache_latents": "auto",
        "enable_bucketing": True,
    },
    "lora": {
        "rank": 32,
        "alpha": 32,
    },
    "training": {
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "batch_size": 4,
        "gradient_accumulation": 1,
        "max_train_steps": None,
        "mixed_precision": "fp16",
        "enable_xformers": False,
        "gradient_checkpointing": False,
        "save_every_n_steps": 500,
        "sample_every_n_steps": 250,
        "seed": 42,
    },
    "output": {
        "output_dir": "./output",
    },
}

# CLI arg name -> (yaml section, yaml key)
_CLI_TO_YAML: dict[str, tuple[str, str]] = {
    "dataset": ("data", "dataset_path"),
    "resolution": ("data", "resolution"),
    "cache_latents": ("data", "cache_latents"),
    "no_bucketing": ("data", "enable_bucketing"),
    "base_model": ("model", "base_model"),
    "rank": ("lora", "rank"),
    "alpha": ("lora", "alpha"),
    "learning_rate": ("training", "learning_rate"),
    "lr_scheduler": ("training", "lr_scheduler"),
    "batch_size": ("training", "batch_size"),
    "gradient_accumulation": ("training", "gradient_accumulation"),
    "max_steps": ("training", "max_train_steps"),
    "seed": ("training", "seed"),
    "mixed_precision": ("training", "mixed_precision"),
    "enable_xformers": ("training", "enable_xformers"),
    "gradient_checkpointing": ("training", "gradient_checkpointing"),
    "output_dir": ("output", "output_dir"),
    "save_every_n_steps": ("training", "save_every_n_steps"),
    "sample_every_n_steps": ("training", "sample_every_n_steps"),
}

# Inverted boolean flags (CLI flag means opposite of YAML value)
_INVERTED_BOOLS: set[str] = {"no_bucketing"}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_nested(config: dict, section: str, key: str, default: Any = None) -> Any:
    """Safely get a nested value from config."""
    return config.get(section, {}).get(key, default)


class ConfigManager:
    """Configuration manager

    Resolves the final config by merging: DEFAULTS < YAML < CLI overrides.
    """

    def load_config(self, config_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text())
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise InvalidConfigError("Config file must contain a mapping at the top level")
        return raw

    def resolve(
        self,
        config_path: Path | None = None,
        args: argparse.Namespace | None = None,
    ) -> dict[str, Any]:
        """Build the final resolved config.

        Priority: DEFAULTS < YAML file < CLI overrides.
        """
        config = copy.deepcopy(DEFAULTS)

        if config_path is not None:
            yaml_config = self.load_config(config_path)
            config = deep_merge(config, yaml_config)

        if args is not None:
            cli_overrides = self.extract_cli_overrides(args)
            config = deep_merge(config, cli_overrides)

        return config

    @staticmethod
    def extract_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
        """Convert CLI args into nested dict structure for merging."""
        overrides: dict[str, Any] = {}
        args_dict = vars(args)

        for cli_name, (section, key) in _CLI_TO_YAML.items():
            value = args_dict.get(cli_name)
            if value is None:
                continue

            if cli_name in _INVERTED_BOOLS:
                value = not value

            overrides.setdefault(section, {})[key] = value

        return overrides

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration, return list of error messages."""
        errors: list[str] = []

        base_model = _get_nested(config, "model", "base_model")
        if not base_model:
            errors.append("model.base_model is required")

        dataset_path = _get_nested(config, "data", "dataset_path")
        if not dataset_path:
            errors.append("data.dataset_path is required")
        elif not Path(dataset_path).exists():
            errors.append(f"dataset_path not found: {dataset_path}")

        rank = _get_nested(config, "lora", "rank")
        if rank is not None:
            if not isinstance(rank, int) or rank < 8 or rank > 64:
                errors.append("lora.rank must be an integer in [8, 64]")

        alpha = _get_nested(config, "lora", "alpha")
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                errors.append("lora.alpha must be positive")

        max_steps = _get_nested(config, "training", "max_train_steps")
        if max_steps is not None:
            if not isinstance(max_steps, int) or max_steps <= 0:
                errors.append("training.max_train_steps must be a positive integer")

        lr = _get_nested(config, "training", "learning_rate")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("training.learning_rate must be positive")

        return errors
