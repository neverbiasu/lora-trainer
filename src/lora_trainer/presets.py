"""Presets - quick/balanced/quality three-tier preset definitions"""

from typing import Any, Dict

# Preset definitions
PRESETS = {
    "quick": {
        "lora": {
            "rank": 16,
            "alpha": 16,
        },
        "training": {
            "learning_rate": 2e-4,
            "lr_scheduler": "constant",
            "batch_size": 4,
            "gradient_accumulation": 1,
            "max_train_steps": 1000,
        },
        "data": {
            "cache_latents": True,
        },
    },
    "balanced": {
        "lora": {
            "rank": 32,
            "alpha": 32,
        },
        "training": {
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine",
            "batch_size": 4,
            "gradient_accumulation": 1,
            "max_train_steps": 1500,
        },
        "data": {
            "cache_latents": "auto",
        },
    },
    "quality": {
        "lora": {
            "rank": 64,
            "alpha": 64,
        },
        "training": {
            "learning_rate": 5e-5,
            "lr_scheduler": "cosine",
            "batch_size": 2,
            "gradient_accumulation": 2,
            "max_train_steps": 3000,
        },
        "data": {
            "cache_latents": "auto",
        },
    },
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get preset configuration

    Args:
        preset_name: Preset name ("quick" / "balanced" / "quality")

    Returns:
        Preset configuration dictionary
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    return PRESETS[preset_name].copy()


def calculate_max_steps(dataset_size: int, preset: str, batch_size: int = 4) -> int:
    """
    Adaptively calculate training steps based on dataset size

    Args:
        dataset_size: Number of images in dataset
        preset: Preset name
        batch_size: batch size

    Returns:
        Recommended training steps
    """
    # TODO: implement adaptive step calculation
    raise NotImplementedError
