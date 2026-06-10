"""Tests for post-training effectiveness checks."""

from src.lora_trainer.training_validation import evaluate_training_effectiveness


def test_effectiveness_passes_for_reasonable_metrics() -> None:
    """Expected metrics should pass all gate checks."""
    config = {
        "validation": {
            "min_effective_steps": 100,
            "min_lora_delta_l2": 1e-6,
            "max_loss_ratio": 1.2,
            "require_loss_drop": False,
        }
    }
    metrics = {
        "total_steps": 300,
        "first_loss": 0.8,
        "final_loss": 0.4,
        "loss_ratio": 0.5,
        "lora_delta_l2": 2e-4,
    }

    report = evaluate_training_effectiveness(metrics, config)

    assert report.passed is True
    assert report.reasons == []


def test_effectiveness_fails_when_steps_too_small() -> None:
    """Too-few steps should fail the quality gate."""
    config = {"validation": {"min_effective_steps": 200}}
    metrics = {
        "total_steps": 20,
        "lora_delta_l2": 1e-3,
        "first_loss": 0.6,
        "final_loss": 0.5,
    }

    report = evaluate_training_effectiveness(metrics, config)

    assert report.passed is False
    assert any("min_effective_steps" in reason for reason in report.reasons)


def test_effectiveness_fails_when_lora_delta_missing() -> None:
    """Missing LoRA delta signal should fail because update cannot be verified."""
    config = {"validation": {"min_effective_steps": 1}}
    metrics = {
        "total_steps": 10,
        "first_loss": 0.7,
        "final_loss": 0.6,
        "lora_delta_l2": None,
    }

    report = evaluate_training_effectiveness(metrics, config)

    assert report.passed is False
    assert any("lora_delta_l2" in reason for reason in report.reasons)


def test_effectiveness_fails_when_loss_ratio_exceeds_threshold() -> None:
    """Loss ratio over threshold should fail."""
    config = {
        "validation": {
            "min_effective_steps": 1,
            "min_lora_delta_l2": 0.0,
            "max_loss_ratio": 1.1,
        }
    }
    metrics = {
        "total_steps": 20,
        "first_loss": 0.5,
        "final_loss": 0.8,
        "lora_delta_l2": 1e-3,
    }

    report = evaluate_training_effectiveness(metrics, config)

    assert report.passed is False
    assert any("loss_ratio" in reason for reason in report.reasons)


def test_min_pixel_mae_fails_when_too_low():
    metrics = {
        "total_steps": 1000,
        "lora_delta_l2": 1.0,
        "first_loss": 0.1,
        "final_loss": 0.08,
        "loss_ratio": 0.8,
        "mean_pixel_mae": 0.001,  # below threshold
    }
    config = {
        "validation": {
            "min_pixel_mae": 0.02,
        }
    }
    report = evaluate_training_effectiveness(metrics, config)
    assert not report.passed
    assert any("pixel_mae" in r for r in report.reasons)


def test_min_pixel_mae_passes_when_above():
    metrics = {
        "total_steps": 1000,
        "lora_delta_l2": 1.0,
        "first_loss": 0.1,
        "final_loss": 0.08,
        "loss_ratio": 0.8,
        "mean_pixel_mae": 5.0,
    }
    config = {
        "validation": {
            "min_pixel_mae": 0.02,
        }
    }
    report = evaluate_training_effectiveness(metrics, config)
    assert report.passed


def test_min_delta_clip_fails_when_too_low():
    metrics = {
        "total_steps": 1000,
        "lora_delta_l2": 1.0,
        "first_loss": 0.1,
        "final_loss": 0.08,
        "loss_ratio": 0.8,
        "delta_clip": 0.001,  # below threshold
    }
    config = {
        "validation": {
            "min_delta_clip": 0.01,
        }
    }
    report = evaluate_training_effectiveness(metrics, config)
    assert not report.passed
    assert any("delta_clip" in r for r in report.reasons)


def test_min_delta_clip_passes_when_above():
    metrics = {
        "total_steps": 1000,
        "lora_delta_l2": 1.0,
        "first_loss": 0.1,
        "final_loss": 0.08,
        "loss_ratio": 0.8,
        "delta_clip": 0.05,
    }
    config = {
        "validation": {
            "min_delta_clip": 0.01,
        }
    }
    report = evaluate_training_effectiveness(metrics, config)
    assert report.passed


def test_new_criteria_skipped_when_not_configured():
    """When min_pixel_mae and min_delta_clip are not in config, they are not checked."""
    metrics = {
        "total_steps": 1000,
        "lora_delta_l2": 1.0,
        "first_loss": 0.1,
        "final_loss": 0.08,
        "loss_ratio": 0.8,
    }
    config = {"validation": {}}
    report = evaluate_training_effectiveness(metrics, config)
    assert report.passed
