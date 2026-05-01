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
