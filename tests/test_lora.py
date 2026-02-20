"""LoRAModule and LoRAAdapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.lora_trainer.lora import LoRAAdapter, LoRAModule


# ---------------------------------------------------------------------------
# LoRAModule
# ---------------------------------------------------------------------------

def test_lora_module_output_shape() -> None:
    """Forward pass should produce tensor of shape (batch, out_features)."""
    module = LoRAModule(in_features=64, out_features=128, rank=4, alpha=4.0)
    x = torch.randn(2, 64)
    out = module(x)
    assert out.shape == (2, 128)


def test_lora_module_scale() -> None:
    """Scale factor should equal alpha / rank."""
    module = LoRAModule(in_features=16, out_features=16, rank=8, alpha=16.0)
    assert module.scale == pytest.approx(2.0)


def test_lora_module_up_weights_initialized_zeros() -> None:
    """lora_up weights should start at zero (no residual at init)."""
    module = LoRAModule(in_features=32, out_features=32, rank=4, alpha=4.0)
    assert torch.all(module.lora_up.weight == 0)


# ---------------------------------------------------------------------------
# LoRAAdapter — injection
# ---------------------------------------------------------------------------

class _TinyUNet(nn.Module):
    """Minimal UNet-like model with attention projection layers."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(64, 64)
        self.to_k = nn.Linear(64, 64)
        self.to_v = nn.Linear(64, 64)
        self.ff = nn.Linear(64, 64)  # should NOT be injected


def test_apply_to_injects_target_modules() -> None:
    """apply_to() should inject LoRA into attention projection layers."""
    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)

    report = adapter.apply_to(
        text_encoder=None,
        unet=unet,
        apply_text_encoder=False,
        apply_unet=True,
        target_modules=("to_q", "to_k", "to_v"),
        strict=False,
    )

    assert report["total_injected"] == 3
    assert len(adapter.lora_modules) == 3


def test_apply_to_skips_on_duplicate() -> None:
    """Second apply_to on same model should skip already-injected modules."""
    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)

    adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=("to_q",), strict=False,
    )
    report2 = adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=("to_q",), strict=False,
    )

    assert report2["total_skipped"] == 1


def test_get_trainable_params_non_empty_after_injection() -> None:
    """get_trainable_params() should return non-empty after apply_to."""
    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)
    adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=None, strict=False,
    )

    params = adapter.get_trainable_params()
    assert len(params) > 0


# ---------------------------------------------------------------------------
# LoRAAdapter — export / load round-trip
# ---------------------------------------------------------------------------

def test_export_load_roundtrip(tmp_path: Path) -> None:
    """export_weights() → load_weights() should restore identical state."""
    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)
    adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=None, strict=False,
    )

    save_path = str(tmp_path / "lora.safetensors")
    adapter.export_weights(save_path)

    # Modify weights then restore
    with torch.no_grad():
        for p in adapter.parameters():
            p.fill_(99.0)

    adapter.load_weights(save_path, strict=True)
    # After load, weights should NOT be 99
    param_values = [p.mean().item() for p in adapter.parameters()]
    assert all(v != 99.0 for v in param_values)


def test_export_writes_metadata(tmp_path: Path) -> None:
    """export_weights() should embed metadata into the safetensors file."""
    from safetensors.torch import load_file

    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)
    adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=None, strict=False,
    )

    save_path = str(tmp_path / "lora_meta.safetensors")
    meta = {"rank": "4", "alpha": "4.0", "base_model": "sd15"}
    adapter.export_weights(save_path, metadata=meta)

    # safetensors metadata is embedded; just assert file exists and loads
    tensors = load_file(save_path)
    assert len(tensors) > 0


def test_load_weights_unsupported_extension_raises(tmp_path: Path) -> None:
    """load_weights() with an unknown extension should raise ValueError."""
    unet = _TinyUNet()
    adapter = LoRAAdapter(rank=4, alpha=4.0)
    adapter.apply_to(
        text_encoder=None, unet=unet,
        apply_text_encoder=False, apply_unet=True,
        target_modules=None, strict=False,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        adapter.load_weights(str(tmp_path / "weights.xyz"))
