"""LoRA implementation for Stable Diffusion fine-tuning."""
import logging
from typing import Any, Optional, cast

import torch
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)

class LoRAModule(nn.Module):
    """Single LoRA layer: ΔW = scale * up(down(x)), scale = alpha / rank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        org_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        self.device = cast(torch.device, org_module.weight.device) if org_module is not None else torch.device("cpu")
        self.dtype = cast(torch.dtype, org_module.weight.dtype) if org_module is not None else torch.float32
        self.to(device=self.device, dtype=self.dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=0)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.lora_up(self.lora_down(x))


class LoRAAdapter(nn.Module):
    """Manages LoRA injection and training for Stable Diffusion models."""

    def __init__(self, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_modules = nn.ModuleDict()
        self.hook_handles: dict[str, Any] = {}
        self.default_target_modules = ("to_q", "to_k", "to_v", "to_out.0")

    def _is_target_module(self, name: str, module: nn.Module, target_modules: tuple[str, ...]) -> bool:
        """Check whether a module should receive LoRA injection."""
        if not isinstance(module, nn.Linear):
            return False
        return any(name.endswith(pattern) or f".{pattern}" in name for pattern in target_modules)

    def _inject_into_model(
        self,
        model: nn.Module,
        target_modules: tuple[str, ...],
        strict: bool,
    ) -> dict[str, Any]:
        """Inject LoRA modules into a model and return an injection report."""
        report = {
            "injected_count": 0,
            "skipped_count": 0,
            "injected_modules": [],
            "skipped_modules": [],
        }

        for name, module in model.named_modules():
            if not self._is_target_module(name, module, target_modules):
                continue

            if name in self.hook_handles:
                report["skipped_count"] += 1
                report["skipped_modules"].append({"name": name, "reason": "already_injected"})
                continue

            linear_module = cast(nn.Linear, module)
            lora_module = LoRAModule(
                in_features=linear_module.in_features,
                out_features=linear_module.out_features,
                rank=self.rank,
                alpha=self.alpha,
                org_module=linear_module,
            )
            lora_key = name.replace(".", "_")
            self.lora_modules[lora_key] = lora_module
            linear_module.requires_grad_(False)

            def _make_forward_hook(lora: LoRAModule):
                def _forward_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
                    if len(inputs) == 0:
                        return output
                    return output + lora(inputs[0])

                return _forward_hook

            handle = linear_module.register_forward_hook(_make_forward_hook(lora_module))
            self.hook_handles[name] = handle

            report["injected_count"] += 1
            report["injected_modules"].append(name)

        if strict and report["injected_count"] == 0:
            raise ValueError(f"No target modules were injected. target_modules={target_modules}")

        return report

    def apply_to(
        self,
        text_encoder: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = False,
        apply_unet: bool = True,
        target_modules: Optional[list[str]] = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Reference-style API compatible with sd-scripts semantics."""
        target_patterns = tuple(target_modules or self.default_target_modules)
        result: dict[str, Any] = {
            "text_encoder": None,
            "unet": None,
            "total_injected": 0,
            "total_skipped": 0,
        }

        if apply_text_encoder and text_encoder is not None:
            report = self._inject_into_model(text_encoder, target_patterns, strict)
            result["text_encoder"] = report
            result["total_injected"] += report["injected_count"]
            result["total_skipped"] += report["skipped_count"]

        if apply_unet and unet is not None:
            report = self._inject_into_model(unet, target_patterns, strict)
            result["unet"] = report
            result["total_injected"] += report["injected_count"]
            result["total_skipped"] += report["skipped_count"]

        return result

    def load_weights(self, weights_path: str, strict: bool = False) -> None:
        """Load LoRA weights from a checkpoint (safetensors or torch).

        Keys in the checkpoint must match the format produced by export_weights,
        i.e. the full state_dict layout with 'lora_modules.' prefix.
        """
        if weights_path.endswith(".safetensors"):
            state_dict = dict(load_safetensors(weights_path))
        elif weights_path.endswith((".pt", ".pth", ".bin", ".ckpt")):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            state_dict = dict(checkpoint)
        else:
            raise ValueError(
                f"Unsupported checkpoint format: '{weights_path}'. "
                "Expected .safetensors, .pt, .pth, .bin, or .ckpt."
            )

        incompatible = self.load_state_dict(state_dict, strict=strict)
        if incompatible.missing_keys:
            logger.warning(
                "load_weights: missing keys in checkpoint: %s",
                incompatible.missing_keys,
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "load_weights: unexpected keys in checkpoint: %s",
                incompatible.unexpected_keys,
            )

    def remove_injection(self) -> None:
        """Remove all forward hooks registered during LoRA injection."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return all LoRA parameters for the optimizer."""
        return list(self.parameters())

    def export_weights(self, save_path: str, metadata: Optional[dict] = None) -> None:
        """Export LoRA weights to safetensors or torch format.
        
        Reference: sd-scripts networks/lora.py:1255-1283 (save_weights).
        Supports both .safetensors and .pt formats.
        """
        state_dict = self.state_dict()
        
        if save_path.endswith(".safetensors"):
            if metadata is None:
                metadata = {}
            save_safetensors(state_dict, save_path, metadata=metadata)
        else:
            torch.save(state_dict, save_path)
