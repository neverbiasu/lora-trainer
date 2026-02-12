"""LoRA implementation for Stable Diffusion fine-tuning."""
from typing import Optional, cast
import torch
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors

class LoRAModule(nn.Module):
    """Single LoRA layer: ΔW = scale * up(down(x)), scale = alpha / rank."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

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

    def _is_attention_module(self, name: str, module: nn.Module) -> bool:
        """Identify UNet attention modules for LoRA injection."""
        return "attn" in name and isinstance(module, nn.Linear)

    def prepare(self, model: nn.Module) -> None:
        """Inject LoRA into UNet attention layers."""
        for name, module in model.named_modules():
            if self._is_attention_module(name, module):
                linear_module = cast(nn.Linear, module)
                lora_module = LoRAModule(
                    in_features=linear_module.in_features,
                    out_features=linear_module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                )
                self.lora_modules[name] = lora_module
                
                linear_module.requires_grad_(False)
                
                def _make_forward_hook(lora: LoRAModule):
                    def _forward_hook(module, input, output):
                        return output + lora(input[0])
                    return _forward_hook
                
                linear_module.register_forward_hook(_make_forward_hook(lora_module))

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
