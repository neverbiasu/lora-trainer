"""AlgoAdapter - algorithm adapter (unified interface for LoRA/LoHA/LoKr)"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn


class AlgoAdapter(ABC):
    """Algorithm adapter base class"""
    
    @abstractmethod
    def prepare(self, model: nn.Module, config: Dict) -> None:
        """Preparation phase: inject LoRA into model"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""
        pass
    
    @abstractmethod
    def get_trainable_params(self) -> list:
        """Get trainable parameters"""
        pass
    
    @abstractmethod
    def export_weights(self, save_path: str, metadata: Optional[Dict] = None) -> None:
        """Export weights"""
        pass


class LoRAAdapter(AlgoAdapter):
    """Standard LoRA adapter (MVP implementation)"""
    
    def __init__(self, rank: int = 32, alpha: float = 32.0):
        self.rank = rank
        self.alpha = alpha
        self.lora_modules = {}
    
    def prepare(self, model: nn.Module, config: Dict) -> None:
        """Inject LoRA into UNet"""
        # TODO: implement LoRA injection
        raise NotImplementedError
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass (with LoRA)"""
        # TODO: implement forward pass
        raise NotImplementedError
    
    def get_trainable_params(self) -> list:
        """Get LoRA parameters"""
        # TODO: implement parameter retrieval
        raise NotImplementedError
    
    def export_weights(self, save_path: str, metadata: Optional[Dict] = None) -> None:
        """Export LoRA weights"""
        # TODO: implement weight export
        raise NotImplementedError
