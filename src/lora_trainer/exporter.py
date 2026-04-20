"""Exporter - LoRA weight export (ComfyUI/A1111 format)"""

from pathlib import Path
from typing import Dict, Optional


class Exporter:
    """Exporter"""

    def __init__(self, target_format: str = "comfyui"):
        """
        Args:
            target_format: Export format ("comfyui" / "a1111")
        """
        self.target_format = target_format

    def export(
        self,
        lora_state_dict: Dict,
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Export LoRA weights

        Args:
            lora_state_dict: LoRA weight dictionary
            save_path: Save path
            metadata: Metadata (base_model, rank, alpha, trigger, etc.)
        """
        # TODO: implement export logic
        raise NotImplementedError


class ComfyUIExporter(Exporter):
    """ComfyUI format exporter (MVP implementation)"""

    def __init__(self):
        super().__init__(target_format="comfyui")

    def export(
        self,
        lora_state_dict: Dict,
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Export to ComfyUI compatible format"""
        # TODO: implement ComfyUI export
        raise NotImplementedError
