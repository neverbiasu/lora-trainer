# src/lora_trainer/sampler.py
"""SampleGenerator — fixed prompt×seed grid sampling for training evaluation."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torchvision.transforms.functional as TF

from src.lora_trainer.model_adapter import SD15ModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class SampleGrid:
    """Fixed prompt × seed grid for reproducible sampling."""

    prompts: list[str]
    seeds: list[int]
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512


class SampleGenerator:
    """Generate a grid of images from a model adapter using fixed prompts and seeds."""

    def __init__(self, model_adapter: SD15ModelAdapter, grid: SampleGrid):
        self.model_adapter = model_adapter
        self.grid = grid

    def generate_grid(self, output_dir: Path) -> list[Path]:
        """Generate len(prompts) × len(seeds) images and save to output_dir.

        Returns list of saved image paths.
        Raises ValueError if prompts list is empty.
        """
        if not self.grid.prompts:
            raise ValueError("SampleGrid must contain at least one prompt")

        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for p_idx, prompt in enumerate(self.grid.prompts):
            for seed in self.grid.seeds:
                image_tensor = self.model_adapter.generate(
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=self.grid.num_inference_steps,
                    guidance_scale=self.grid.guidance_scale,
                    width=self.grid.width,
                    height=self.grid.height,
                )
                pil_image = TF.to_pil_image(image_tensor.clamp(0, 1))
                filename = f"p{p_idx}_seed{seed}.png"
                save_path = output_dir / filename
                pil_image.save(save_path)
                logger.info("Saved sample: %s (prompt=%r)", save_path, prompt[:50])
                paths.append(save_path)

        return paths
