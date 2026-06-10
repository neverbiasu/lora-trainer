# src/lora_trainer/evaluator.py
"""TrainingEvaluator — pixel diff, CLIPScore, and comparison sheet generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _list_images(directory: Path) -> dict[str, Path]:
    """Return {filename: path} for image files in directory."""
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


@dataclass
class EvaluationReport:
    """Complete evaluation report for a training run."""

    mean_pixel_mae: float
    mean_pixel_mse: float
    baseline_clip_sim: float | None = None
    lora_clip_sim: float | None = None
    delta_clip: float | None = None
    comparison_sheet_path: Path | None = None
    per_image_details: list[dict[str, Any]] = field(default_factory=list)


class TrainingEvaluator:
    """Evaluate training effectiveness via pixel diff and CLIPScore."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip_model: Any = None
        self._clip_processor: Any = None

    # -- Pixel diff ----------------------------------------------------------

    def compute_pixel_diff(
        self, baseline_dir: Path, final_dir: Path
    ) -> tuple[float, float]:
        """Compute mean MAE and MSE between matched images in two directories.

        Returns (mean_mae, mean_mse). Only files with the same name are compared.
        Returns (0.0, 0.0) if no matching files exist.
        """
        baseline_images = _list_images(baseline_dir)
        final_images = _list_images(final_dir)
        names = sorted(set(baseline_images) & set(final_images))

        if not names:
            return 0.0, 0.0

        mae_values: list[float] = []
        mse_values: list[float] = []

        for name in names:
            ref = np.asarray(
                Image.open(baseline_images[name]).convert("RGB"), dtype=np.float32
            )
            cand = np.asarray(
                Image.open(final_images[name]).convert("RGB"), dtype=np.float32
            )
            # Handle size mismatch by cropping to the smaller dimensions
            min_h = min(ref.shape[0], cand.shape[0])
            min_w = min(ref.shape[1], cand.shape[1])
            ref = ref[:min_h, :min_w, :]
            cand = cand[:min_h, :min_w, :]

            diff = ref - cand
            mae_values.append(float(np.mean(np.abs(diff))))
            mse_values.append(float(np.mean(diff * diff)))

        return float(np.mean(mae_values)), float(np.mean(mse_values))

    # -- Comparison sheet ----------------------------------------------------

    def create_comparison_sheet(
        self,
        baseline_dir: Path,
        final_dir: Path,
        output_path: Path,
        max_pairs: int = 9,
    ) -> Path:
        """Create a side-by-side baseline vs final comparison grid.

        Raises FileNotFoundError if no matching filenames exist.
        """
        baseline_images = _list_images(baseline_dir)
        final_images = _list_images(final_dir)
        names = sorted(set(baseline_images) & set(final_images))[:max_pairs]

        if not names:
            raise FileNotFoundError(
                "No matching image filenames between baseline and final directories"
            )

        panels: list[Image.Image] = []
        for name in names:
            ref = Image.open(baseline_images[name]).convert("RGB")
            cand = Image.open(final_images[name]).convert("RGB")
            # Resize to match if needed
            if ref.size != cand.size:
                size = (min(ref.width, cand.width), min(ref.height, cand.height))
                ref = ref.resize(size, Image.LANCZOS)
                cand = cand.resize(size, Image.LANCZOS)

            label_height = 28
            width = ref.width + cand.width
            height = max(ref.height, cand.height) + label_height
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(ref, (0, label_height))
            canvas.paste(cand, (ref.width, label_height))
            panels.append(canvas)

        sheet_width = max(p.width for p in panels)
        sheet_height = sum(p.height for p in panels)
        sheet = Image.new("RGB", (sheet_width, sheet_height), "#f5f5f5")

        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        logger.info("Saved comparison sheet: %s (%d pairs)", output_path, len(names))
        return output_path

    # -- CLIP similarity -----------------------------------------------------

    def _load_clip(self) -> None:
        """Lazy-load CLIP model and processor."""
        if self._clip_model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        logger.info("Loading CLIP model: %s", model_name)
        self._clip_processor = CLIPProcessor.from_pretrained(model_name)
        self._clip_model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def _get_clip_image_embedding(self, image_path: Path) -> "torch.Tensor":
        """Return L2-normalized CLIP image embedding for a single image."""
        import torch

        self._load_clip()
        image = Image.open(image_path).convert("RGB")
        inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0]  # shape: (512,)

    def _compute_dataset_centroid(self, dataset_path: str) -> "torch.Tensor":
        """Compute mean CLIP embedding over all images in the dataset directory."""
        import torch

        dataset_dir = Path(dataset_path)
        image_paths = sorted(
            p for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in dataset: {dataset_path}")

        embeddings = []
        for img_path in image_paths:
            emb = self._get_clip_image_embedding(img_path)
            embeddings.append(emb)

        centroid = torch.stack(embeddings).mean(dim=0)
        centroid = centroid / centroid.norm()
        return centroid

    def compute_clip_similarity(
        self, generated_dir: Path, reference_dir: Path
    ) -> float:
        """Compute mean CLIP cosine similarity between generated images and reference centroid.

        Returns average cosine similarity as a float in [-1, 1].
        """
        import torch

        centroid = self._compute_dataset_centroid(str(reference_dir))
        gen_images = _list_images(generated_dir)

        if not gen_images:
            return 0.0

        similarities: list[float] = []
        for name, path in gen_images.items():
            emb = self._get_clip_image_embedding(path)
            sim = float(torch.dot(emb, centroid).item())
            similarities.append(sim)

        return float(np.mean(similarities))

    # -- Full evaluation pipeline -------------------------------------------

    def evaluate(
        self,
        baseline_dir: Path,
        final_dir: Path,
        dataset_path: str,
        output_dir: Path,
    ) -> EvaluationReport:
        """Run full evaluation: pixel diff + CLIPScore + comparison sheet."""
        mean_mae, mean_mse = self.compute_pixel_diff(baseline_dir, final_dir)

        baseline_clip = self.compute_clip_similarity(baseline_dir, Path(dataset_path))
        lora_clip = self.compute_clip_similarity(final_dir, Path(dataset_path))
        delta_clip = lora_clip - baseline_clip

        output_dir.mkdir(parents=True, exist_ok=True)
        sheet_path = output_dir / "comparison.png"
        try:
            self.create_comparison_sheet(baseline_dir, final_dir, sheet_path)
        except FileNotFoundError:
            sheet_path = None
            logger.warning("Could not create comparison sheet: no matching images")

        report = EvaluationReport(
            mean_pixel_mae=mean_mae,
            mean_pixel_mse=mean_mse,
            baseline_clip_sim=baseline_clip,
            lora_clip_sim=lora_clip,
            delta_clip=delta_clip,
            comparison_sheet_path=sheet_path,
        )

        logger.info(
            "Evaluation complete: pixel_mae=%.2f clip_baseline=%.4f clip_lora=%.4f delta_clip=%.4f",
            mean_mae,
            baseline_clip,
            lora_clip,
            delta_clip,
        )
        return report
