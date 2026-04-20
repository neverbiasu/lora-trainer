"""
Data loading and validation for LoRA training.
Handles image+caption pairs, aspect ratio bucketing, and validation reporting.
"""

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.lora_trainer.errors import InvalidConfigError

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Single validation issue."""

    level: str  # "error", "warning", "info"
    file: str
    message: str


@dataclass
class ValidationReport:
    """Complete validation report."""

    total_images: int
    total_captions: int
    valid_pairs: int
    missing_captions: int
    missing_images: int
    corrupted_images: int
    empty_captions: int
    oversized_captions: int
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Dataset is valid if no errors."""
        return len(self.errors) == 0 and self.valid_pairs > 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_images": self.total_images,
                "total_captions": self.total_captions,
                "valid_pairs": self.valid_pairs,
                "missing_captions": self.missing_captions,
                "missing_images": self.missing_images,
                "corrupted_images": self.corrupted_images,
                "empty_captions": self.empty_captions,
            },
            "issues": {
                "errors": [asdict(e) for e in self.errors],
                "warnings": [asdict(w) for w in self.warnings],
            },
            "status": "valid" if self.is_valid else "invalid",
        }


class DataValidator:
    """Validate image+caption dataset structure and content."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    MAX_CAPTION_LENGTH = 500

    def __init__(self, dataset_path: str):
        """Initialize validator for a dataset directory."""
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise InvalidConfigError(
                f"Dataset directory not found: {dataset_path}",
                suggestions=[
                    "Check dataset_path in config",
                    f"Create directory: mkdir -p {dataset_path}",
                ],
            )

    def validate(self) -> ValidationReport:
        """Run full validation of dataset."""
        report = ValidationReport(
            total_images=0,
            total_captions=0,
            valid_pairs=0,
            missing_captions=0,
            missing_images=0,
            corrupted_images=0,
            empty_captions=0,
            oversized_captions=0,
        )

        image_files = self._find_images()
        caption_files = self._find_captions()

        report.total_images = len(image_files)
        report.total_captions = len(caption_files)

        image_dict = {f.stem: f for f in image_files}
        caption_dict = {f.stem: f for f in caption_files}

        for stem in image_dict:
            if stem not in caption_dict:
                report.missing_captions += 1
                report.errors.append(
                    ValidationIssue(
                        level="error",
                        file=image_dict[stem].name,
                        message=f"Missing caption file: {stem}.txt",
                    )
                )
            else:
                if self._validate_image(image_dict[stem]):
                    caption_file = caption_dict[stem]
                    caption_issue = self._validate_caption(caption_file, stem)

                    if caption_issue is None:
                        report.valid_pairs += 1
                    else:
                        if caption_issue.level == "error":
                            report.errors.append(caption_issue)
                        else:
                            report.warnings.append(caption_issue)
                else:
                    report.corrupted_images += 1

        for stem in caption_dict:
            if stem not in image_dict:
                report.missing_images += 1
                report.errors.append(
                    ValidationIssue(
                        level="error",
                        file=caption_dict[stem].name,
                        message=f"Missing image file for caption: {stem}.*",
                    )
                )

        logger.info(f"Validation complete: {report.valid_pairs} valid pairs")
        return report

    def _find_images(self) -> List[Path]:
        """Find all supported image files."""
        images = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(self.dataset_path.glob(f"*{ext}"))
            images.extend(self.dataset_path.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _find_captions(self) -> List[Path]:
        """Find all caption files."""
        return sorted(self.dataset_path.glob("*.txt"))

    def _validate_image(self, image_path: Path) -> bool:
        """Check if image is readable and valid."""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path.name}: {e}")
            return False

    def _validate_caption(self, caption_path: Path, stem: str) -> Optional[ValidationIssue]:
        """Validate caption content."""
        try:
            text = caption_path.read_text(encoding="utf-8").strip()

            if not text:
                return ValidationIssue(
                    level="warning",
                    file=caption_path.name,
                    message="Empty caption",
                )

            if len(text) > self.MAX_CAPTION_LENGTH:
                return ValidationIssue(
                    level="warning",
                    file=caption_path.name,
                    message=f"Caption too long ({len(text)} > {self.MAX_CAPTION_LENGTH})",
                )

            if any(ord(c) < 32 and c not in "\n\t" for c in text):
                return ValidationIssue(
                    level="error",
                    file=caption_path.name,
                    message="Caption contains control characters",
                )

            return None

        except UnicodeDecodeError as e:
            return ValidationIssue(
                level="error",
                file=caption_path.name,
                message=f"Invalid encoding: {e}",
            )
        except IOError as e:
            return ValidationIssue(
                level="error",
                file=caption_path.name,
                message=f"Cannot read file: {e}",
            )


class AspectRatioBucketer:
    """Group images by aspect ratio for efficient batching."""

    STANDARD_RATIOS = [
        0.5,
        0.67,
        0.75,
        1.0,
        1.33,
        1.5,
        2.0,
    ]

    def __init__(self, target_area: int = 512 * 512):
        """Initialize bucketer."""
        self.target_area = target_area

    def bucket_images(self, image_paths: List[Path]) -> Dict[str, List[int]]:
        """Assign images to buckets by aspect ratio."""
        buckets = defaultdict(list)

        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                w, h = img.size
                ratio = w / h if h > 0 else 1.0

                closest_ratio = min(
                    self.STANDARD_RATIOS,
                    key=lambda r: abs(r - ratio),
                )

                bucket_key = f"{closest_ratio:.2f}"
                buckets[bucket_key].append(idx)

            except Exception as e:
                logger.warning(f"Cannot read {img_path.name}: {e}")

        return dict(buckets)


class LoRADataset(Dataset):
    """PyTorch Dataset for LoRA training."""

    def __init__(
        self,
        dataset_path: str,
        resolution: int = 512,
    ):
        """Initialize dataset."""
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution

        self.image_paths = sorted(
            list(self.dataset_path.glob("*.png"))
            + list(self.dataset_path.glob("*.jpg"))
            + list(self.dataset_path.glob("*.jpeg"))
            + list(self.dataset_path.glob("*.webp"))
        )

        if not self.image_paths:
            raise InvalidConfigError(
                f"No images found in {dataset_path}",
                suggestions=["Check dataset_path", "Ensure PNG/JPG files exist"],
            )

        self.captions = {}
        for img_path in self.image_paths:
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.captions[str(img_path)] = caption_path.read_text(encoding="utf-8").strip()
            else:
                logger.warning(f"Missing caption for {img_path.name}")
                self.captions[str(img_path)] = ""

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get image and caption."""
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load {img_path.name}: {e}")
            raise

        image_tensor = self.transform(img)
        caption = self.captions[str(img_path)]

        return image_tensor, caption


def create_data_loader(
    dataset_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    resolution: int = 512,
) -> DataLoader:
    """Create DataLoader for training."""
    dataset = LoRADataset(dataset_path, resolution=resolution)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Created DataLoader: {len(dataset)} images, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    return loader
