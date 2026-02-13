"""Tests for data loading and validation."""

import tempfile
from pathlib import Path
import pytest
import torch
from PIL import Image

from src.lora_trainer.data_loader import (
    DataValidator,
    ValidationIssue,
    AspectRatioBucketer,
    LoRADataset,
    create_data_loader,
)


@pytest.fixture
def temp_dataset():
    """Create a temporary test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(i * 85, i * 85, i * 85))
            img.save(tmpdir / f"image_{i:03d}.png")
            
            caption_path = tmpdir / f"image_{i:03d}.txt"
            caption_path.write_text(f"test caption {i}")
        
        yield tmpdir


def test_validate_complete_pairs(temp_dataset):
    """Test validation of complete image+caption pairs."""
    validator = DataValidator(str(temp_dataset))
    report = validator.validate()
    
    assert report.total_images == 3
    assert report.total_captions == 3
    assert report.valid_pairs == 3
    assert report.is_valid


def test_validate_missing_caption(temp_dataset):
    """Test detection of missing caption."""
    (temp_dataset / "image_999.png").write_text("fake")
    
    validator = DataValidator(str(temp_dataset))
    report = validator.validate()
    
    assert report.missing_captions == 1
    assert not report.is_valid
    assert len(report.errors) == 1
    assert "Missing caption" in report.errors[0].message


def test_validate_missing_image(temp_dataset):
    """Test detection of missing image."""
    (temp_dataset / "orphan.txt").write_text("no image")
    
    validator = DataValidator(str(temp_dataset))
    report = validator.validate()
    
    assert report.missing_images == 1
    assert not report.is_valid


def test_validate_empty_caption(temp_dataset):
    """Test detection of empty captions."""
    (temp_dataset / "image_000.txt").write_text("")
    
    validator = DataValidator(str(temp_dataset))
    report = validator.validate()
    
    assert len(report.warnings) > 0
    assert any("Empty caption" in w.message for w in report.warnings)


def test_validate_nonexistent_directory():
    """Test error on nonexistent dataset directory."""
    with pytest.raises(Exception):
        DataValidator("/nonexistent/path")


def test_bucketing_aspect_ratios():
    """Test aspect ratio bucketing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        img_1x1 = Image.new("RGB", (512, 512))
        img_3x2 = Image.new("RGB", (768, 512))
        img_2x3 = Image.new("RGB", (512, 768))
        
        img_1x1.save(tmpdir / "sq.png")
        img_3x2.save(tmpdir / "wide.png")
        img_2x3.save(tmpdir / "tall.png")
        
        image_paths = sorted(tmpdir.glob("*.png"))
        bucketer = AspectRatioBucketer()
        buckets = bucketer.bucket_images(image_paths)
        
        assert len(buckets) > 0
        assert sum(len(v) for v in buckets.values()) == 3


def test_dataset_batch_format(temp_dataset):
    """Test DataLoader batch format."""
    dataset = LoRADataset(str(temp_dataset), resolution=512)
    image, caption = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 512, 512)
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_data_loader_iteration(temp_dataset):
    """Test DataLoader iteration."""
    loader = create_data_loader(
        str(temp_dataset),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    
    batches = list(loader)
    assert len(batches) == 2
    
    images, captions = batches[0]
    assert images.shape[0] == 2
    assert len(captions) == 2


def test_dataset_missing_caption(temp_dataset):
    """Test dataset with missing caption."""
    no_cap_img = Image.new("RGB", (512, 512), color=(200, 200, 200))
    no_cap_img.save(temp_dataset / "no_caption.png")
    
    dataset = LoRADataset(str(temp_dataset))
    image, caption = dataset[3]
    
    assert isinstance(image, torch.Tensor)
    assert caption == ""


def test_validation_report_dict():
    """Test validation report serialization."""
    issue = ValidationIssue(level="error", file="test.txt", message="test error")
    report = __import__("src.lora_trainer.data_loader", fromlist=["ValidationReport"]).ValidationReport(
        total_images=10,
        total_captions=9,
        valid_pairs=8,
        missing_captions=1,
        missing_images=0,
        corrupted_images=0,
        empty_captions=1,
        oversized_captions=0,
        errors=[issue],
    )
    
    report_dict = report.to_dict()
    assert report_dict["summary"]["valid_pairs"] == 8
    assert len(report_dict["issues"]["errors"]) == 1
    assert report_dict["status"] == "invalid"
