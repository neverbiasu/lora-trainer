"""Automation agent for Colab-style LoRA training workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class PairValidationResult:
    """Dataset pair validation summary."""

    image_count: int
    missing_caption_count: int
    missing_caption_files: list[str]


@dataclass
class ImageComparisonSummary:
    """Aggregate image comparison statistics."""

    matched_pairs: int
    mean_mae: float
    mean_mse: float


@dataclass
class RunAnalysisSummary:
    """Quantitative analysis summary for a completed run."""

    steps: int
    first_loss: float | None
    final_loss: float | None
    loss_ratio: float | None
    lora_delta_l2: float | None
    lora_delta_mean_abs: float | None
    effectiveness_passed: bool | None
    effectiveness_reasons: list[str]


def auto_detect_single_zip(upload_dir: Path) -> Path:
    """Return a single zip file from upload directory."""
    candidates = sorted(upload_dir.glob("*.zip"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No zip file found in: {upload_dir}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple zip files found. Specify one explicitly: {names}")
    return candidates[0]


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract zip file into destination directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)
    return extract_dir


def validate_image_caption_pairs(dataset_root: Path) -> PairValidationResult:
    """Validate one-to-one image/caption pairs in dataset tree."""
    image_files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    missing: list[str] = []
    for image in image_files:
        caption = image.with_suffix(".txt")
        if not caption.exists():
            missing.append(str(image))

    return PairValidationResult(
        image_count=len(image_files),
        missing_caption_count=len(missing),
        missing_caption_files=missing,
    )


def apply_trigger_token(dataset_root: Path, trigger_token: str) -> int:
    """Prefix all captions with trigger token if not already present."""
    updated = 0
    for txt_path in sorted(dataset_root.rglob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if text.startswith(trigger_token):
            continue
        new_text = f"{trigger_token}, {text}" if text else trigger_token
        txt_path.write_text(new_text, encoding="utf-8")
        updated += 1
    return updated


def compare_image_dirs(reference_dir: Path, candidate_dir: Path) -> ImageComparisonSummary:
    """Compare images by filename intersection and return MAE/MSE summary."""
    ref_images = {
        path.name: path
        for path in reference_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    cand_images = {
        path.name: path
        for path in candidate_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    names = sorted(set(ref_images.keys()) & set(cand_images.keys()))
    if not names:
        return ImageComparisonSummary(matched_pairs=0, mean_mae=0.0, mean_mse=0.0)

    mae_values: list[float] = []
    mse_values: list[float] = []

    for name in names:
        ref = np.asarray(Image.open(ref_images[name]).convert("RGB"), dtype=np.float32)
        cand = np.asarray(Image.open(cand_images[name]).convert("RGB"), dtype=np.float32)
        if ref.shape != cand.shape:
            min_h = min(ref.shape[0], cand.shape[0])
            min_w = min(ref.shape[1], cand.shape[1])
            ref = ref[:min_h, :min_w, :]
            cand = cand[:min_h, :min_w, :]

        diff = ref - cand
        mae_values.append(float(np.mean(np.abs(diff))))
        mse_values.append(float(np.mean(diff * diff)))

    return ImageComparisonSummary(
        matched_pairs=len(names),
        mean_mae=float(np.mean(mae_values)),
        mean_mse=float(np.mean(mse_values)),
    )


def create_comparison_sheet(
    reference_dir: Path,
    candidate_dir: Path,
    output_path: Path,
    max_pairs: int = 8,
) -> Path:
    """Create a side-by-side comparison sheet for matching sample images."""
    ref_images = {
        path.name: path
        for path in reference_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    cand_images = {
        path.name: path
        for path in candidate_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    names = sorted(set(ref_images.keys()) & set(cand_images.keys()))[:max_pairs]
    if not names:
        raise FileNotFoundError("No matching images available for comparison sheet")

    panels: list[Image.Image] = []
    for name in names:
        ref = Image.open(ref_images[name]).convert("RGB")
        cand = Image.open(cand_images[name]).convert("RGB")
        if ref.size != cand.size:
            size = (
                min(ref.size[0], cand.size[0]),
                min(ref.size[1], cand.size[1]),
            )
            ref = ref.resize(size)
            cand = cand.resize(size)

        width = ref.width + cand.width
        height = max(ref.height, cand.height) + 28
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(ref, (0, 28))
        canvas.paste(cand, (ref.width, 28))
        panels.append(canvas)

    sheet_width = max(panel.width for panel in panels)
    sheet_height = sum(panel.height for panel in panels)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#f5f5f5")

    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def extract_log_highlights(log_path: Path, max_lines: int = 80) -> str:
    """Extract the most relevant training log lines for terminal reporting."""
    if not log_path.exists():
        return f"Log file not found: {log_path}"

    interesting_markers = (
        "Precision config",
        "Dataset ready",
        "Training metrics tracked",
        "Loading base model",
        "Initializing LoRA adapter",
        "LoRA injection report",
        "Training ready",
        "=== TRAIN START ===",
        "Training loop complete",
        "=== TRAIN END",
        "Training summary",
        "Training effectiveness reasons",
        "Exported final LoRA weights",
        "Saved checkpoint",
        "Saved sample",
        "WARNING",
        "ERROR",
        "Non-finite loss",
    )

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = [line for line in lines if any(marker in line for marker in interesting_markers)]
    if len(selected) > max_lines:
        head = selected[: max_lines // 2]
        tail = selected[-(max_lines // 2) :]
        selected = [*head, "... truncated ...", *tail]

    return "\n".join(selected)


def build_analysis_summary(metadata: dict[str, Any]) -> RunAnalysisSummary:
    """Convert run metadata into a structured analysis summary."""
    metrics = metadata.get("training_metrics", {}) or {}
    return RunAnalysisSummary(
        steps=int(metrics.get("total_steps", 0) or 0),
        first_loss=metrics.get("first_loss"),
        final_loss=metrics.get("final_loss"),
        loss_ratio=metrics.get("loss_ratio"),
        lora_delta_l2=metrics.get("lora_delta_l2"),
        lora_delta_mean_abs=metrics.get("lora_delta_mean_abs"),
        effectiveness_passed=metrics.get("effectiveness_passed"),
        effectiveness_reasons=list(metrics.get("effectiveness_reasons", []) or []),
    )


def latest_run_dir(run_base: Path) -> Path:
    """Find the latest run directory under run base."""
    candidates = [path for path in run_base.glob("run_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {run_base}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def run_training_cli(
    config_path: Path,
    dataset_path: Path,
    run_dir: Path,
    assert_effective_training: bool,
) -> None:
    """Execute dry-run then training using project CLI."""
    base_cmd = [
        sys.executable,
        "-m",
        "src.lora_trainer.cli",
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_path),
        "--run-dir",
        str(run_dir),
    ]

    dry_run_cmd = [*base_cmd, "--dry-run"]
    subprocess.run(dry_run_cmd, check=True)

    train_cmd = list(base_cmd)
    if assert_effective_training:
        train_cmd.append("--assert-effective-training")
    subprocess.run(train_cmd, check=True)


def archive_run(run_path: Path, archive_path: Path) -> Path:
    """Create a zip archive for a run directory."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    stem = archive_path.with_suffix("")
    generated = shutil.make_archive(
        str(stem), "zip", root_dir=run_path.parent, base_dir=run_path.name
    )
    generated_path = Path(generated)
    if generated_path != archive_path:
        shutil.move(str(generated_path), str(archive_path))
    return archive_path


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Colab agent."""
    parser = argparse.ArgumentParser(
        prog="lora-colab-agent",
        description="Automate Colab LoRA workflow: dataset -> train -> archive -> compare",
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=Path, required=True, help="Base output directory")

    parser.add_argument("--dataset-path", type=Path, help="Existing dataset directory")
    parser.add_argument("--dataset-zip", type=Path, help="Dataset zip path")
    parser.add_argument(
        "--upload-dir",
        type=Path,
        default=Path("/content"),
        help="Directory for auto zip detection",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("/content/dataset"),
        help="Where dataset zip is extracted",
    )

    parser.add_argument("--trigger-token", type=str, help="Prefix token for captions")
    parser.add_argument(
        "--assert-effective-training",
        action="store_true",
        help="Enable post-training effectiveness gate",
    )
    parser.add_argument(
        "--archive-output",
        type=Path,
        default=Path("/content/lora_run_artifacts.zip"),
        help="Zip output path for run artifacts",
    )
    parser.add_argument(
        "--reference-samples-dir",
        type=Path,
        help="Optional baseline sample dir for automatic image diff summary",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("/content/lora_agent_report.json"),
        help="Where to save JSON report",
    )

    return parser


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    """Resolve dataset path from existing directory or zip workflow."""
    if args.dataset_path is not None:
        return args.dataset_path

    zip_path = args.dataset_zip
    if zip_path is None:
        zip_path = auto_detect_single_zip(args.upload_dir)

    return extract_zip(zip_path, args.extract_dir)


def main() -> None:
    """Run end-to-end Colab workflow automation."""
    parser = create_parser()
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args)
    validation = validate_image_caption_pairs(dataset_path)
    if validation.image_count == 0:
        raise RuntimeError("No images found in dataset")
    if validation.missing_caption_count > 0:
        preview = validation.missing_caption_files[:10]
        raise RuntimeError("Missing caption files detected: " + ", ".join(preview))

    trigger_updates = 0
    if args.trigger_token:
        trigger_updates = apply_trigger_token(dataset_path, args.trigger_token)

    run_training_cli(
        config_path=args.config,
        dataset_path=dataset_path,
        run_dir=args.run_dir,
        assert_effective_training=args.assert_effective_training,
    )

    run_path = latest_run_dir(args.run_dir)
    archive_path = archive_run(run_path, args.archive_output)

    metadata_path = run_path / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    analysis = build_analysis_summary(metadata)

    log_report = extract_log_highlights(run_path / "logs" / "train.log")
    log_report_path = args.report_path.with_name(f"{args.report_path.stem}.log.txt")
    log_report_path.write_text(log_report + "\n", encoding="utf-8")

    comparison: dict[str, Any] | None = None
    comparison_sheet_path: str | None = None
    if args.reference_samples_dir:
        compared = compare_image_dirs(args.reference_samples_dir, run_path / "samples")
        comparison = asdict(compared)
        sheet_path = args.report_path.with_name(f"{args.report_path.stem}.comparison.png")
        comparison_sheet_path = str(
            create_comparison_sheet(args.reference_samples_dir, run_path / "samples", sheet_path)
        )

    quantitative_analysis = {
        **asdict(analysis),
        "loss_drop": None,
    }
    if analysis.first_loss is not None and analysis.final_loss is not None:
        quantitative_analysis["loss_drop"] = analysis.first_loss - analysis.final_loss

    report = {
        "dataset_path": str(dataset_path),
        "validation": asdict(validation),
        "trigger_token": args.trigger_token,
        "trigger_updated_files": trigger_updates,
        "run_path": str(run_path),
        "archive_path": str(archive_path),
        "log_report_path": str(log_report_path),
        "quantitative_analysis": quantitative_analysis,
        "comparison": comparison,
        "comparison_sheet_path": comparison_sheet_path,
    }

    report["analysis_text"] = textwrap.dedent(f"""
        Run summary:
        - steps: {quantitative_analysis["steps"]}
        - first_loss: {quantitative_analysis["first_loss"]}
        - final_loss: {quantitative_analysis["final_loss"]}
        - loss_ratio: {quantitative_analysis["loss_ratio"]}
        - lora_delta_l2: {quantitative_analysis["lora_delta_l2"]}
        - lora_delta_mean_abs: {quantitative_analysis["lora_delta_mean_abs"]}
        - effectiveness_passed: {quantitative_analysis["effectiveness_passed"]}
        - effectiveness_reasons: {quantitative_analysis["effectiveness_reasons"]}
        """).strip()

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
