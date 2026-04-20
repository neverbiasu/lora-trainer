"""Run manager - handles run directory creation, snapshot saving, metadata management."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from safetensors.torch import save_file as save_safetensors


class RunManager:
    """Run manager"""

    def __init__(self, output_dir: Path = Path("./output")):
        self.output_dir = output_dir
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.metadata: dict[str, Any] | None = None
        self._logger = logging.getLogger(__name__)

    def start(self, config: dict[str, Any]) -> Path:
        """Create run directory structure and initialize snapshots."""
        self.run_id = self._generate_run_id(config)
        export_dir = Path(config.get("export", {}).get("output_dir", self.output_dir))
        self.run_dir = export_dir / self.run_id

        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        (self.run_dir / "export").mkdir(exist_ok=True)

        self.save_config_snapshot(config)
        self.metadata = self.init_metadata(config)
        self.save_metadata(self.metadata)
        self.setup_logging(self.run_dir)
        self._logger.info("Starting training run: %s", self.run_id)
        self._logger.info("Run directory: %s", self.run_dir)

        return self.run_dir

    def save_checkpoint(self, step: int, weights: dict[str, Any]) -> Path:
        """Save checkpoint weights to the run directory."""
        run_dir = self._require_run_dir()
        path = run_dir / "checkpoints" / f"step_{step:04d}.safetensors"
        save_safetensors(weights, path)
        self._logger.info("Saved checkpoint: step=%s path=%s", step, path)
        return path

    def save_sample(self, step: int, image: Any) -> Path:
        """Save a sample image to the run directory."""
        run_dir = self._require_run_dir()
        path = run_dir / "samples" / f"step_{step:04d}.png"
        image.save(path)
        self._logger.info("Saved sample: step=%s path=%s", step, path)
        return path

    def end(self, metrics: dict[str, Any]) -> None:
        """Finalize run metadata."""
        if self.metadata is None:
            self.metadata = self.init_metadata({})
        self.metadata["training_metrics"] = metrics
        self.save_metadata(self.metadata)

    def save_config_snapshot(self, config: dict[str, Any]) -> None:
        """Save configuration snapshot."""
        run_dir = self._require_run_dir()
        snapshot = dict(config)
        snapshot["_snapshot_meta"] = {
            "created_at": self._utc_now(),
            "original_config_path": config.get("_config_path"),
            "config_version": config.get("_config_version", "0.1"),
        }
        snapshot_path = run_dir / "config_snapshot.yaml"
        snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata (7-element snapshot)."""
        run_dir = self._require_run_dir()
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True))

    def update_training_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log training metrics."""
        metrics_str = " ".join(f"{key}={value}" for key, value in metrics.items())
        self._logger.info("step=%s %s", step, metrics_str)

    def init_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        """Initialize metadata skeleton."""
        seed = config.get("training", {}).get("seed")
        return {
            "run_id": self.run_id,
            "created_at": self._utc_now(),
            "reproducibility": {
                "base_model_hash": None,
                "config_hash": None,
                "seed": seed,
                "data_manifest_hash": None,
                "code_version": None,
                "environment": {},
            },
            "training_metrics": {},
        }

    def setup_logging(self, run_dir: Path) -> None:
        """Configure logging to stdout and run log file."""
        log_file = run_dir / "logs" / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def _generate_run_id(self, config: dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get("model", {}).get("base_model", "model")
        model_name = str(model_name).split("/")[-1]
        algo = config.get("lora", {}).get("algorithm", "lora")
        return f"run_{timestamp}_{model_name}_{algo}"

    def _require_run_dir(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("run_dir is not set; call start() first")
        return self.run_dir

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
