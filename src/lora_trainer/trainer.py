"""Trainer - training orchestration layer"""
import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from diffusers.schedulers import DDPMScheduler

from src.lora_trainer.data_loader import create_data_loader
from src.lora_trainer.lora import LoRAAdapter
from src.lora_trainer.model_adapter import SD15ModelAdapter
from src.lora_trainer.run_manager import RunManager


logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator - manages the complete training workflow."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_adapter: SD15ModelAdapter | None = None
        self.lora_adapter: LoRAAdapter | None = None
        self.text_encoder: nn.Module | None = None
        self.unet: nn.Module | None = None
        self.vae: nn.Module | None = None

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.noise_scheduler: DDPMScheduler | None = None
        self.data_loader: Any = None

        self.run_manager: RunManager | None = None
        self.global_step = 0

    def start(self) -> Path:
        """Initialize training: load models, inject LoRA, setup optimizer."""
        model_path = self._resolve_model_path()
        self.config.setdefault("model", {})["model_path"] = model_path

        self.run_manager = RunManager()
        run_dir = self.run_manager.start(self.config)

        logger.info("Loading base model from %s", model_path)
        self.model_adapter = SD15ModelAdapter(model_path)
        self.vae, self.unet, self.text_encoder = self.model_adapter.load_models()

        logger.info("Initializing LoRA adapter (rank=%d, alpha=%.1f)", self.config["lora"]["rank"], self.config["lora"]["alpha"])
        self.lora_adapter = LoRAAdapter(
            rank=self.config["lora"]["rank"],
            alpha=self.config["lora"]["alpha"],
        )

        lora_report = self.lora_adapter.apply_to(
            text_encoder=self.text_encoder,
            unet=self.unet,
            apply_text_encoder=self.config["lora"].get("apply_text_encoder", False),
            apply_unet=True,
            target_modules=None,
            strict=False,
        )
        logger.info("LoRA injection report: %s", lora_report)

        torch.manual_seed(self.config["training"]["seed"])

        dataset_path = self.config.get("data", {}).get("dataset_path") or self.config.get("training", {}).get("dataset")
        self.data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=self.config["training"]["batch_size"],
            resolution=self.config.get("data", {}).get("resolution", 512),
        )

        logger.info("Initializing optimizer and scheduler")
        self.optimizer = AdamW(
            self.lora_adapter.get_trainable_params(),
            lr=self.config["training"]["learning_rate"],
        )

        max_steps = self.config["training"].get("max_steps") or self.config["training"].get("max_train_steps")
        if max_steps is None:
            num_epochs = self.config["training"].get("num_epochs", 10)
            max_steps = num_epochs * len(self.data_loader)

        if self.config["training"].get("lr_scheduler") == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps)
        else:
            self.scheduler = ConstantLR(self.optimizer)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )

        logger.info("Training ready: max_steps=%d device=%s", max_steps, self.device)
        return run_dir

    def _resolve_model_path(self) -> str:
        """Resolve model path from config."""
        if "model" in self.config and "model_path" in self.config["model"]:
            return self.config["model"]["model_path"]
        
        base_model = self.config.get("model", {}).get("base_model", "sd15")
        if base_model == "sd15":
            return "runwayml/stable-diffusion-v1-5"
        raise ValueError(f"Unsupported base_model: {base_model}")

    def train(self) -> None:
        """Main training loop."""
        if self.optimizer is None or self.data_loader is None:
            raise RuntimeError("call start() first")

        max_steps = self.config["training"].get("max_steps") or self.config["training"].get("max_train_steps")
        if max_steps is None:
            num_epochs = self.config["training"].get("num_epochs", 10)
            max_steps = num_epochs * len(self.data_loader)

        save_every_n_steps = self.config["training"].get("save_every_n_steps", 500)

        for epoch in range(self.config["training"].get("num_epochs", 10)):
            for batch in self.data_loader:
                if self.global_step >= max_steps:
                    logger.info("Reached max_steps=%d, stopping training", max_steps)
                    return

                loss = self.train_step(batch)
                self.run_manager.update_training_metrics(
                    self.global_step, {"loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}
                )

                if self.global_step % save_every_n_steps == 0:
                    self.save_checkpoint(self.global_step)

                self.global_step += 1

    def train_step(self, batch: dict[str, Any]) -> float:
        """Single training step."""
        if self.optimizer is None or self.lora_adapter is None or self.unet is None or self.vae is None:
            raise RuntimeError("call start() first")
        if self.model_adapter is None or self.noise_scheduler is None:
            raise RuntimeError("model_adapter or noise_scheduler not initialized")

        images, captions = batch
        images = images.to(self.device)

        with torch.no_grad():
            latents = self.model_adapter.encode_image(images)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            text_embeddings = self.model_adapter.encode_prompt(list(captions))

        model_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save_checkpoint(self, step: int) -> None:
        """Save LoRA checkpoint."""
        if self.lora_adapter is None or self.run_manager is None:
            raise RuntimeError("call start() first")

        weights = self.lora_adapter.state_dict()
        self.run_manager.save_checkpoint(step, weights)

    def end(self) -> None:
        """Export final LoRA model."""
        if self.lora_adapter is None or self.run_manager is None:
            raise RuntimeError("call start() first")

        export_path = self.run_manager.run_dir / "export" / "lora_final.safetensors"
        self.lora_adapter.export_weights(str(export_path))
        logger.info("Exported final LoRA weights to %s", export_path)

        metrics = {
            "total_steps": self.global_step,
            "final_loss": 0.0,
        }
        self.run_manager.end(metrics)
