"""Trainer - training orchestration layer"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from tqdm import tqdm

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
        self.last_loss = 0.0

    def start(self, resume: str | None = None) -> Path:
        """Initialize training: load models, inject LoRA, setup optimizer."""
        model_path = self._resolve_model_path()
        self.config.setdefault("model", {})["model_path"] = model_path

        self.run_manager = RunManager()
        run_dir = self.run_manager.start(self.config)

        logger.info("Loading base model from %s", model_path)
        self.model_adapter = SD15ModelAdapter(model_path)
        self.vae, self.unet, self.text_encoder = self.model_adapter.load_models()

        logger.info(
            "Initializing LoRA adapter (rank=%d, alpha=%.1f)",
            self.config["lora"]["rank"],
            self.config["lora"]["alpha"],
        )
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

        dataset_path = self.config.get("data", {}).get("dataset_path") or self.config.get(
            "training", {}
        ).get("dataset")
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

        max_steps = self.config["training"].get("max_steps") or self.config["training"].get(
            "max_train_steps"
        )
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

        if resume is not None:
            logger.info("Resuming from checkpoint: %s", resume)
            cast(LoRAAdapter, self.lora_adapter).load_weights(resume, strict=False)
            meta_path = Path(resume).with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    ckpt_meta = json.load(f)
                self.global_step = ckpt_meta.get("global_step", 0)
                logger.info("Restored global_step=%d from checkpoint metadata", self.global_step)
            else:
                m = re.search(r"step_(\d+)", Path(resume).stem)
                if m:
                    self.global_step = int(m.group(1))
                    logger.info("Restored global_step=%d from filename", self.global_step)

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

    def train(self, resume: str | None = None) -> None:
        """Main training loop - manages start -> loop -> end lifecycle."""
        self.start(resume=resume)

        max_steps = self.config["training"].get("max_steps") or self.config["training"].get(
            "max_train_steps"
        )
        if max_steps is None:
            num_epochs = self.config["training"].get("num_epochs", 10)
            max_steps = num_epochs * len(self.data_loader)

        save_every = self.config["training"].get("save_every_n_steps", 500)
        validate_every = self.config.get("validation", {}).get("every_n_steps", 0)
        show_progress = self.config["training"].get("show_progress", True)
        optimizer = cast(torch.optim.Optimizer, self.optimizer)
        run_manager = cast(RunManager, self.run_manager)

        logger.info("=== TRAIN START ===")
        logger.info(
            "max_steps=%d, save_every=%d, validate_every=%d, initial_global_step=%d",
            max_steps,
            save_every,
            validate_every,
            self.global_step,
        )
        logger.info("data_loader length=%d", len(self.data_loader))

        progress_bar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="steps",
            dynamic_ncols=True,
            disable=not show_progress,
        )

        try:
            data_loader_cycle = iter(self.data_loader)
            logger.info("Starting training loop: %d < %d", self.global_step, max_steps)
            while self.global_step < max_steps:
                logger.debug("=== ITERATION START: step=%d ===", self.global_step)
                try:
                    batch = next(data_loader_cycle)
                    logger.debug(
                        "Got batch: images shape=%s",
                        batch[0].shape if isinstance(batch, tuple) else "unknown",
                    )
                except StopIteration:
                    logger.debug("Data loader exhausted, restarting cycle")
                    data_loader_cycle = iter(self.data_loader)
                    batch = next(data_loader_cycle)

                loss = self.train_step(batch)
                is_finite = math.isfinite(loss)
                logger.debug(
                    "train_step completed: step=%d loss=%.6f isfinite=%s",
                    self.global_step,
                    loss,
                    is_finite,
                )
                if not is_finite:
                    logger.error(
                        "Non-finite loss at step %d. last_loss=%.6f, current_loss=%s",
                        self.global_step,
                        self.last_loss,
                        loss,
                    )
                    raise RuntimeError(
                        "Non-finite loss detected. "
                        "Try lower learning rate, verify captions/dataset quality, "
                        "or reduce LoRA rank/alpha."
                    )
                self.last_loss = loss
                run_manager.update_training_metrics(
                    self.global_step, {"loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                )

                if validate_every > 0 and self.global_step % validate_every == 0:
                    logger.info("Running validation at step %d", self.global_step)
                    self.validate(self.global_step)
                if self.global_step % save_every == 0:
                    logger.info("Saving checkpoint at step %d", self.global_step)
                    self.save_checkpoint(self.global_step)

                self.global_step += 1
                logger.debug("step=%d/%d loss=%f (incremented)", self.global_step, max_steps, loss)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )
            logger.info(
                "Training loop complete: global_step=%d >= max_steps=%d",
                self.global_step,
                max_steps,
            )
        except Exception as e:
            logger.error("Exception in training loop: %s", str(e), exc_info=True)
            raise
        finally:
            logger.info(
                "=== TRAIN END: final_global_step=%d, final_loss=%f ===",
                self.global_step,
                self.last_loss,
            )
            progress_bar.close()
            self.end()

    def train_step(self, batch: tuple[Any, ...]) -> float:
        """Single training step."""
        if (
            self.optimizer is None
            or self.lora_adapter is None
            or self.unet is None
            or self.vae is None
        ):
            raise RuntimeError("call start() first")
        if self.model_adapter is None or self.noise_scheduler is None:
            raise RuntimeError("model_adapter or noise_scheduler not initialized")

        images_tensor, captions = batch[0], batch[1]
        images = cast(torch.Tensor, images_tensor).to(self.device)

        with torch.no_grad():
            latents = self.model_adapter.encode_image(images)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        noise_scheduler = cast(DDPMScheduler, self.noise_scheduler)
        timesteps = torch.randint(
            0, int(noise_scheduler.config["num_train_timesteps"]), (bsz,), device=self.device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, cast(Any, timesteps))

        with torch.no_grad():
            text_embeddings = self.model_adapter.encode_prompt(list(captions))

        model_pred = cast(Any, self.unet(noisy_latents, timesteps, text_embeddings)).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)
        loss_val = loss.item()
        logger.debug(
            "step=%d loss=%.6f, model_pred_range=[%.4f, %.4f], noise_range=[%.4f, %.4f]",
            self.global_step,
            loss_val,
            model_pred.min().item(),
            model_pred.max().item(),
            noise.min().item(),
            noise.max().item(),
        )

        grad_accum = self.config["training"].get("gradient_accumulation", 1)
        (loss / grad_accum).backward()

        if (self.global_step + 1) % grad_accum == 0:
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.lora_adapter.get_trainable_params(), max_grad_norm)
            grad_norm = (
                sum(
                    p.grad.norm().item() ** 2
                    for p in self.lora_adapter.get_trainable_params()
                    if p.grad is not None
                )
                ** 0.5
            )
            logger.debug(
                "step=%d grad_norm=%.6f after clip(%.1f)",
                self.global_step,
                grad_norm,
                max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss_val

    def validate(self, step: int) -> None:
        """Generate a fixed-prompt sample and save to run/samples/."""
        val_cfg = self.config.get("validation")
        if not val_cfg:
            return
        if self.model_adapter is None or self.run_manager is None:
            raise RuntimeError("call start() first")

        image_tensor = self.model_adapter.generate(
            prompt=val_cfg.get("prompt", ""),
            negative_prompt=val_cfg.get("negative_prompt", ""),
            seed=val_cfg.get("seed", 0),
            num_inference_steps=val_cfg.get("num_inference_steps", 20),
            guidance_scale=val_cfg.get("guidance_scale", 7.5),
            width=val_cfg.get("width", 512),
            height=val_cfg.get("height", 512),
        )
        pil_image = TF.to_pil_image(image_tensor)
        self.run_manager.save_sample(step, pil_image)
        logger.info("step=%d sample saved", step)

    def save_checkpoint(self, step: int) -> None:
        """Save LoRA weights and step metadata."""
        if self.lora_adapter is None or self.run_manager is None:
            raise RuntimeError("call start() first")

        weights_path = self.run_manager.save_checkpoint(step, self.lora_adapter.state_dict())
        meta = {
            "global_step": step,
            "rank": self.config["lora"]["rank"],
            "alpha": self.config["lora"]["alpha"],
        }
        meta_path = Path(str(weights_path).replace(".safetensors", ".json"))
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Saved checkpoint: step=%d path=%s", step, weights_path)

    def end(self) -> None:
        """Export final LoRA model."""
        if self.lora_adapter is None or self.run_manager is None:
            raise RuntimeError("call start() first")

        run_dir = cast(RunManager, self.run_manager).run_dir
        export_path = cast(Path, run_dir) / "export" / "lora_final.safetensors"
        base_model = self.config.get("model", {}).get("base_model", "sd15")
        metadata = {
            "rank": str(self.config["lora"]["rank"]),
            "alpha": str(self.config["lora"]["alpha"]),
            "base_model": base_model,
            "format_version": "1.0",
        }
        self.lora_adapter.export_weights(str(export_path), metadata=metadata)
        logger.info("Exported final LoRA weights to %s", export_path)

        metrics = {
            "total_steps": self.global_step,
            "final_loss": self.last_loss,
        }
        self.run_manager.end(metrics)
