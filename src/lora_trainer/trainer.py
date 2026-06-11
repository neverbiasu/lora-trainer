"""Trainer - training orchestration layer"""

import json
import logging
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from tqdm import tqdm

from src.lora_trainer.data_loader import create_data_loader
from src.lora_trainer.errors import IneffectiveTrainingError
from src.lora_trainer.evaluator import TrainingEvaluator
from src.lora_trainer.lora import LoRAAdapter
from src.lora_trainer.model_adapter import SD15ModelAdapter
from src.lora_trainer.run_manager import RunManager
from src.lora_trainer.sampler import SampleGenerator, SampleGrid
from src.lora_trainer.training_validation import evaluate_training_effectiveness

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator - manages the complete training workflow."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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
        self.first_loss: float | None = None
        self.loss_history: list[tuple[int, float]] = []
        self.initial_lora_state: dict[str, torch.Tensor] | None = None
        self.mixed_precision_mode = "fp32"
        self.amp_autocast_enabled = False
        self.amp_dtype: torch.dtype | None = None
        self.grad_scaler: torch.amp.GradScaler | None = None
        self._sample_grid: SampleGrid | None = None

    def start(self, resume: str | None = None) -> Path:
        """Initialize training: load models, inject LoRA, setup optimizer."""
        model_path = self._resolve_model_path()
        self.config.setdefault("model", {})["model_path"] = model_path

        self.run_manager = RunManager()
        run_dir = self.run_manager.start(self.config)

        self._configure_precision()
        logger.info(
            "Precision config: mixed_precision=%s autocast=%s amp_dtype=%s scaler=%s device=%s",
            self.mixed_precision_mode,
            self.amp_autocast_enabled,
            self.amp_dtype,
            self.grad_scaler.is_enabled() if self.grad_scaler is not None else False,
            self.device,
        )

        logger.info("Loading base model from %s", model_path)
        self.model_adapter = SD15ModelAdapter(model_path)

        # Decide the dtype to load the base model in
        load_dtype = torch.float32 if self.mixed_precision_mode == "fp32" else torch.float16

        self.vae, self.unet, self.text_encoder = self.model_adapter.load_models(
            target_dtype=load_dtype
        )

        # Freeze base models and set to eval mode
        self.unet.requires_grad_(False)
        self.unet.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()

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
        self.initial_lora_state = {
            key: tensor.detach().float().cpu().clone()
            for key, tensor in self.lora_adapter.state_dict().items()
        }
        logger.info("Trainable LoRA tensors=%d", len(self.initial_lora_state))

        torch.manual_seed(self.config["training"]["seed"])

        dataset_path = self.config.get("data", {}).get("dataset_path") or self.config.get(
            "training", {}
        ).get("dataset")
        self.data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=self.config["training"]["batch_size"],
            resolution=self.config.get("data", {}).get("resolution", 512),
        )
        logger.info(
            "Dataset ready: path=%s batch_size=%d resolution=%d batches=%d",
            dataset_path,
            self.config["training"]["batch_size"],
            self.config.get("data", {}).get("resolution", 512),
            len(self.data_loader),
        )

        logger.info("Initializing optimizer and scheduler")

        # Split parameters into UNet and Text Encoder groups for separate learning rates
        unet_params = []
        te_params = []
        for name, param in self.lora_adapter.named_parameters():
            if not param.requires_grad:
                continue
            # Text encoder parameters usually end up under 'lora_modules.text_encoder...'
            # This depends on how apply_to registers them. `apply_to` iterates over named_modules.
            # In diffusers SD1.5, UNet layers often start with 'down_blocks', 'up_blocks', 'mid_block'.
            # TextEncoder layers start with 'text_model'.
            if "text_model" in name or "text_encoder" in name:
                te_params.append(param)
            else:
                unet_params.append(param)

        unet_lr = self.config["training"]["learning_rate"]
        te_lr = self.config["training"].get("text_encoder_lr", unet_lr * 0.1)

        param_groups = [{"params": unet_params, "lr": unet_lr}]
        if te_params:
            param_groups.append({"params": te_params, "lr": te_lr})
            logger.info("Configured separate learning rate for Text Encoder: %.2e", te_lr)

        self.optimizer = AdamW(param_groups)

        max_steps = self.config["training"].get("max_steps") or self.config["training"].get(
            "max_train_steps"
        )
        if max_steps is None:
            num_epochs = self.config["training"].get("num_epochs", 10)
            max_steps = num_epochs * len(self.data_loader)

        lr_scheduler_name = self.config["training"].get("lr_scheduler", "constant")
        lr_warmup_steps = self.config["training"].get("lr_warmup_steps", 0)

        # Fallback to defaults or native PyTorch if generic scheduler not found via diffusers
        try:
            self.scheduler = get_scheduler(
                lr_scheduler_name,
                optimizer=self.optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=max_steps,
            )
        except Exception:
            # Fallback to existing manual assignment
            if lr_scheduler_name == "cosine":
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

        # Generate baseline samples (LoRA weights are zero-init, output == base model)
        self._sample_grid = self._build_sample_grid()
        if self._sample_grid is not None and self.model_adapter is not None:
            sampler = SampleGenerator(self.model_adapter, self._sample_grid)
            baseline_dir = cast(Path, run_dir) / "samples" / "baseline"
            logger.info("Generating baseline samples to %s", baseline_dir)
            sampler.generate_grid(baseline_dir)
            # Pipeline inference can corrupt model dtypes; fix them
            self._enforce_model_dtypes()

        return run_dir

    def _configure_precision(self) -> None:
        """Configure mixed precision and loss scaling for the current device."""
        training_cfg = self.config.get("training", {})
        self.mixed_precision_mode = str(training_cfg.get("mixed_precision", "fp32"))

        # MPS has limited support for fp16 autocast with SD1.5, often it's better to stay fp32, but we preserve original intent.
        if self.device.type not in ["cuda", "mps"] or self.mixed_precision_mode == "fp32":
            if self.mixed_precision_mode != "fp32":
                logger.warning(
                    "Mixed precision=%s requested but device=%s; falling back to fp32",
                    self.mixed_precision_mode,
                    self.device,
                )
            self.amp_autocast_enabled = False
            self.amp_dtype = None
            self.grad_scaler = None
            return

        if self.mixed_precision_mode == "fp16":
            self.amp_autocast_enabled = True
            self.amp_dtype = torch.float16
            self.grad_scaler = torch.amp.GradScaler("cuda")
            return

        if self.mixed_precision_mode == "bf16":
            self.amp_autocast_enabled = True
            self.amp_dtype = torch.bfloat16
            self.grad_scaler = None
            return

        raise ValueError(f"Unsupported mixed_precision: {self.mixed_precision_mode}")

    def _autocast_context(self):
        """Return an autocast context for the current precision mode."""
        if not self.amp_autocast_enabled or self.amp_dtype is None:
            return nullcontext()
        # Fallback to cpu autocast if device is not cuda/mps, else use device.type
        # Note MPS doesnt support all autocast dtypes currently but diffusers handles it loosely
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _resolve_model_path(self) -> str:
        """Resolve model path from config."""
        if "model" in self.config and "model_path" in self.config["model"]:
            return self.config["model"]["model_path"]

        base_model = self.config.get("model", {}).get("base_model", "sd15")
        if base_model == "sd15":
            return "runwayml/stable-diffusion-v1-5"
        raise ValueError(f"Unsupported base_model: {base_model}")

    def _build_sample_grid(self) -> SampleGrid | None:
        """Build a SampleGrid from config. Returns None if sampling is disabled."""
        sampling_cfg = self.config.get("sampling", {})
        sample_every = sampling_cfg.get("sample_every_n_steps", 0)

        # Fallback: check old config paths for backward compat
        if not sample_every:
            sample_every = self.config.get("training", {}).get("sample_every_n_steps", 0)
        if not sample_every:
            sample_every = self.config.get("validation", {}).get("every_n_steps", 0)

        if not sample_every:
            return None

        prompts = list(sampling_cfg.get("prompts", []))
        if not prompts:
            # Fallback: pick up to 3 captions from the dataset
            dataset_path = self.config.get("data", {}).get("dataset_path")
            if dataset_path:
                caption_files = sorted(Path(dataset_path).glob("*.txt"))[:3]
                for cf in caption_files:
                    text = cf.read_text(encoding="utf-8").strip()
                    if text:
                        prompts.append(text)
            if not prompts:
                prompts = ["a photo"]

        return SampleGrid(
            prompts=prompts,
            seeds=sampling_cfg.get("seeds", [42, 123, 999]),
            num_inference_steps=sampling_cfg.get("num_inference_steps", 20),
            guidance_scale=sampling_cfg.get("guidance_scale", 7.5),
            width=self.config.get("data", {}).get("resolution", 512),
            height=self.config.get("data", {}).get("resolution", 512),
        )

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
        logger.info(
            "Training metrics tracked: step/loss/lr, final loss_ratio, lora_delta_l2, lora_delta_mean_abs"
        )

        # Diagnose and fix dtype mismatches after baseline sampling
        self._enforce_model_dtypes()

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
                if self.first_loss is None:
                    self.first_loss = loss
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
                self.loss_history.append((self.global_step, loss))
                run_manager.update_training_metrics(
                    self.global_step, {"loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                )

                if validate_every > 0 and self.global_step % validate_every == 0:
                    logger.info("Running validation at step %d", self.global_step)
                    self.validate(self.global_step)
                if self.global_step % save_every == 0:
                    logger.info("Saving checkpoint at step %d", self.global_step)
                    self.save_checkpoint(self.global_step)

                # Intermediate sampling
                sample_every = self.config.get("sampling", {}).get("sample_every_n_steps", 0)
                if (
                    sample_every > 0
                    and self._sample_grid is not None
                    and self.model_adapter is not None
                    and self.global_step > 0
                    and self.global_step % sample_every == 0
                ):
                    step_sample_dir = (
                        cast(Path, run_manager.run_dir) / "samples" / f"step_{self.global_step:04d}"
                    )
                    logger.info("Generating intermediate samples at step %d", self.global_step)
                    SampleGenerator(self.model_adapter, self._sample_grid).generate_grid(
                        step_sample_dir
                    )
                    self._enforce_model_dtypes()

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

        with torch.no_grad(), self._autocast_context():
            latents = self.model_adapter.encode_image(images)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        noise_scheduler = cast(DDPMScheduler, self.noise_scheduler)
        timesteps = torch.randint(
            0, int(noise_scheduler.config["num_train_timesteps"]), (bsz,), device=self.device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, cast(Any, timesteps))

        apply_te = self.config.get("lora", {}).get("apply_text_encoder", True)

        with self._autocast_context():
            text_embeddings = self.model_adapter.encode_prompt(list(captions), enable_grad=apply_te)

        # Ensure inputs match UNet dtype (noise_scheduler.add_noise may return float32)
        unet_dtype = next(self.unet.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=unet_dtype)
        text_embeddings = text_embeddings.to(dtype=unet_dtype)

        with self._autocast_context():
            model_pred = cast(Any, self.unet(noisy_latents, timesteps, text_embeddings)).sample

        loss_tensor = torch.nn.functional.mse_loss(
            model_pred.float(), noise.float(), reduction="none"
        )
        loss_tensor = loss_tensor.mean(dim=[1, 2, 3])

        min_snr_gamma = self.config["training"].get("min_snr_gamma")
        if min_snr_gamma is not None:
            # Apply Min-SNR Weighting to balance learning across timesteps
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(self.device)
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            min_snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr
            loss_tensor = loss_tensor * min_snr_weight

        loss = loss_tensor.mean()
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
        scaled_loss = loss / grad_accum
        scaler = self.grad_scaler
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (self.global_step + 1) % grad_accum == 0:
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            params = self.lora_adapter.get_trainable_params()
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            grad_norm = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
            logger.debug(
                "step=%d grad_norm=%.6f after clip(%.1f)",
                self.global_step,
                grad_norm,
                max_grad_norm,
            )

            if scaler is not None and scaler.is_enabled():
                scaler.step(self.optimizer)
                scaler.update()
            else:
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

        # Generate final samples and run evaluation
        eval_metrics: dict[str, Any] = {}
        if self._sample_grid is not None and self.model_adapter is not None:
            run_dir_path = cast(Path, cast(RunManager, self.run_manager).run_dir)
            final_dir = run_dir_path / "samples" / "final"
            baseline_dir = run_dir_path / "samples" / "baseline"

            logger.info("Generating final samples to %s", final_dir)
            SampleGenerator(self.model_adapter, self._sample_grid).generate_grid(final_dir)
            self._enforce_model_dtypes()

            if baseline_dir.exists():
                dataset_path = self.config.get("data", {}).get("dataset_path", "")
                evaluator = TrainingEvaluator(device=str(self.device))
                eval_report = evaluator.evaluate(
                    baseline_dir=baseline_dir,
                    final_dir=final_dir,
                    dataset_path=dataset_path,
                    output_dir=run_dir_path / "samples",
                )
                eval_metrics = {
                    "mean_pixel_mae": eval_report.mean_pixel_mae,
                    "mean_pixel_mse": eval_report.mean_pixel_mse,
                    "baseline_clip_sim": eval_report.baseline_clip_sim,
                    "lora_clip_sim": eval_report.lora_clip_sim,
                    "delta_clip": eval_report.delta_clip,
                }
                logger.info(
                    "Evaluation: pixel_mae=%.2f delta_clip=%.4f",
                    eval_report.mean_pixel_mae,
                    eval_report.delta_clip if eval_report.delta_clip is not None else 0.0,
                )

        lora_delta_metrics = self._compute_lora_delta_metrics()
        loss_ratio = None
        if self.first_loss is not None and self.first_loss > 0:
            loss_ratio = self.last_loss / self.first_loss

        metrics = {
            "total_steps": self.global_step,
            "final_loss": self.last_loss,
            "first_loss": self.first_loss,
            "loss_ratio": loss_ratio,
            **lora_delta_metrics,
            **eval_metrics,
        }

        report = evaluate_training_effectiveness(metrics, self.config)
        metrics["effectiveness_passed"] = report.passed
        metrics["effectiveness_reasons"] = report.reasons
        logger.info(
            "Training summary: steps=%d first_loss=%s final_loss=%s loss_ratio=%s lora_delta_l2=%s lora_delta_mean_abs=%s effectiveness=%s",
            self.global_step,
            self.first_loss,
            self.last_loss,
            loss_ratio,
            lora_delta_metrics.get("lora_delta_l2"),
            lora_delta_metrics.get("lora_delta_mean_abs"),
            report.passed,
        )
        if report.reasons:
            logger.warning("Training effectiveness reasons: %s", report.reasons)

        if self.loss_history:
            try:
                import csv

                loss_csv_path = cast(Path, run_dir) / "logs" / "loss_history.csv"
                with open(loss_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "loss"])
                    writer.writerows(self.loss_history)
                logger.info("Saved loss history CSV to %s", loss_csv_path)
            except Exception as e:
                logger.warning("Failed to save loss history CSV: %s", e)

            try:
                import matplotlib.pyplot as plt

                steps, losses = zip(*self.loss_history)
                plt.figure(figsize=(10, 6))
                plt.plot(
                    steps, losses, label="Training Loss", color="blue", linewidth=1.5, alpha=0.8
                )

                # Apply moving average smoothing for better visibility
                if len(losses) > 10:
                    window = min(10, len(losses) // 5)
                    smoothed = [
                        sum(losses[i : i + window]) / window
                        for i in range(len(losses) - window + 1)
                    ]
                    smooth_steps = steps[window - 1 :]
                    plt.plot(
                        smooth_steps,
                        smoothed,
                        label=f"Smoothed Loss (window={window})",
                        color="red",
                        linewidth=2,
                    )

                plt.xlabel("Global Step")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()

                loss_chart_path = cast(Path, run_dir) / "logs" / "loss_curve.png"
                plt.savefig(loss_chart_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved loss curve chart to %s", loss_chart_path)
            except ImportError:
                logger.warning("matplotlib is not installed. Loss chart will not be generated.")
            except Exception as e:
                logger.warning("Failed to generate loss curve chart: %s", e)

        self.run_manager.end(metrics)

        if self.config.get("validation", {}).get("assert_effective_training") and not report.passed:
            raise IneffectiveTrainingError(
                "Training did not pass effectiveness gate checks",
                suggestions=report.reasons,
            )

    def _enforce_model_dtypes(self) -> None:
        """Check and fix dtype mismatches in model parameters.

        StableDiffusionPipeline can leave some parameters (especially biases)
        in float32 even when the model was loaded in fp16. This method detects
        and corrects such mismatches.
        """
        if self.unet is None or self.text_encoder is None:
            return

        expected_dtype = torch.float16 if self.mixed_precision_mode != "fp32" else torch.float32
        fixed_count = 0

        for model_name, model in [("unet", self.unet), ("text_encoder", self.text_encoder)]:
            for name, param in model.named_parameters():
                if param.dtype != expected_dtype:
                    logger.warning(
                        "Dtype mismatch: %s.%s is %s, expected %s — fixing",
                        model_name,
                        name,
                        param.dtype,
                        expected_dtype,
                    )
                    param.data = param.data.to(expected_dtype)
                    fixed_count += 1
            for name, buf in model.named_buffers():
                if buf.is_floating_point() and buf.dtype != expected_dtype:
                    logger.warning(
                        "Buffer dtype mismatch: %s.%s is %s — fixing",
                        model_name,
                        name,
                        buf.dtype,
                    )
                    buf.data = buf.data.to(expected_dtype)
                    fixed_count += 1

        if fixed_count > 0:
            logger.warning("Fixed %d dtype mismatches in model parameters/buffers", fixed_count)
        else:
            logger.info("All model parameters have correct dtype (%s)", expected_dtype)

    def _compute_lora_delta_metrics(self) -> dict[str, float | None]:
        """Return aggregate LoRA parameter delta statistics."""
        if self.lora_adapter is None or self.initial_lora_state is None:
            return {
                "lora_delta_l2": None,
                "lora_delta_mean_abs": None,
            }

        total_sq = 0.0
        total_abs = 0.0
        total_count = 0

        current_state = self.lora_adapter.state_dict()
        for key, initial_tensor in self.initial_lora_state.items():
            current_tensor = current_state.get(key)
            if current_tensor is None:
                continue
            current_cpu = current_tensor.detach().float().cpu()
            delta = current_cpu - initial_tensor

            total_sq += float((delta * delta).sum().item())
            total_abs += float(delta.abs().sum().item())
            total_count += int(delta.numel())

        if total_count == 0:
            return {
                "lora_delta_l2": None,
                "lora_delta_mean_abs": None,
            }

        return {
            "lora_delta_l2": math.sqrt(total_sq),
            "lora_delta_mean_abs": total_abs / total_count,
        }
