"""ModelAdapter - model loading and conditioning construction"""
import logging
from pathlib import Path
from typing import Any, List, Mapping, Tuple, cast

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

def load_checkpoint_with_text_encoder_conversion(ckpt_path: str, device: torch.device) -> Tuple:
    """Load checkpoint and convert text encoder key format for compatibility.

    Handles models where text_model structure differs from standard format.
    """
    text_encoder_key_replacements = [
        ("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings."),
        ("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder."),
        ("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm."),
    ]

    checkpoint = None
    try:
        state_dict = dict(cast(Mapping[str, Any], load_single_file_checkpoint(ckpt_path)))
    except Exception:
        if Path(ckpt_path).suffix == ".safetensors":
            state_dict = dict(load_file(ckpt_path))
        else:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "state_dict" in checkpoint:
                state_dict = dict(checkpoint["state_dict"])
            else:
                state_dict = dict(checkpoint)
                checkpoint = None

    key_reps = []
    for rep_from, rep_to in text_encoder_key_replacements:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from):]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict

class ModelAdapter:
    """Model adapter base class"""

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self) -> Tuple[nn.Module, ...]:
        """Load model components (VAE, UNet, TE)."""
        raise NotImplementedError

    def get_target_modules(self) -> List[str]:
        """Return LoRA injection target modules"""
        raise NotImplementedError

    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """Encode text"""
        raise NotImplementedError

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image to latent"""
        raise NotImplementedError

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> torch.Tensor:
        """Run inference, return image tensor [3, H, W] in [0, 1] on CPU."""
        raise NotImplementedError

    def _is_checkpoint(self, path: Path) -> bool:
        """Determine if a path is a model checkpoint (e.g. .safetensors)"""
        return path.suffix in {".safetensors", ".pt", ".pth"}


class SD15ModelAdapter(ModelAdapter):
    """SD1.5 model adapter (MVP implementation)"""

    def __init__(self, model_name_or_path: str = "runwayml/stable-diffusion-v1-5"):
        super().__init__(model_name_or_path)
        self.vae: AutoencoderKL | None = None
        self.unet: UNet2DConditionModel | None = None
        self.text_encoder: CLIPTextModel | None = None
        self.tokenizer: CLIPTokenizer | None = None

    def load_models(self) -> Tuple[nn.Module, ...]:
        """Load SD1.5 model components (VAE, UNet, text encoder)."""
        if self._is_checkpoint(Path(self.model_name_or_path)):
            _, state_dict = load_checkpoint_with_text_encoder_conversion(
                self.model_name_or_path,
                self.device,
            )

            unet_config = create_unet_diffusers_config(state_dict, image_size=512)
            converted_unet_state_dict = convert_ldm_unet_checkpoint(state_dict, unet_config)
            self.unet = UNet2DConditionModel(**unet_config)
            nn.Module.to(self.unet, self.device)
            unet_info = self.unet.load_state_dict(converted_unet_state_dict)
            logger.info("Loaded UNet with info: %s", unet_info)

            vae_config = create_vae_diffusers_config(state_dict, image_size=512)
            converted_vae_state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config)
            self.vae = AutoencoderKL(**vae_config)
            nn.Module.to(self.vae, self.device)
            vae_info = self.vae.load_state_dict(converted_vae_state_dict)
            logger.info("Loaded VAE with info: %s", vae_info)

            clip_config = CLIPTextConfig(
                vocab_size=49408,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=77,
                hidden_act="quick_gelu",
                layer_norm_eps=1e-05,
                dropout=0.0,
                attention_dropout=0.0,
                initializer_range=0.02,
                initializer_factor=1.0,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                model_type="clip_text_model",
                projection_dim=768,
                torch_dtype="float32",
            )
            converted_text_encoder_checkpoint = cast(
                Mapping[str, Any],
                convert_ldm_clip_checkpoint(state_dict),
            )

            self.text_encoder = CLIPTextModel(clip_config)
            nn.Module.to(self.text_encoder, self.device)
            clip_info = self.text_encoder.load_state_dict(converted_text_encoder_checkpoint)
            logger.info("Loaded CLIP Text Encoder with info: %s", clip_info)

            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        else:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
            )
            pipe.to(self.device)

            self.vae = pipe.vae
            self.unet = pipe.unet
            self.text_encoder = pipe.text_encoder
            self.tokenizer = cast(CLIPTokenizer, pipe.tokenizer)

        assert self.vae is not None
        assert self.unet is not None
        assert self.text_encoder is not None
        return self.vae, self.unet, self.text_encoder

    def get_target_modules(self) -> List[str]:
        """SD1.5 LoRA target modules"""
        return ["to_q", "to_k", "to_v", "to_out.0"]

    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """Encode text to embedding"""
        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.text_encoder is not None

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        return embeddings

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image to latent"""
        self._ensure_loaded()
        assert self.vae is not None

        with torch.no_grad():
            image_tensor = images.to(device=self.device, dtype=self.vae.dtype)
            encoded_output = cast(Any, self.vae.encode(image_tensor))
            if hasattr(encoded_output, "latent_dist"):
                latent_dist = encoded_output.latent_dist
            else:
                latent_dist = encoded_output[0]

            if hasattr(latent_dist, "sample"):
                latent_sample = latent_dist.sample()
            else:
                latent_sample = latent_dist

            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
            latents = cast(torch.Tensor, latent_sample) * scaling_factor
        return latents

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        self._ensure_loaded()
        assert self.vae is not None

        with torch.no_grad():
            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
            latent_tensor = latents.to(device=self.device, dtype=self.vae.dtype) / scaling_factor
            decoded_output = cast(Any, self.vae.decode(cast(Any, latent_tensor)))
            if hasattr(decoded_output, "sample"):
                decoded = decoded_output.sample
            else:
                decoded = decoded_output[0]
        return cast(torch.Tensor, decoded)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> torch.Tensor:
        """Run DDIM inference via StableDiffusionPipeline, return [3,H,W] tensor in [0,1] on CPU."""
        self._ensure_loaded()
        assert self.vae is not None
        assert self.unet is not None
        assert self.text_encoder is not None
        assert self.tokenizer is not None

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=cast(Any, scheduler),
            safety_checker=None,  # type: ignore[arg-type]
            feature_extractor=None,  # type: ignore[arg-type]
            requires_safety_checker=False,
        ).to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = cast(
            Any,
            pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ),
        )
        pil_image = result.images[0]
        del pipe
        return TF.to_tensor(pil_image)

    def _ensure_loaded(self) -> None:
        """Lazy-load SD1.5 models if not loaded yet."""
        if self.vae is None or self.unet is None or self.text_encoder is None:
            self.load_models()
