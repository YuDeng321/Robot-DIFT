"""Frozen Stage-I Robot-DIFT Student features for the Stage-II candidate.

This loader uses the checkpoint's Student U-Net and learned Student timestep.
It loads the SD2.1 VAE and text conditioner, but never constructs the diffusion
Teacher, training adapters, or the Stage-I S2-FPN/query readout.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn


SD21_FEATURE_DIMS = {"us3": 1280, "us6": 1280, "us8": 640, "us10": 320}
_STUDENT_WEIGHT_NAMES = (
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)


def _read_student_state(checkpoint_dir: Path) -> Mapping[str, Tensor]:
    unet_dir = checkpoint_dir / "unet"
    candidates = [unet_dir / name for name in _STUDENT_WEIGHT_NAMES if (unet_dir / name).is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Stage-I Student component weight file under {unet_dir}; "
            f"found {[path.name for path in candidates]}"
        )
    path = candidates[0]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValueError(f"Invalid Student state dictionary in {path}")
    return state


def _read_stage1_metadata(checkpoint_dir: Path, feature_dims: Mapping[str, int]) -> str:
    path = checkpoint_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage-I checkpoint metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("model_type") != "cleandift" or metadata.get("sd_version") != "sd21":
        raise ValueError(f"Checkpoint is not a Stage-I SD2.1 CleanDIFT/Robot-DIFT snapshot: {path}")
    if metadata.get("use_text_condition") is not True:
        raise ValueError(f"Stage-II candidate requires a text-conditioned Stage-I Student: {path}")
    component = metadata.get("components", {}).get("student_unet")
    if not isinstance(component, str) or component.strip("/") != "unet":
        raise ValueError(f"Checkpoint metadata does not identify its Student U-Net component: {path}")
    recorded_dims = metadata.get("feature_dims")
    if not isinstance(recorded_dims, Mapping):
        raise ValueError(f"Checkpoint metadata lacks Student feature dimensions: {path}")
    mismatched = {
        key: (recorded_dims.get(key), channels)
        for key, channels in feature_dims.items()
        if recorded_dims.get(key) != channels
    }
    if mismatched:
        raise ValueError(f"Stage-I Student feature dimensions differ from the candidate: {mismatched}")
    # Snapshots produced before this field was introduced used posterior.sample().
    latent_mode = metadata.get("vae_latent_mode", "sample")
    if latent_mode not in {"sample", "mode"}:
        raise ValueError(f"Invalid Stage-I VAE latent mode in {path}: {latent_mode!r}")
    return latent_mode


def _read_student_timestep(checkpoint_dir: Path) -> Tensor:
    path = checkpoint_dir / "timestep.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Missing learned Stage-I Student timestep: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or set(state) != {"timestep"}:
        raise ValueError(f"Invalid Stage-I Student timestep component: {path}")
    value = state["timestep"]
    if not isinstance(value, Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError(f"Stage-I Student timestep must be one finite tensor: {path}")
    return value.detach().float().reshape(())


class RobotDIFTStudentFeatureExtractor(nn.Module):
    """Return raw ``us3/us6/us8`` maps from a frozen, checkpointed Student.

    Call the module with RGB tensors shaped ``[B, 3, H, W]``. Floating-point
    inputs default to ``[-1, 1]``; pass ``input_range="zero_one"`` for ``[0, 1]``
    or ``input_range="uint8"`` for byte images. Resize images before calling.
    The checkpoint controls VAE sampling: ``mode`` is deterministic, whereas
    legacy ``sample`` checkpoints retain their original sampling behavior.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_repo: str | None = None,
        *,
        device: str = "cuda",
        feature_keys: Sequence[str] = ("us3", "us6", "us8"),
        feature_dims: Mapping[str, int] | None = None,
        vae_latent_mode: str = "auto",
        use_fp32: bool = False,
        student_unet: nn.Module | None = None,
        ae: nn.Module | None = None,
        tokenizer=None,
        text_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if vae_latent_mode not in {"auto", "sample", "mode"}:
            raise ValueError("vae_latent_mode must be 'auto', 'sample', or 'mode'")
        self.feature_keys = tuple(str(key) for key in feature_keys)
        if not self.feature_keys or len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("feature_keys must be nonempty and unique")
        dimensions = dict(SD21_FEATURE_DIMS if feature_dims is None else feature_dims)
        if any(key not in dimensions or int(dimensions[key]) < 1 for key in self.feature_keys):
            raise ValueError("feature_dims must specify a positive channel count for every feature key")
        self.feature_dims = {key: int(dimensions[key]) for key in self.feature_keys}

        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_latent_mode = _read_stage1_metadata(checkpoint, self.feature_dims)
        if vae_latent_mode != "auto" and vae_latent_mode != checkpoint_latent_mode:
            raise ValueError(
                f"Requested VAE latent mode {vae_latent_mode!r} differs from Stage-I "
                f"checkpoint metadata {checkpoint_latent_mode!r}: {checkpoint}"
            )
        self.vae_latent_mode = checkpoint_latent_mode
        student_timestep = _read_student_timestep(checkpoint)
        student_state = _read_student_state(checkpoint)

        injected = (student_unet, ae, tokenizer, text_encoder)
        if any(item is not None for item in injected) and not all(item is not None for item in injected):
            raise ValueError("Provide all Student/VAE/tokenizer/text modules together for injection")
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {target}")
        if all(item is not None for item in injected):
            self.student_unet = student_unet
            self.ae = ae
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
        else:
            if not model_repo or not os.path.isdir(model_repo):
                raise FileNotFoundError(f"Stage-II requires a local SD2.1 model_repo: {model_repo}")
            for env_name in ("ROBOT_DIFT_MODEL_DIR", "ROBOT_DIFT_CLEANDIFT_MODEL_REPO"):
                repo_override = os.environ.get(env_name)
                if repo_override and Path(repo_override).resolve() != Path(model_repo).resolve():
                    raise ValueError(
                        f"{env_name} differs from Stage-II model_repo; "
                        "the VAE wrapper would load a different SD2.1 snapshot"
                    )
            from transformers import CLIPTextModel, CLIPTokenizer

            from agents.encoders.cleandift.src.ae import AutoencoderKL
            from agents.encoders.cleandift.src.sd_feature_extraction import SD21UNetFeatureExtractor

            self.student_unet = SD21UNetFeatureExtractor()
            self.ae = AutoencoderKL(repo=model_repo)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_repo, subfolder="tokenizer", local_files_only=True
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_repo, subfolder="text_encoder", local_files_only=True
            )

        if self.vae_latent_mode == "mode" and not hasattr(getattr(self.ae, "ae", None), "encode"):
            raise TypeError("VAE mode extraction requires an AutoencoderKL-style .ae.encode posterior")

        try:
            self.student_unet.load_state_dict(student_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Stage-I Student U-Net checkpoint failed strict loading: {checkpoint}: {exc}"
            ) from exc
        del student_state
        self.register_buffer("student_timestep", student_timestep)

        dtype_setting = (
            os.environ.get("ROBOT_DIFT_CLEANDIFT_BACKBONE_DTYPE")
            or os.environ.get("ROBOT_DIFT_DROID_AMP_DTYPE")
            or ""
        ).lower()
        if use_fp32 or target.type != "cuda" or dtype_setting in {"float32", "fp32", "full"}:
            self.model_dtype = torch.float32
        elif dtype_setting in {"float16", "fp16", "half"}:
            self.model_dtype = torch.float16
        else:
            self.model_dtype = torch.bfloat16
        self.to(device=target, dtype=self.model_dtype)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def extract(
        self,
        images: Tensor,
        captions: Sequence[str] | str | None = None,
        *,
        input_range: str = "minus_one_one",
    ) -> dict[str, Tensor]:
        """Extract frozen feature maps without constructing a Teacher or readout.

        Images must be RGB, finite, and have spatial dimensions divisible by
        64. A string caption is shared by the batch; a sequence supplies one
        caption per image. Outputs are float32 tensors on the model's device.
        """
        if not isinstance(images, Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError("Student images must have nonempty shape [B,3,H,W]")
        if any(size < 64 or size % 64 for size in images.shape[-2:]):
            raise ValueError("Image height and width must be positive multiples of 64")
        if input_range == "uint8":
            if images.dtype != torch.uint8:
                raise TypeError("input_range='uint8' requires torch.uint8 images")
            normalized = images.float().div(127.5).sub(1.0)
        elif input_range in {"zero_one", "minus_one_one"}:
            if not images.is_floating_point():
                raise TypeError("Floating-point input ranges require floating-point images")
            lower = 0.0 if input_range == "zero_one" else -1.0
            if not bool((torch.isfinite(images) & (images >= lower) & (images <= 1.0)).all()):
                raise ValueError(f"Images must be finite and within [{lower}, 1.0]")
            normalized = images.float()
            if input_range == "zero_one":
                normalized = normalized.mul(2.0).sub(1.0)
        else:
            raise ValueError("input_range must be 'minus_one_one', 'zero_one', or 'uint8'")
        return self._encode_backbone(normalized, captions)

    def forward(
        self,
        images: Tensor,
        captions: Sequence[str] | str | None = None,
        *,
        input_range: str = "minus_one_one",
    ) -> dict[str, Tensor]:
        return self.extract(images, captions, input_range=input_range)

    def _encode_latents(self, images: Tensor) -> Tensor:
        if self.vae_latent_mode == "sample":
            return self.ae.encode(images)
        vae = getattr(self.ae, "ae", None)
        if vae is None or not hasattr(vae, "encode"):
            raise TypeError("VAE mode extraction requires an AutoencoderKL-style .ae.encode posterior")
        posterior = vae.encode(images, return_dict=False)[0]
        if not hasattr(posterior, "mode"):
            raise TypeError("VAE posterior does not expose .mode()")
        return (posterior.mode() - self.ae.shift) * self.ae.scale

    def _prompt_embeds(self, captions: list[str], device: torch.device, dtype: torch.dtype) -> Tensor:
        encoded = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        kwargs = {"input_ids": input_ids}
        if getattr(self.text_encoder.config, "use_attention_mask", False):
            kwargs["attention_mask"] = encoded["attention_mask"].to(device)
        return self.text_encoder(**kwargs)[0].to(dtype=dtype)

    @torch.no_grad()
    def _encode_backbone(self, images: Tensor, captions: Sequence[str] | str | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Student images must have shape [B,3,H,W]")
        batch = images.shape[0]
        if captions is None:
            prompts = [""] * batch
        elif isinstance(captions, str):
            prompts = [captions] * batch
        else:
            prompts = list(captions)
            if len(prompts) != batch or any(not isinstance(value, str) for value in prompts):
                raise ValueError(f"Expected {batch} text prompts for the Student images")

        device = self.student_timestep.device
        dtype = next(self.student_unet.parameters()).dtype
        images = images.to(device=device, dtype=dtype).contiguous()
        latents = self._encode_latents(images)
        text_hidden = self._prompt_embeds(prompts, device, latents.dtype)
        timesteps = self.student_timestep.expand(batch).to(device=device)
        requested_keys = (
            {"requested_feature_keys": self.feature_keys}
            if getattr(self.student_unet, "supports_requested_feature_keys", False)
            else {}
        )
        maps = self.student_unet(
            latents,
            timesteps,
            encoder_hidden_states=text_hidden,
            added_cond_kwargs={},
            **requested_keys,
        )
        if not isinstance(maps, Mapping):
            raise TypeError("Student U-Net must return a feature-map dictionary")
        selected = {}
        for key in self.feature_keys:
            tensor = maps.get(key)
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                raise KeyError(f"Student U-Net did not return a 4-D {key} feature map")
            if tensor.shape[0] != batch or tensor.shape[1] != self.feature_dims[key]:
                raise ValueError(f"Student {key} feature shape differs from checkpoint metadata")
            selected[key] = tensor.float()
        return selected
