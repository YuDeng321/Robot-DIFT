import os
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import einops
from .compat import patch_transformers_for_diffusers_compat

patch_transformers_for_diffusers_compat()
from diffusers import DiffusionPipeline
from jaxtyping import Float, Int
from pydoc import locate
from typing import Literal, Sequence
from .layers import FeedForwardBlock, FourierFeatures, Linear, MappingNetwork
from .model_repo import resolve_sd_model_repo
from .min_sd15 import SD15UNetModel
from .min_sd21 import SD21UNetModel


class SD15UNetFeatureExtractor(SD15UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }


class SD21UNetFeatureExtractor(SD21UNetModel):
    supports_requested_feature_keys = True

    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        requested_feature_keys = kwargs.pop("requested_feature_keys", None)
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        if requested_feature_keys is not None and set(requested_feature_keys).issubset(
            {"mid", "us1", "us2", "us3", "us4", "us5", "us6", "us7", "us8", "us9"}
        ):
            # Deployed us3/us6/us8 maps cannot depend on the subsequent up block.
            # Training alignment still requests us10 and executes the full U-Net.
            return {
                "mid": sample_mid, "us1": us1, "us2": us2, "us3": us3,
                "us4": us4, "us5": us5, "us6": us6,
                "us7": us7, "us8": us8, "us9": us9,
            }

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }

class FeedForwardBlockCustom(FeedForwardBlock):
    def __init__(self, d_model: int, d_ff: int, d_cond_norm: int = None, norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True):
        super().__init__(d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm)
        if not use_gating:
            self.up_proj = LinearSwish(d_model, d_ff, bias=False)
        if norm_type == 'FiLM':
            self.norm = FiLMNorm(d_model, d_cond_norm)

class FFNStack(nn.Module):
    def __init__(self, dim: int, depth: int, ffn_expansion: float, dim_cond: int,
                 norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FeedForwardBlockCustom(d_model=dim, d_ff=int(dim * ffn_expansion), d_cond_norm=dim_cond, norm_type=norm_type, use_gating=use_gating)
             for _ in range(depth)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond_norm=cond)
        return x

class FiLMNorm(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = Linear(cond_features, features * 2, bias=False)
        self.feature_dim = features

    def forward(self, x, cond):
        B, _, D = x.shape
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # broadcast scale and shift across all features
        scale = scale.view(B, 1, D)
        shift = shift.view(B, 1, D)
        return scale * x + shift

class LinearSwish(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.silu(super().forward(x))


class ArgSequential(nn.Module):  # Utility class to enable instantiating nn.Sequential instances with Hydra
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

class StableFeatureAligner(nn.Module):
    """Frozen noisy-input SD Teacher, clean-input Student, and training-only adapters.

    Stage I trains the Student so that ``adapter_l(Student_l(x_0), t)`` matches
    ``Teacher_l(x_t, t)`` for every aligned map ``l``. The public interface
    (``get_features`` / ``student_maps``) exposes the raw Student maps only.
    """

    def __init__(
        self,
        ae: nn.Module,
        mapping,
        adapter_layer_class: str,
        feature_dims: dict[str, int],
        feature_extractor_cls: str,
        sd_version: Literal["sd15", "sd21"],
        adapter_layer_params: dict = {},
        use_text_condition: bool = False,
        t_min: int = 1,
        t_max: int = 999,
        t_max_model: int = 999,
        num_t_stratification_bins: int = 3,
        alignment_loss: Literal["cossim", "mse", "l1"] = "cossim",
        train_unet: bool = True,
        train_adapter: bool = True,
        t_init: int = 261,
        learn_timestep: bool = False,
        val_dataset: torch.utils.data.Dataset | None = None,
        val_t: int = 261,
        val_feature_key: str = "us6",
        val_chunk_size: int = 10,
        use_adapters: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.ae = ae
        self.ae.eval()
        self.ae.requires_grad_(False)
        self.sd_version = sd_version
        self.val_t = val_t
        self.val_feature_key = val_feature_key
        self.val_dataset = val_dataset
        self.val_chunk_size = val_chunk_size
        self.use_adapters = use_adapters
        self.device = device

        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                cuda_device = int(device.split(":")[1])
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized():
                        torch.cuda.set_device(cuda_device)
                except ImportError:
                    torch.cuda.set_device(cuda_device)
        target_device = device

        if sd_version == "sd15":
            default_repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif sd_version == "sd21":
            default_repo = "sd2-community/stable-diffusion-2-1-base"
        else:
            raise ValueError(f"Invalid SD version: {sd_version}")
        self.repo = resolve_sd_model_repo(default_repo)

        self._compile_enabled = os.environ.get("TORCHDYNAMO_DISABLE", "").lower() not in {"1", "true"}

        self.mapping = None
        if use_adapters:
            self.time_emb = FourierFeatures(1, mapping.width)
            self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
            self.mapping = MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout)
            if self._compile_enabled:
                try:
                    self.mapping.compile()
                except Exception:
                    warnings.warn("Failed to compile MappingNetwork, falling back to eager.")

        if use_adapters:
            self.adapters = nn.ModuleDict()
            for k, dim in feature_dims.items():
                self.adapters[k] = locate(adapter_layer_class)(dim=dim, **adapter_layer_params)
                self.adapters[k].requires_grad_(train_adapter)
        self.feature_keys = tuple(feature_dims.keys())

        self.unet_feature_extractor_base = locate(feature_extractor_cls)().to(target_device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            # Load in float32: the Teacher and the initial Student are exact
            # copies of the released SD weights.
            self.pipe = DiffusionPipeline.from_pretrained(
                self.repo,
                torch_dtype=torch.float32,
                use_safetensors=True,
            ).to(target_device)
        self.unet_feature_extractor_base.load_state_dict(self.pipe.unet.state_dict())
        self.unet_feature_extractor_base.eval()
        self.unet_feature_extractor_base.requires_grad_(False)
        if self._compile_enabled:
            try:
                self.unet_feature_extractor_base.compile()
            except Exception:
                warnings.warn("Failed to compile base UNet, falling back to eager.")

        self.unet_feature_extractor_cleandift = locate(feature_extractor_cls)().to(target_device)
        self.unet_feature_extractor_cleandift.load_state_dict(
            {k: v.detach().clone() for k, v in self.unet_feature_extractor_base.state_dict().items()}
        )

        if train_unet or learn_timestep:
            self.unet_feature_extractor_cleandift.train()
        else:
            self.unet_feature_extractor_cleandift.eval()
        self.unet_feature_extractor_cleandift.requires_grad_(train_unet)
        if self._compile_enabled:
            try:
                self.unet_feature_extractor_cleandift.compile()
            except Exception:
                warnings.warn("Failed to compile finetune UNet, falling back to eager.")

        self.use_text_condition = use_text_condition
        if self.use_text_condition:
            text_encoder = getattr(self.pipe, "text_encoder", None)
            if isinstance(text_encoder, nn.Module):
                text_encoder.eval()
                text_encoder.requires_grad_(False)
            if self._compile_enabled:
                try:
                    self.pipe.text_encoder.compile()
                except Exception:
                    warnings.warn("Failed to compile text encoder, falling back to eager.")
        else:
            prompt_embeds_dict = self.get_prompt_embeds([""])
            self._empty_prompt_embeds = prompt_embeds_dict["prompt_embeds"]
            del self.pipe.text_encoder

        del self.pipe.unet, self.pipe.vae

        self.t_min = t_min
        self.t_max = t_max
        self.t_max_model = t_max_model
        self.num_t_stratification_bins = num_t_stratification_bins
        if alignment_loss not in {"mse", "l1", "cossim"}:
            raise ValueError(f"Invalid alignment loss type: {alignment_loss}")
        self.alignment_loss = alignment_loss
        self.timestep = nn.Parameter(
            torch.tensor(float(t_init), requires_grad=learn_timestep), requires_grad=learn_timestep
        )

    # ------------------------------------------------------------------
    # Conditioning and latents
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_prompt_embeds(self, prompt: list[str]) -> dict[str, torch.Tensor | None]:
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=torch.device(self.device),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": self.prompt_embeds}

    def _prompt_embeddings(self, prompts: list[str], device, dtype) -> torch.Tensor:
        """Encode each distinct prompt once; views and frames share their caption."""
        if not self.use_text_condition:
            return einops.repeat(
                self._empty_prompt_embeds, "b ... -> (B b) ...", B=len(prompts)
            ).to(dtype=dtype, device=device)
        unique = list(dict.fromkeys(prompts))
        embeds = self.get_prompt_embeds(unique)["prompt_embeds"]
        if len(unique) != len(prompts):
            lookup = {prompt: index for index, prompt in enumerate(unique)}
            index = torch.tensor([lookup[prompt] for prompt in prompts], device=embeds.device)
            embeds = embeds.index_select(0, index)
        return embeds.to(dtype=dtype, device=device)

    def _get_unet_conds(self, prompts: list[str], device, dtype, N_T) -> dict[str, torch.Tensor]:
        embeds = self._prompt_embeddings(list(prompts), device, dtype)
        if N_T != 1:
            embeds = einops.repeat(embeds, "B ... -> (B N_T) ...", N_T=N_T)
        return {"encoder_hidden_states": embeds, "added_cond_kwargs": {}}

    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Map images in [-1, 1] to scaled SD latents (the Student's clean input).

        Latents are float32: under bf16 autocast the posterior mean would
        otherwise stay bf16, and ``add_noise`` would then round the noise
        schedule (at t=1 the Teacher would see no noise at all).
        """
        return self.ae.encode(x).float()

    def _adapter_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Adapter conditioning for integer timesteps, computed in float32.

        bf16 cannot represent t/999 finely enough (e.g. 998 and 999 collapse),
        so the time embedding and mapping network run outside autocast.
        """
        with torch.autocast(device_type=timesteps.device.type, enabled=False):
            return self.mapping(
                self.time_in_proj(self.time_emb(timesteps.float().reshape(-1, 1) / self.t_max_model))
            )

    @staticmethod
    def _input_for(module: nn.Module, sample: torch.Tensor) -> torch.Tensor:
        """Match a U-Net's parameter dtype (noise is already mixed in float32)."""
        parameter = next(module.parameters(), None)
        return sample if parameter is None else sample.to(dtype=parameter.dtype)

    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        """Stratified Teacher timesteps ``[B, N_T]`` in ``[t_min, t_max)``."""
        t_range_per_bin = (self.t_max - self.t_min) / self.num_t_stratification_bins
        return (
            self.t_min
            + torch.rand((batch_size, self.num_t_stratification_bins), device=device) * t_range_per_bin
            + torch.arange(0, self.num_t_stratification_bins, device=device)[None, :] * t_range_per_bin
        ).long()

    # ------------------------------------------------------------------
    # Student and Teacher
    # ------------------------------------------------------------------

    def student_maps(self, x_0: torch.Tensor, unet_conds: dict, feature_keys=None) -> dict[str, torch.Tensor]:
        """Single clean-input Student pass at its learned timestep.

        When every requested key precedes ``us10`` the final up block is skipped,
        which is exact because those maps do not depend on it.
        """
        batch = x_0.shape[0]
        student_t = torch.ones((batch,), device=x_0.device, dtype=self.timestep.dtype) * self.timestep
        requested = {}
        if feature_keys is not None and getattr(
            self.unet_feature_extractor_cleandift, "supports_requested_feature_keys", False
        ):
            requested = {"requested_feature_keys": list(feature_keys)}
        student = self.unet_feature_extractor_cleandift
        return student(self._input_for(student, x_0), student_t, **unet_conds, **requested)

    def _alignment_term(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        teacher = teacher.detach()
        if self.alignment_loss == "mse":
            return F.mse_loss(student, teacher)
        if self.alignment_loss == "l1":
            return F.l1_loss(student, teacher)
        return -F.cosine_similarity(student, teacher, dim=-1).mean()

    def _loss_name(self, key: str) -> str:
        return f"{'neg_cossim' if self.alignment_loss == 'cossim' else self.alignment_loss}_{key}"

    def alignment_terms(
        self,
        x_0: torch.Tensor,
        unet_conds: dict,
        student_features: dict[str, torch.Tensor],
        apply_adapter: bool | None = None,
        return_raw_cosine: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-map Teacher alignment terms for an existing Student pass.

        ``student_features`` holds ``[B, C, H, W]`` maps computed from ``x_0``.
        The Student is evaluated once; each of the ``N_T`` sampled Teacher
        timesteps reuses it, which equals ``N_T`` identical Student passes.
        """
        missing = [key for key in self.feature_keys if key not in student_features]
        if missing:
            raise KeyError(f"Student pass is missing alignment maps: {missing}")
        B = x_0.shape[0]
        t = self.sample_timesteps(B, x_0.device)
        N_T = t.shape[1]
        t_flat = einops.rearrange(t, "B N_T -> (B N_T)")

        with torch.no_grad():
            teacher_conds = {
                "encoder_hidden_states": einops.repeat(
                    unet_conds["encoder_hidden_states"], "B ... -> (B N_T) ...", N_T=N_T
                ),
                "added_cond_kwargs": {},
            }
            x_0_rep = einops.repeat(x_0, "B ... -> (B N_T) ...", N_T=N_T)
            noise = torch.randn_like(x_0_rep)
            x_t = self.pipe.scheduler.add_noise(x_0_rep, noise, t_flat)
            teacher = self.unet_feature_extractor_base
            requested = {}
            if getattr(teacher, "supports_requested_feature_keys", False):
                requested = {"requested_feature_keys": list(self.feature_keys)}
            teacher_out = teacher(self._input_for(teacher, x_t), t_flat, **teacher_conds, **requested)
            feats_teacher = {
                key: einops.rearrange(teacher_out[key], "(B N_T) D H W -> B N_T (H W) D", B=B, N_T=N_T)
                for key in self.feature_keys
            }
            del teacher_out

        raw_student = {
            key: einops.repeat(student_features[key], "B D H W -> B N_T (H W) D", N_T=N_T)
            for key in self.feature_keys
        }

        if apply_adapter is None:
            apply_adapter = self.use_adapters
        apply_adapter = bool(apply_adapter and self.use_adapters and hasattr(self, "adapters"))
        aligned_student = raw_student
        if apply_adapter:
            map_cond = self._adapter_condition(t_flat)
            aligned_student = {
                key: einops.rearrange(
                    self.adapters[key](einops.rearrange(value, "B N_T L D -> (B N_T) L D"), cond=map_cond),
                    "(B N_T) L D -> B N_T L D",
                    B=B,
                    N_T=N_T,
                )
                for key, value in raw_student.items()
            }

        losses = {
            self._loss_name(key): self._alignment_term(aligned_student[key], feats_teacher[key])
            for key in self.feature_keys
        }
        metrics = {}
        if return_raw_cosine:
            with torch.no_grad():
                for key in self.feature_keys:
                    metrics[f"raw_cosine_{key}"] = F.cosine_similarity(
                        raw_student[key].float(), feats_teacher[key].float(), dim=-1
                    ).mean()
        return losses, metrics

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
        caption: list[str],
        apply_adapter: bool | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Alignment terms for images in [-1, 1] (Student and Teacher from scratch)."""
        if kwargs:
            raise TypeError(f"Unsupported StableFeatureAligner.forward arguments: {sorted(kwargs)}")
        unet_conds = self._get_unet_conds(caption, x.device, x.dtype, 1)
        x_0 = self.encode_latents(x)
        student = self.student_maps(x_0, unet_conds)
        losses, _ = self.alignment_terms(x_0, unet_conds, student, apply_adapter=apply_adapter)
        return losses

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _normalize_feat_key(self, feat_key):
        if feat_key is None:
            return None, None, True
        if isinstance(feat_key, str):
            return [feat_key], feat_key, False
        if isinstance(feat_key, Sequence):
            keys = [str(k) for k in feat_key]
            if len(keys) == 0:
                raise ValueError("feat_key sequence must be non-empty.")
            if len(keys) == 1:
                key = keys[0]
                return keys, key, False
            return keys, None, True
        raise TypeError(f"Unsupported feat_key type: {type(feat_key)}")

    def _prepare_timestep_vector(self, t, batch_size: int, device: torch.device) -> torch.Tensor:
        if t is None:
            base = self.timestep.detach().to(device=device)
            return base.repeat(batch_size)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=self.timestep.dtype)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.shape[0] != batch_size:
            raise ValueError(
                f"Expected timestep tensor with first dimension {batch_size}, got {tuple(t.shape)}"
            )
        return t.to(device=device, dtype=self.timestep.dtype)

    @staticmethod
    def _finalize_feature_output(feature_dict: dict[str, torch.Tensor], single_key: str | None, return_dict: bool):
        if not return_dict:
            key = single_key if single_key is not None else next(iter(feature_dict))
            return feature_dict[key]
        return feature_dict

    def get_features(
        self,
        x: Float[torch.Tensor, "b c h w"],
        caption: list[str] | None,
        t: Int[torch.Tensor, "b"] | None,
        feat_key,
        use_base_model: bool = False,
        input_pure_noise: bool = False,
        eps: torch.Tensor = None,
        apply_adapter: bool | None = None,
    ):
        keys, single_key, return_dict = self._normalize_feat_key(feat_key)
        B = x.shape[0]
        if caption is None:
            caption = [""] * B
        unet_conds = self._get_unet_conds(caption, x.device, x.dtype, 1)
        x_0 = self.encode_latents(x)

        if use_base_model:
            if t is None:
                raise ValueError("Base model feature extraction requires timestep tensor `t`.")
            eps = torch.randn_like(x_0) if eps is None else eps
            if input_pure_noise:
                if not torch.allclose(t, torch.full_like(t, 999)):
                    raise ValueError("Pure noise input expects all timesteps to be 999.")
                x_t = eps
            else:
                x_t = self.pipe.scheduler.add_noise(x_0, eps, t)

            teacher = self.unet_feature_extractor_base
            feats = teacher(self._input_for(teacher, x_t), t, **unet_conds)
            if keys is None:
                feature_dict = dict(feats)
            else:
                feature_dict = {}
                for key in keys:
                    if key not in feats:
                        raise KeyError(f"Feature key '{key}' not available in base model outputs.")
                    feature_dict[key] = feats[key]
            return self._finalize_feature_output(feature_dict, single_key, return_dict)

        raw_feats = self.student_maps(x_0, unet_conds, keys)
        key_iterable = list(raw_feats.keys()) if keys is None else list(keys)

        if apply_adapter is None:
            apply_adapter = t is not None
        apply_adapter = bool(self.use_adapters and hasattr(self, "adapters") and apply_adapter)
        cond = None
        if apply_adapter:
            cond = self._adapter_condition(self._prepare_timestep_vector(t, B, x.device))

        feature_dict = {}
        for key in key_iterable:
            if key not in raw_feats:
                raise KeyError(f"Feature key '{key}' not found in CleanDIFT features.")
            tensor = raw_feats[key]
            if apply_adapter:
                tensor = einops.rearrange(tensor, "B D H W -> B (H W) D")
                tensor = self.adapters[key](tensor, cond=cond)
                tensor = einops.rearrange(tensor, "B (H W) D -> B D H W", H=raw_feats[key].shape[-2])
            feature_dict[key] = tensor

        return self._finalize_feature_output(feature_dict, single_key, return_dict)
