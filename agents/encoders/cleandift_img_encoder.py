"""
Robot-DIFT Student encoder (SD2.1 U-Net) with training-only Teacher alignment.

The Student exposes raw decoder maps (``us3/us6/us8`` for the released
interface). During Stage I the same module owns the frozen SD2.1 Teacher and
the timestep-conditioned adapters used by the alignment loss. The built-in
S2-FPN/learnable-query readout is selected with ``readout="legacy"``; the
paper readout lives in :mod:`agents.encoders.robot_dift_paper_readout`.
"""

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import logging
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple
from omegaconf import OmegaConf, ListConfig

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pydoc import locate

try:
    from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner
    from agents.encoders.cleandift.src.ae import AutoencoderKL
    from agents.encoders.cleandift.src.utils import MappingSpec
except ImportError:
    from cleandift.src.sd_feature_extraction import StableFeatureAligner
    from cleandift.src.ae import AutoencoderKL
    from cleandift.src.utils import MappingSpec


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """2D sin-cos absolute position embedding, returns [1, h*w, dim]."""
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    dim_half = dim // 2
    dim_y, dim_x = dim_half, dim - dim_half
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    def _pe(vec, d):
        L = max(1, d // 2)
        omega = 1.0 / (10000 ** (torch.arange(L, device=device, dtype=dtype) / max(1, L - 1)))
        out = torch.einsum("n,d->nd", vec.flatten(), omega)
        pe = torch.cat([out.sin(), out.cos()], dim=-1)
        if pe.shape[-1] < d:
            pe = F.pad(pe, (0, d - pe.shape[-1]))
        return pe

    pey, pex = _pe(yy, dim_y), _pe(xx, dim_x)
    pe = torch.cat([pey, pex], dim=-1).unsqueeze(0)
    return pe


def _make_group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Build GroupNorm with automatic group size selection."""
    for g in [num_groups, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def _checkpoint_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path.rstrip(os.sep))
    suffix = name.split("checkpoint-", 1)[-1]
    epoch = suffix.split("-", 1)[0]
    return (int(epoch) if epoch.isdigit() else -1, name)


def reduce_alignment_loss_terms(loss_dict: Dict[str, torch.Tensor], reduction: str) -> torch.Tensor:
    """Reduce Student/Teacher map losses with an explicit layer convention.

    CleanDIFT returns negative cosine terms. The paper's ``1 - cosine`` adds
    a constant one per map; this changes reported loss but not gradients.
    ``mean`` retains historical checkpoint training behavior.
    """
    if not loss_dict:
        raise ValueError("Alignment returned no feature-map loss terms")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported alignment layer reduction: {reduction}")
    terms = [value + 1.0 if reduction == "sum" and name.startswith("neg_cossim_") else value
             for name, value in loss_dict.items()]
    loss = sum(terms)
    return loss / len(terms) if reduction == "mean" else loss


STUDENT_UNET_WEIGHT_NAMES = (
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)


def _student_unet_weight_file(path: str) -> Optional[str]:
    """Return the single Student U-Net weight file of a component checkpoint."""
    unet_dir = os.path.join(path, "unet")
    found = [os.path.join(unet_dir, name) for name in STUDENT_UNET_WEIGHT_NAMES
             if os.path.isfile(os.path.join(unet_dir, name))]
    if len(found) > 1:
        raise ValueError(f"Ambiguous Student U-Net weights under {unet_dir}: {found}")
    return found[0] if found else None


def _has_dift_component_checkpoint(path: str) -> bool:
    return _student_unet_weight_file(path) is not None


# =============================================================================
# S2-FPN Components
# =============================================================================

class S2FPNFuseBlock(nn.Module):
    """Single fusion block for S2-FPN: Conv + GroupNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = _make_group_norm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class S2FPNBidirectionalFusion(nn.Module):
    """
    S2-FPN Bidirectional Feature Fusion Module.

    Top-Down Path: High semantic → Low semantic (spatial refinement)
    Bottom-Up Path: High resolution → Low resolution (semantic enhancement)

    Args:
        feature_dims: Dict mapping feature keys to channel dimensions
        feature_keys: List of feature keys in order (low-res to high-res)
        fpn_dim: Unified channel dimension after projection
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        feature_keys: List[str],
        fpn_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.feature_keys = feature_keys
        self.fpn_dim = fpn_dim

        # 1. Per-scale projection layers (keep original resolution)
        self.proj_layers = nn.ModuleDict()
        for key in feature_keys:
            c_in = feature_dims[key]
            self.proj_layers[key] = nn.Sequential(
                nn.Conv2d(c_in, fpn_dim, kernel_size=1, bias=True),
                _make_group_norm(fpn_dim),
                nn.GELU(),
            ).to(device)

        # 2. Top-Down fusion blocks (semantic enhancement to geometric layers)
        # Number of fusions = len(feature_keys) - 1
        self.td_fuse_blocks = nn.ModuleList([
            S2FPNFuseBlock(fpn_dim * 2, fpn_dim).to(device)
            for _ in range(len(feature_keys) - 1)
        ])

        # 3. Bottom-Up fusion blocks (geometric detail back to semantic layers)
        self.bu_fuse_blocks = nn.ModuleList([
            S2FPNFuseBlock(fpn_dim * 2, fpn_dim).to(device)
            for _ in range(len(feature_keys) - 1)
        ])

        # 4. Final fusion: combine top-down and bottom-up
        self.final_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, kernel_size=1, bias=True),
            _make_group_norm(fpn_dim),
            nn.GELU(),
        ).to(device)

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_maps: Dict of {key: [B, C, H, W]} tensors

        Returns:
            Fused feature map [B, fpn_dim, H_max, W_max]
        """
        # Record original resolutions before any processing
        original_resolutions = {key: feature_maps[key].shape[-1] for key in self.feature_keys}

        # Sort keys by resolution: low-res (high semantic) first
        sorted_keys = sorted(self.feature_keys, key=lambda k: original_resolutions[k])

        # Step 1: Project all features to fpn_dim (keep original resolutions)
        projected = {}
        for key in sorted_keys:
            fmap = feature_maps[key]
            if fmap.dtype != torch.float32:
                fmap = fmap.float()
            projected[key] = self.proj_layers[key](fmap)

        # Get target resolution (highest resolution for final output)
        target_h = max(f.shape[-2] for f in projected.values())
        target_w = max(f.shape[-1] for f in projected.values())

        # Step 2: Top-Down Path (low-res → high-res, semantic enhancement)
        # P_low → upsample → concat with P_high → fuse
        td_features = {}
        td_features[sorted_keys[0]] = projected[sorted_keys[0]]

        for i, key in enumerate(sorted_keys[1:]):
            prev_key = sorted_keys[i]
            prev_feat = td_features[prev_key]
            curr_feat = projected[key]

            # Upsample previous (lower-res) to current resolution
            if prev_feat.shape[-2:] != curr_feat.shape[-2:]:
                prev_feat = F.interpolate(
                    prev_feat, size=curr_feat.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            # Concat and fuse
            fused = torch.cat([prev_feat, curr_feat], dim=1)
            td_features[key] = self.td_fuse_blocks[i](fused)

        # The old bottom-up blocks remain registered for checkpoint loading,
        # but their outputs never reach the highest-resolution final feature.
        # Skip their unused computation while preserving the exact final input.
        final_key = sorted_keys[-1]
        td_final = td_features[final_key]
        if td_final.shape[-2:] != (target_h, target_w):
            td_final = F.interpolate(td_final, size=(target_h, target_w), mode="bilinear", align_corners=False)

        output = self.final_fuse(torch.cat([td_final, td_final], dim=1))

        return output


class S2FPNGlobalToFineFusion(nn.Module):
    """Trainable three-level fusion with a live coarse-to-fine path.

    Module names mirror the Stage-II readout so its fusion weights can be
    transferred after a Stage-I run, subject to an explicit parity check.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        feature_keys: List[str],
        fpn_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        if not feature_keys or len(set(feature_keys)) != len(feature_keys):
            raise ValueError("feature_keys must be nonempty and unique")
        self.feature_keys = tuple(feature_keys)
        self.lateral = nn.ModuleDict({
            key: nn.Sequential(
                nn.Conv2d(feature_dims[key], fpn_dim, 1),
                _make_group_norm(fpn_dim),
                nn.GELU(),
            )
            for key in self.feature_keys
        })
        self.fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * fpn_dim, fpn_dim, 3, padding=1), _make_group_norm(fpn_dim), nn.GELU())
            for _ in self.feature_keys[1:]
        ])
        self.output_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1), _make_group_norm(fpn_dim), nn.GELU(),
        )
        self.to(device)

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        if set(feature_maps) != set(self.feature_keys):
            raise ValueError(f"Expected feature maps {self.feature_keys}, got {tuple(feature_maps)}")
        fused = None
        previous_area = 0
        for index, key in enumerate(self.feature_keys):
            feature = feature_maps[key]
            if feature.ndim != 4:
                raise ValueError(f"Feature map {key} must be [B, C, H, W]")
            area = feature.shape[-2] * feature.shape[-1]
            if area < previous_area:
                raise ValueError("feature_keys must be ordered from coarse to fine")
            previous_area = area
            projected = self.lateral[key](feature.to(dtype=self.lateral[key][0].weight.dtype))
            if fused is None:
                fused = projected
            else:
                fused = F.interpolate(fused, size=projected.shape[-2:], mode="bilinear", align_corners=False)
                fused = self.fusions[index - 1](torch.cat((fused, projected), dim=1))
        return self.output_fusion(fused)


class MultiLayerMapFusion(nn.Module):
    """Projection-based multi-layer fusion for controlled concat ablations."""

    def __init__(
        self,
        feature_dims: Dict[str, int],
        feature_keys: List[str],
        fpn_dim: int = 256,
        fusion_mode: str = "concat",
        device: str = "cuda",
    ):
        super().__init__()
        self.feature_keys = feature_keys
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode != "concat":
            raise ValueError(f"Unsupported CleanDIFT map fusion mode: {fusion_mode}. Expected concat.")

        self.proj_layers = nn.ModuleDict()
        for key in feature_keys:
            c_in = int(feature_dims[key])
            self.proj_layers[key] = nn.Sequential(
                nn.Conv2d(c_in, fpn_dim, kernel_size=1, bias=True),
                _make_group_norm(fpn_dim),
                nn.GELU(),
            ).to(device)

        self.concat_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(feature_keys), fpn_dim, kernel_size=1, bias=True),
            _make_group_norm(fpn_dim),
            nn.GELU(),
        ).to(device)

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_h = max(feature_maps[key].shape[-2] for key in self.feature_keys)
        target_w = max(feature_maps[key].shape[-1] for key in self.feature_keys)

        projected = []
        for key in self.feature_keys:
            fmap = feature_maps[key]
            if fmap.dtype != torch.float32:
                fmap = fmap.float()
            fmap = self.proj_layers[key](fmap)
            if fmap.shape[-2:] != (target_h, target_w):
                fmap = F.interpolate(fmap, size=(target_h, target_w), mode="bilinear", align_corners=False)
            projected.append(fmap)

        return self.concat_fuse(torch.cat(projected, dim=1))


class QueriesAttentionPooling(nn.Module):
    """
    Learnable Queries Attention Pooling with Layer Scale.

    Cross-attention from learnable queries to spatial tokens,
    producing fixed-size global representation.

    Args:
        dim: Feature dimension
        num_queries: Number of learnable query vectors
        num_heads: Number of attention heads
        dropout: Dropout rate
        layer_scale_init: Initial value for layer scale (default 0.1)
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_scale_init: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        # Learnable queries
        self.queries = nn.Parameter(torch.empty(num_queries, dim, device=device))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Normalization
        self.token_norm = nn.LayerNorm(dim).to(device)
        self.query_norm = nn.LayerNorm(dim).to(device)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        ).to(device)

        # FFN
        self.ffn_norm = nn.LayerNorm(dim).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        ).to(device)

        # Layer Scale (DeiT-III style)
        self.attn_layer_scale = nn.Parameter(torch.ones(dim, device=device) * layer_scale_init)
        self.ffn_layer_scale = nn.Parameter(torch.ones(dim, device=device) * layer_scale_init)

        # Position encoding dropout
        self.pos_dropout = nn.Dropout(dropout)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] fused feature map

        Returns:
            queries: [B, num_queries, dim] attended query vectors
        """
        B, C, H, W = feature_map.shape

        # Flatten to tokens
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Add positional encoding
        pe = _build_2d_sincos_pos_embed(H, W, C, device=tokens.device, dtype=tokens.dtype)
        tokens = self.pos_dropout(tokens + pe)

        # Normalize tokens
        tokens_norm = self.token_norm(tokens)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Pre-Norm + Layer Scale Attention
        queries_norm = self.query_norm(queries)
        attn_out, _ = self.attn(
            query=queries_norm,
            key=tokens_norm,
            value=tokens_norm,
            need_weights=False
        )
        queries = queries + self.attn_layer_scale * attn_out

        # Pre-Norm + Layer Scale FFN
        ffn_out = self.ffn(self.ffn_norm(queries))
        queries = queries + self.ffn_layer_scale * ffn_out

        return queries


# =============================================================================
# Main Encoder Class
# =============================================================================

class CleanDIFTImgEncoder(nn.Module):
    """SD2.1 Robot-DIFT Student with training-only Teacher alignment.

    The Student maps images in [-1, 1] to raw decoder maps such as
    ``us3/us6/us8``. When the backbone is trainable, the same module also owns
    the frozen SD2.1 Teacher and the timestep-conditioned alignment adapters.

    ``readout`` selects what sits on top of the Student:

    * ``"none"``: backbone only. The caller owns the readout; the paper Stage-I
      encoder (:mod:`agents.encoders.robot_dift_stage1_encoder`) uses this.
    * ``"legacy"``: the built-in S2-FPN + learnable-query head
      (``--stage1_readout legacy`` and the raw-dense Stage-II control).

    Args:
        sd_version: "sd15" or "sd21"
        feature_key: Single key or list of keys returned to the readout
        freeze_backbone: Whether to freeze the Student
        map_out_dim: Output latent dimension of the legacy readout
        fpn_dim: Legacy FPN intermediate dimension (default 256)
        fpn_num_queries: Number of legacy learnable queries
        fpn_dropout: Legacy readout dropout rate
    """

    def __init__(
        self,
        sd_version: str = "sd21",
        feature_key: str = "us6",
        freeze_backbone: bool = True,
        freeze_head: bool = False,
        device: str = "cuda",
        camera_names: Optional[List[str]] = None,
        use_text_condition: bool = False,
        yaml_file: Optional[str] = None,
        map_out_dim: Optional[int] = None,
        use_fp32: Optional[bool] = None,
        custom_checkpoint: Optional[str] = None,
        load_full_encoder_checkpoint: bool = False,
        strict_full_encoder_checkpoint: bool = False,
        alignment_cfg: Optional[Any] = None,
        num_t_stratification_bins: Optional[int] = None,
        t_min: Optional[int] = None,
        t_max: Optional[int] = None,
        vae_latent_mode: str = "sample",
        # Legacy S2-FPN readout parameters
        fpn_dim: int = 256,
        fpn_num_queries: int = 4,
        fpn_dropout: float = 0.1,
        layer_scale_init: float = 0.1,
        fusion_mode: str = "s2fpn",
        output_mode: str = "pooled",
        apply_feature_adapter: bool = False,
        feature_adapter_timestep: Optional[float] = None,
        alignment_apply_feature_adapter: Optional[bool] = None,
        alignment_feature_keys: Optional[List[str]] = None,
        alignment_layer_reduction: str = "mean",
        student_init: str = "cleandift",
        readout: str = "legacy",
    ):
        super().__init__()
        self._logger = logging.getLogger(__name__)

        # =================================================================
        # 1. Configuration
        # =================================================================
        self.sd_version = sd_version
        self.freeze_backbone = freeze_backbone
        self.freeze_head = bool(freeze_head)
        self.device = device
        self.camera_names = camera_names
        self.use_text_condition = use_text_condition
        self.map_out_dim = map_out_dim
        self.custom_checkpoint = custom_checkpoint
        self.load_full_encoder_checkpoint = bool(load_full_encoder_checkpoint)
        self.strict_full_encoder_checkpoint = bool(strict_full_encoder_checkpoint)
        self._fpn_dim = fpn_dim
        self._fpn_num_queries = fpn_num_queries
        self._fpn_dropout = fpn_dropout
        self.fusion_mode = str(fusion_mode).lower()
        self.output_mode = str(output_mode).lower()
        self.readout = str(readout).strip().lower()
        if self.readout not in {"legacy", "none"}:
            raise ValueError(f"Unsupported CleanDIFT readout={readout!r}. Expected 'legacy' or 'none'.")
        self.apply_feature_adapter = bool(apply_feature_adapter)
        self.alignment_apply_feature_adapter = (
            self.apply_feature_adapter
            if alignment_apply_feature_adapter is None
            else bool(alignment_apply_feature_adapter)
        )
        self.feature_adapter_timestep = feature_adapter_timestep
        self.student_init = str(student_init).strip().lower()
        if self.student_init not in {"cleandift", "sd_teacher"}:
            raise ValueError(
                f"Unsupported student_init={student_init!r}. Expected 'cleandift' or 'sd_teacher'."
            )
        if self.student_init == "sd_teacher" and custom_checkpoint:
            raise ValueError(
                "student_init='sd_teacher' cannot be combined with custom_checkpoint; "
                "the checkpoint would overwrite the copied SD2.1 Teacher weights."
            )
        self.alignment_layer_reduction = str(alignment_layer_reduction).strip().lower()
        if self.alignment_layer_reduction not in {"mean", "sum"}:
            raise ValueError("alignment_layer_reduction must be 'mean' or 'sum'")
        if vae_latent_mode not in {"sample", "mode"}:
            raise ValueError("vae_latent_mode must be 'sample' or 'mode'")
        self.vae_latent_mode = vae_latent_mode
        if self.output_mode not in {"pooled", "queries"}:
            raise ValueError(
                f"Unsupported CleanDIFT output_mode={output_mode}. "
                "Expected one of: pooled, queries."
            )
        self._pending_full_encoder_state = None

        # Parse feature keys
        if isinstance(feature_key, ListConfig):
            feature_key = list(feature_key)
        if isinstance(feature_key, (list, tuple)):
            if len(feature_key) == 0:
                raise ValueError("feature_key must contain at least one entry.")
            self.feature_keys = [str(k) for k in feature_key]
        else:
            self.feature_keys = [str(feature_key)]

        self.multi_feature_mode = len(self.feature_keys) > 1
        self.primary_feature_key = self.feature_keys[0]
        if alignment_feature_keys is None:
            self.alignment_feature_keys = list(self.feature_keys)
        else:
            if isinstance(alignment_feature_keys, ListConfig):
                alignment_feature_keys = list(alignment_feature_keys)
            if isinstance(alignment_feature_keys, str):
                alignment_feature_keys = [alignment_feature_keys]
            if not isinstance(alignment_feature_keys, (list, tuple)) or not alignment_feature_keys:
                raise ValueError("alignment_feature_keys must be a non-empty list of feature keys.")
            self.alignment_feature_keys = [str(key) for key in alignment_feature_keys]
            if len(set(self.alignment_feature_keys)) != len(self.alignment_feature_keys):
                raise ValueError("alignment_feature_keys must not contain duplicates.")
        if self.apply_feature_adapter and not set(self.feature_keys).issubset(self.alignment_feature_keys):
            raise ValueError(
                "Policy feature adapters require every readout feature_key to appear in "
                "alignment_feature_keys."
            )
        if self.readout == "legacy" and self.multi_feature_mode and self.fusion_mode not in {
            "s2fpn", "global_to_fine", "concat"
        }:
            raise ValueError(
                f"Unsupported CleanDIFT fusion_mode={fusion_mode}. "
                "Expected one of: s2fpn, global_to_fine, concat."
            )

        # Precision settings
        self._force_backbone_fp32 = bool(use_fp32) if use_fp32 is not None else False

        # Alignment config
        if alignment_cfg is not None and not isinstance(alignment_cfg, dict):
            alignment_cfg = OmegaConf.to_container(alignment_cfg, resolve=True)
        self.alignment_cfg = alignment_cfg or {}

        # =================================================================
        # 2. Load CleanDIFT Backbone
        # =================================================================
        self._init_backbone(
            sd_version,
            yaml_file,
            device,
            custom_checkpoint,
            num_t_stratification_bins=num_t_stratification_bins,
            t_min=t_min,
            t_max=t_max,
        )

        # =================================================================
        # 3. Legacy readout (skipped for backbone-only use)
        # =================================================================
        if self.readout == "legacy":
            if self.multi_feature_mode:
                target_dim = map_out_dim if map_out_dim and map_out_dim > 0 else fpn_dim
                self._build_multi_feature_head(
                    target_dim,
                    fpn_dim,
                    fpn_num_queries,
                    fpn_dropout,
                    layer_scale_init,
                    device,
                )
            else:
                target_dim = map_out_dim if map_out_dim and map_out_dim > 0 else self.feature_dims[self.primary_feature_key]
                self._build_single_feature_head(target_dim, device)
            self._multi_feature_out_dim = target_dim
        else:
            self._multi_feature_out_dim = None
        self._load_pending_full_encoder_state()

        # =================================================================
        # 4. Freeze Backbone if needed
        # =================================================================
        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        if self.freeze_head:
            self._set_head_trainable(False)
        self._keep_frozen_reference_eval()

        self._log_init_summary()

    @property
    def model_dtype(self) -> torch.dtype:
        """Current Student parameter dtype.

        Robomimic casts the whole policy to float32 after construction, so
        inputs follow the live parameters rather than the dtype chosen in
        ``_init_backbone``, and images are not quantized before the VAE.
        """
        try:
            return next(self.model.parameters()).dtype
        except (AttributeError, StopIteration):
            return torch.float32

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and self.model is not None:
            self.model.eval()
        if self.freeze_head:
            self._set_head_trainable(False)
        self._keep_frozen_reference_eval()
        return self

    def _keep_frozen_reference_eval(self) -> None:
        """Keep fixed diffusion references deterministic after parent .train() calls."""
        if self.model is None:
            return
        for name in ("unet_feature_extractor_base", "ae"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                module.eval()
        pipe = getattr(self.model, "pipe", None)
        text_encoder = getattr(pipe, "text_encoder", None) if pipe is not None else None
        if isinstance(text_encoder, nn.Module):
            text_encoder.eval()
            text_encoder.requires_grad_(False)

    def _init_backbone(
        self,
        sd_version: str,
        yaml_file: Optional[str],
        device: str,
        custom_checkpoint: Optional[str],
        num_t_stratification_bins: Optional[int] = None,
        t_min: Optional[int] = None,
        t_max: Optional[int] = None,
    ):
        """Initialize CleanDIFT backbone."""
        # Resolve config file
        if sd_version == "sd15":
            default_yaml = "sd15_feature_extractor.yaml"
            checkpoint_name = "cleandift_sd15_full.safetensors"
        else:
            default_yaml = "sd21_feature_extractor.yaml"
            checkpoint_name = "cleandift_sd21_full.safetensors"

        here = os.path.dirname(os.path.abspath(__file__))
        yaml_name = yaml_file or default_yaml
        candidates = [
            os.path.join(here, "cleandift", "configs", yaml_name),
            os.path.join(here, yaml_name),
        ]
        yaml_path = None
        for c in candidates:
            if os.path.exists(c):
                yaml_path = c
                break
        if yaml_path is None:
            raise FileNotFoundError(f"Could not find YAML: {yaml_name}. Tried: {candidates}")

        cfg = OmegaConf.load(yaml_path)
        mconf = cfg["model"]

        # Feature dimensions
        self.feature_dims = dict(mconf.get("feature_dims", {}))
        for key in self.feature_keys:
            if key not in self.feature_dims:
                raise KeyError(f"Feature key '{key}' not in config. Available: {list(self.feature_dims.keys())}")
        for key in self.alignment_feature_keys:
            if key not in self.feature_dims:
                raise KeyError(
                    f"Alignment feature key '{key}' not in config. Available: {list(self.feature_dims.keys())}"
                )

        # Build model
        ae = AutoencoderKL(
            repo=mconf["ae"]["repo"], latent_mode=self.vae_latent_mode
        ).to(device)
        mapping = MappingSpec(
            depth=mconf["mapping"]["depth"],
            width=mconf["mapping"]["width"],
            d_ff=mconf["mapping"]["d_ff"],
            dropout=mconf["mapping"]["dropout"],
        )

        def fix_class_path(path: str) -> str:
            if path.startswith("src."):
                try_path = "agents.encoders.cleandift." + path
                return try_path if locate(try_path) is not None else ("cleandift." + path)
            return path

        selected_feature_dims = {key: self.feature_dims[key] for key in self.alignment_feature_keys}
        feature_extractor_cls_path = fix_class_path(mconf["feature_extractor_cls"])
        self._feature_extractor_cls_path = feature_extractor_cls_path
        self.model = StableFeatureAligner(
            sd_version=sd_version,
            t_min=mconf.get("t_min", 1) if t_min is None else int(t_min),
            t_max=mconf.get("t_max", 999) if t_max is None else int(t_max),
            num_t_stratification_bins=(
                mconf.get("num_t_stratification_bins", 3)
                if num_t_stratification_bins is None
                else int(num_t_stratification_bins)
            ),
            train_unet=mconf.get("train_unet", True),
            learn_timestep=mconf.get("learn_timestep", True),
            use_text_condition=self.use_text_condition,
            ae=ae,
            mapping=mapping,
            adapter_layer_class=fix_class_path(mconf["adapter_layer_class"]),
            adapter_layer_params=mconf.get("adapter_layer_params", {}),
            feature_extractor_cls=feature_extractor_cls_path,
            feature_dims=selected_feature_dims,
            device=device,
        )

        # Precision. ``backbone_dtype`` is the autocast compute dtype. A frozen
        # Student may also store its weights in it; a trainable Student keeps
        # float32 master weights so the SD2.1 copy and the learned timestep
        # (261 is not representable in bfloat16) stay exact.
        if self._force_backbone_fp32 or not (device.startswith("cuda") and torch.cuda.is_available()):
            backbone_dtype = torch.float32
        else:
            dtype_override = (
                os.environ.get("ROBOT_DIFT_CLEANDIFT_BACKBONE_DTYPE")
                or os.environ.get("ROBOT_DIFT_DROID_AMP_DTYPE")
                or ""
            ).lower()
            if dtype_override in {"float16", "fp16", "half"}:
                backbone_dtype = torch.float16
            elif dtype_override in {"float32", "fp32", "full"}:
                backbone_dtype = torch.float32
            else:
                backbone_dtype = torch.bfloat16
        parameter_dtype = backbone_dtype if self.freeze_backbone else torch.float32
        self.model = self.model.to(device).to(dtype=parameter_dtype)

        self._backbone_dtype = backbone_dtype
        self._amp_enabled = backbone_dtype in {torch.float16, torch.bfloat16}
        self._amp_autocast_kwargs = {"device_type": "cuda", "dtype": backbone_dtype} if self._amp_enabled else {}

        # Load weights
        if custom_checkpoint:
            self._load_custom_checkpoint(custom_checkpoint)
            self._checkpoint_source = os.path.abspath(os.path.expanduser(str(custom_checkpoint)))
        elif self.student_init == "cleandift":
            ckpt_pth = hf_hub_download(repo_id="CompVis/cleandift", filename=checkpoint_name)
            state_dict = load_file(ckpt_pth)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                raise RuntimeError(f"Missing CleanDIFT checkpoint keys for selected layers: {missing[:8]}")
            if unexpected:
                warnings.warn(f"Ignoring unused CleanDIFT checkpoint keys for unselected layers: {unexpected[:8]}...")
            self._checkpoint_source = f"CompVis/cleandift:{os.path.basename(ckpt_pth)}"
        else:
            # StableFeatureAligner has already copied the frozen SD U-Net into
            # its clean Student. Keep that initialization for paper-protocol runs.
            self._checkpoint_source = f"{self.model.repo}:sd_teacher_weight_copy"

    @staticmethod
    def _count_params(parameters) -> Tuple[int, int]:
        params = list(parameters)
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        return total, trainable

    def _head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = []
        if self.readout != "legacy":
            return modules

        if self.multi_feature_mode:
            for name in ("feature_fusion", "queries_pooling", "final_proj", "query_proj"):
                module = getattr(self, name, None)
                if isinstance(module, nn.Module):
                    modules.append(module)
        else:
            for name in ("single_proj", "single_norm"):
                module = getattr(self, name, None)
                if isinstance(module, nn.Module):
                    modules.append(module)

        return modules

    def _set_head_trainable(self, trainable: bool) -> None:
        for module in self._head_modules():
            module.train(trainable)
            module.requires_grad_(trainable)

    def _head_parameters(self) -> List[nn.Parameter]:
        head_params: List[nn.Parameter] = []
        for module in self._head_modules():
            head_params.extend(module.parameters())

        return head_params

    def _log_init_summary(self) -> None:
        model_total, model_trainable = self._count_params(self.model.parameters())
        ae_total, ae_trainable = (0, 0)
        if hasattr(self.model, "ae"):
            ae_total, ae_trainable = self._count_params(self.model.ae.parameters())
        head_total, head_trainable = self._count_params(self._head_parameters())
        source = getattr(self, "_checkpoint_source", "unknown")

        self._logger.info(
            "CleanDIFTImgEncoder initialized: checkpoint=%s, readout=%s, freeze_backbone=%s, "
            "freeze_head=%s, feature_keys=%s, alignment_feature_keys=%s, student_init=%s, "
            "vae_latent_mode=%s, fusion=%s, output_mode=%s, use_text_condition=%s, "
            "policy_apply_adapter=%s, alignment_apply_adapter=%s, "
            "backbone_trainable=%d/%d, ae_trainable=%d/%d, head_trainable=%d/%d",
            source,
            self.readout,
            self.freeze_backbone,
            self.freeze_head,
            self.feature_keys,
            self.alignment_feature_keys,
            self.student_init,
            self.vae_latent_mode,
            self.fusion_mode,
            self.output_mode,
            self.use_text_condition,
            self.apply_feature_adapter,
            self.alignment_apply_feature_adapter,
            model_trainable,
            model_total,
            ae_trainable,
            ae_total,
            head_trainable,
            head_total,
        )

        if not self.freeze_backbone and not self.custom_checkpoint and self.student_init == "cleandift":
            self._logger.warning(
                "CleanDIFT backbone is trainable but custom_checkpoint is not set. "
                "This run starts from the public CleanDIFT checkpoint, not a Robot-DIFT/DROID-finetuned checkpoint."
            )

    def _build_multi_feature_head(
        self,
        target_dim: int,
        fpn_dim: int,
        num_queries: int,
        dropout: float,
        layer_scale_init: float,
        device: str,
    ):
        """Build multi-layer fusion and attention pooling modules."""
        if self.fusion_mode == "s2fpn":
            self.feature_fusion = S2FPNBidirectionalFusion(
                feature_dims=self.feature_dims,
                feature_keys=self.feature_keys,
                fpn_dim=fpn_dim,
                device=device,
            )
        elif self.fusion_mode == "global_to_fine":
            self.feature_fusion = S2FPNGlobalToFineFusion(
                self.feature_dims, self.feature_keys, fpn_dim=fpn_dim, device=device,
            )
        elif self.fusion_mode == "concat":
            self.feature_fusion = MultiLayerMapFusion(
                feature_dims=self.feature_dims,
                feature_keys=self.feature_keys,
                fpn_dim=fpn_dim,
                fusion_mode="concat",
                device=device,
            )
        else:
            raise ValueError(f"Unsupported CleanDIFT fusion_mode={self.fusion_mode}.")

        # Queries Attention Pooling
        num_heads = max(1, min(8, fpn_dim // 64))
        while num_heads > 1 and fpn_dim % num_heads != 0:
            num_heads -= 1

        self.queries_pooling = QueriesAttentionPooling(
            dim=fpn_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            device=device,
        )

        # Final projection: [num_queries * fpn_dim] → [target_dim]
        flat_dim = fpn_dim * num_queries
        self.final_proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, max(target_dim, flat_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(target_dim, flat_dim // 2), target_dim),
        ).to(device)

        if self.output_mode == "queries":
            self.query_proj = nn.Sequential(
                nn.LayerNorm(fpn_dim),
                nn.Linear(fpn_dim, target_dim),
            ).to(device)

    @property
    def s2fpn_fusion(self):
        """Backward-compatible alias without duplicate module registration."""
        return getattr(self, "feature_fusion", None)

    def _build_single_feature_head(self, target_dim: int, device: str):
        """Build simple head for single feature mode."""
        c_in = self.feature_dims[self.primary_feature_key]

        self.single_proj = nn.Conv2d(c_in, target_dim, kernel_size=1, bias=True).to(device)
        self.single_norm = nn.LayerNorm(target_dim).to(device)
        self.single_pool = nn.AdaptiveAvgPool2d(1)

        nn.init.normal_(self.single_proj.weight, std=0.02)
        nn.init.zeros_(self.single_proj.bias)

    # =========================================================================
    # Forward Methods
    # =========================================================================

    @staticmethod
    def _normalize_captions(caption, batch_size: int) -> List[str]:
        """Return one text prompt per image, preserving temporal flattening order."""
        if caption is None:
            return [""] * batch_size
        if isinstance(caption, str):
            return [caption] * batch_size
        if isinstance(caption, ListConfig):
            caption = list(caption)
        if isinstance(caption, tuple):
            caption = list(caption)
        if not isinstance(caption, list):
            raise TypeError(f"CleanDIFT captions must be str/list/tuple, got {type(caption)}")
        captions = [str(item) for item in caption]
        if len(captions) == batch_size:
            return captions
        if len(captions) == 1:
            return captions * batch_size
        if len(captions) > 0 and batch_size % len(captions) == 0:
            repeat = batch_size // len(captions)
            return [item for item in captions for _ in range(repeat)]
        raise ValueError(
            f"Cannot align {len(captions)} CleanDIFT captions with image batch {batch_size}."
        )

    def _amp_context(self):
        return torch.autocast(**self._amp_autocast_kwargs) if self._amp_enabled else contextlib.nullcontext()

    def forward(self, x: torch.Tensor, lang_cond=None, alignment_context=None) -> tuple:
        """Legacy readout forward.

        Args:
            x: Input images [B, 3, H, W] in [-1, 1]
            lang_cond: Optional language condition
            alignment_context: Optional alignment loss context

        Returns:
            features: [B, 1, latent_dim] global features (or query tokens)
            alignment_loss: Optional alignment loss tensor
        """
        if self.readout != "legacy":
            raise RuntimeError("CleanDIFTImgEncoder(readout='none') has no readout; use encode_student().")
        x = x.to(self.device, dtype=self.model_dtype).contiguous()
        caption = self._normalize_captions(lang_cond, x.shape[0]) if self.use_text_condition else None

        features = self._extract_features(x, caption)

        # Compute alignment loss if needed
        alignment_loss = None
        if alignment_context is not None and not self.freeze_backbone:
            alignment_loss = self._compute_alignment_loss(alignment_context, caption)

        return features, alignment_loss

    def _extract_features(self, x: torch.Tensor, caption=None) -> torch.Tensor:
        """Extract and process features through the legacy readout pipeline."""
        feature_maps = self._encode_backbone(x, caption)

        if self.multi_feature_mode:
            fused = self.feature_fusion(feature_maps)

            # Queries attention pooling
            queries = self.queries_pooling(fused)  # [B, num_queries, fpn_dim]

            if self.output_mode == "queries":
                return self.query_proj(queries)  # [B, num_queries, target_dim]

            # Final projection
            pooled = queries.flatten(1)  # [B, num_queries * fpn_dim]
            output = self.final_proj(pooled)  # [B, target_dim]

            return output.unsqueeze(1)  # [B, 1, target_dim]
        else:
            # Single feature mode
            fmap = feature_maps[self.primary_feature_key]
            if fmap.dtype != torch.float32:
                fmap = fmap.float()

            proj = self.single_proj(fmap)  # [B, target_dim, H, W]
            pooled = self.single_pool(proj).flatten(1)  # [B, target_dim]
            output = self.single_norm(pooled)

            return output.unsqueeze(1)  # [B, 1, target_dim]

    def _encode_backbone(self, x: torch.Tensor, caption=None) -> Dict[str, torch.Tensor]:
        """Extract readout feature maps from the Student."""
        if x.dtype != self.model_dtype:
            x = x.to(self.model_dtype)
        x = x.contiguous()
        if self.use_text_condition:
            caption = self._normalize_captions(caption, x.shape[0])

        grad_ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        adapter_timestep = self.feature_adapter_timestep if self.apply_feature_adapter else None

        with grad_ctx:
            with self._amp_context():
                if self.multi_feature_mode:
                    features = self.model.get_features(
                        x, caption=caption, t=adapter_timestep,
                        feat_key=self.feature_keys,
                        use_base_model=False,
                        apply_adapter=self.apply_feature_adapter,
                    )
                else:
                    features = self.model.get_features(
                        x, caption=caption, t=adapter_timestep,
                        feat_key=self.primary_feature_key,
                        use_base_model=False,
                        apply_adapter=self.apply_feature_adapter,
                    )
                    if not isinstance(features, dict):
                        features = {self.primary_feature_key: features}

        # Convert to float32
        return {k: v.float() if v.dtype != torch.float32 else v for k, v in features.items()}

    def encode_student(
        self,
        x: torch.Tensor,
        captions=None,
        feature_keys: Optional[List[str]] = None,
        alignment: bool = False,
        return_raw_cosine: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """One Student pass for raw ``feature_keys`` and, optionally, Teacher alignment.

        With ``alignment=True`` the pass computes every aligned map and the same
        activations feed both the returned maps and the alignment loss, instead
        of running the Student a second time on the same images.

        Returns ``(maps, alignment_loss, metrics)``. Maps are float32; the loss
        is ``None`` when ``alignment`` is False. ``metrics`` holds detached
        per-map alignment terms and, on request, the raw Student/Teacher cosine.
        """
        keys = list(self.feature_keys if feature_keys is None else feature_keys)
        if alignment and self.freeze_backbone:
            raise RuntimeError("Teacher alignment requires a trainable Student")
        x = x.to(self.device, dtype=self.model_dtype).contiguous()
        captions = self._normalize_captions(captions if self.use_text_condition else None, x.shape[0])

        grad_ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        loss = None
        metrics: Dict[str, torch.Tensor] = {}
        with grad_ctx, self._amp_context():
            unet_conds = self.model._get_unet_conds(captions, x.device, x.dtype, 1)
            x_0 = self.model.encode_latents(x)
            requested_keys = list(dict.fromkeys([*keys, *self.alignment_feature_keys])) if alignment else keys
            student = self.model.student_maps(x_0, unet_conds, requested_keys)
            if alignment:
                terms, metrics = self.model.alignment_terms(
                    x_0,
                    unet_conds,
                    {key: student[key] for key in self.alignment_feature_keys},
                    apply_adapter=self.alignment_apply_feature_adapter,
                    return_raw_cosine=return_raw_cosine,
                )
                loss = reduce_alignment_loss_terms(terms, self.alignment_layer_reduction).float()
                metrics.update({name: value.detach().float() for name, value in terms.items()})
        maps = {key: student[key].float() for key in keys}
        return maps, loss, metrics

    def _compute_alignment_loss(self, alignment_context: dict, caption) -> Optional[torch.Tensor]:
        """Compute optional alignment loss."""
        align_images = alignment_context.get("images")
        if align_images is None or align_images.shape[0] == 0:
            return None
        align_captions = alignment_context.get("captions")
        if align_captions is None:
            align_captions = caption
        return self.compute_alignment_loss(align_images, align_captions)

    def compute_alignment_loss(
        self,
        x: torch.Tensor,
        caption: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Teacher alignment loss for images in [-1, 1] (separate Student pass).

        Args:
            x: Input images [B, 3, H, W]
            caption: Optional list of captions

        Returns:
            Alignment loss tensor (scalar)
        """
        x = x.to(self.device, dtype=self.model_dtype).contiguous()

        # If backbone is frozen, no alignment loss
        if self.freeze_backbone:
            return torch.tensor(0.0, device=x.device)

        caption = self._normalize_captions(caption, x.shape[0])
        with self._amp_context():
            loss_dict = self.model.forward(
                x,
                caption,
                apply_adapter=self.alignment_apply_feature_adapter,
            )

        loss = reduce_alignment_loss_terms(loss_dict, self.alignment_layer_reduction)
        return loss.float() if loss.dtype != torch.float32 else loss

    def extract_feature_map(self, x: torch.Tensor, lang_cond=None, **_unused) -> torch.Tensor:
        """Extract a dense feature map (for visualization/analysis)."""
        caption = self._normalize_captions(lang_cond, x.shape[0]) if self.use_text_condition else None
        features = self._encode_backbone(x.to(self.device, dtype=self.model_dtype), caption)

        if self.readout == "legacy" and self.multi_feature_mode:
            return self.feature_fusion(features)
        return features[self.primary_feature_key]

    def get_parameter_groups(
        self,
        base_lr: float,
        adapter_lr_multiplier: float = 1.0,
        backbone_lr_multiplier: float = 0.1,
        head_lr_multiplier: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        student_lr_multiplier: Optional[float] = None,
        **kwargs,
    ) -> List[dict]:
        """
        Get parameter groups with different learning rates.

        ``backbone`` holds the Student, alignment adapters, and learned Student
        timestep; ``head`` holds the legacy readout (empty for ``readout='none'``).
        """
        del kwargs
        head_lr_multiplier = adapter_lr_multiplier if head_lr_multiplier is None else head_lr_multiplier
        groups = []

        # Backbone parameters
        if not self.freeze_backbone and self.model is not None:
            backbone_params = [p for p in self.model.parameters() if p.requires_grad]
            if student_lr_multiplier is not None:
                if not math.isfinite(student_lr_multiplier) or student_lr_multiplier <= 0:
                    raise ValueError("student_lr_multiplier must be finite and positive")
                student_params = list(self.model.unet_feature_extractor_cleandift.parameters())
                if isinstance(getattr(self.model, "timestep", None), nn.Parameter):
                    student_params.append(self.model.timestep)
                student_params = [p for p in student_params if p.requires_grad]
                student_ids = {id(p) for p in student_params}
                backbone_params = [p for p in backbone_params if id(p) not in student_ids]
                if student_params:
                    groups.append({
                        "params": student_params,
                        "lr": base_lr * student_lr_multiplier,
                        "name": "student",
                        **({"weight_decay": backbone_weight_decay} if backbone_weight_decay is not None else {}),
                    })
            if backbone_params:
                groups.append({
                    "params": backbone_params,
                    "lr": base_lr * backbone_lr_multiplier,
                    "name": "alignment" if student_lr_multiplier is not None else "backbone",
                    **({"weight_decay": backbone_weight_decay} if backbone_weight_decay is not None else {}),
                })
            elif student_lr_multiplier is None:
                self._logger.warning(
                    "CleanDIFT backbone is marked trainable, but no trainable backbone parameters were found."
                )

        # Head parameters
        head_params = [p for p in self._head_parameters() if p.requires_grad]

        if head_params:
            groups.append({
                "params": head_params,
                "lr": base_lr * head_lr_multiplier,
                "name": "head",
                **({"weight_decay": head_weight_decay} if head_weight_decay is not None else {}),
            })

        return groups

    # =========================================================================
    # Checkpoint loading
    # =========================================================================

    def _load_custom_checkpoint(self, checkpoint_dir: str):
        """Load a Robot-DIFT Student checkpoint; missing Student weights raise."""
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        if os.path.isdir(checkpoint_dir):
            has_direct = (
                os.path.exists(os.path.join(checkpoint_dir, "robot_dift_encoder_state.pt"))
                or os.path.exists(os.path.join(checkpoint_dir, "cleandift_full_state.safetensors"))
                or os.path.exists(os.path.join(checkpoint_dir, "cleandift_full_state.pt"))
                or _has_dift_component_checkpoint(checkpoint_dir)
            )
            if not has_direct:
                checkpoint_dirs = sorted(
                    [
                        os.path.join(checkpoint_dir, name)
                        for name in os.listdir(checkpoint_dir)
                        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, name))
                    ],
                    key=_checkpoint_sort_key,
                )
                for candidate in reversed(checkpoint_dirs):
                    if (
                        os.path.exists(os.path.join(candidate, "robot_dift_encoder_state.pt"))
                        or os.path.exists(os.path.join(candidate, "cleandift_full_state.safetensors"))
                        or os.path.exists(os.path.join(candidate, "cleandift_full_state.pt"))
                        or _has_dift_component_checkpoint(candidate)
                    ):
                        checkpoint_dir = candidate
                        break

        full_encoder_path = os.path.join(checkpoint_dir, "robot_dift_encoder_state.pt")
        safetensor_path = os.path.join(checkpoint_dir, "cleandift_full_state.safetensors")
        pt_path = os.path.join(checkpoint_dir, "cleandift_full_state.pt")

        # DIFT-format checkpoints contain only the StableFeatureAligner backbone.
        # Prefer them by default so downstream tasks learn their own S2-FPN/readout.
        if self.load_full_encoder_checkpoint and os.path.exists(full_encoder_path):
            payload = torch.load(full_encoder_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            self._pending_full_encoder_state = self._clean_encoder_state_keys(state_dict)
            self._logger.info(
                "Loaded pending Robot-DIFT full encoder checkpoint from %s: tensors=%d",
                full_encoder_path,
                len(self._pending_full_encoder_state),
            )
            return

        if self.load_full_encoder_checkpoint and self.strict_full_encoder_checkpoint:
            raise FileNotFoundError(
                "load_full_encoder_checkpoint=True requires robot_dift_encoder_state.pt, "
                f"but it was not found under {checkpoint_dir}. Use cleandift_droid_ft_encoder "
                "for DIFT-format backbone-only checkpoints, or save the DROID run with "
                "--save_robot_dift_full_encoder."
            )

        state_dict = None
        if os.path.exists(safetensor_path):
            state_dict = load_file(safetensor_path)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)

        if state_dict is not None:
            # Handle legacy prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = OrderedDict(
                    (k.replace("model.", "", 1) if k.startswith("model.") else k, v)
                    for k, v in state_dict.items()
                )

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            missing_student = [key for key in missing if key.startswith("unet_feature_extractor_cleandift.")]
            if missing_student:
                raise RuntimeError(
                    f"Robot-DIFT checkpoint {checkpoint_dir} lacks Student U-Net tensors: {missing_student[:5]}"
                )
            self._logger.info(
                "Loaded CleanDIFT custom backbone checkpoint from %s: tensors=%d, missing=%d, unexpected=%d",
                checkpoint_dir,
                len(state_dict),
                len(missing),
                len(unexpected),
            )
            if missing:
                warnings.warn(f"Missing keys: {missing[:5]}...")
            if unexpected:
                warnings.warn(f"Unexpected keys: {unexpected[:5]}...")
        elif _has_dift_component_checkpoint(checkpoint_dir):
            self._load_dift_components(checkpoint_dir)
        elif os.path.exists(full_encoder_path):
            warnings.warn(
                "Only robot_dift_encoder_state.pt was found and no DIFT component checkpoint "
                "is available. Loading the full image encoder, including S2-FPN/readout. "
                "Prefer load_full_encoder_checkpoint=True for explicit full-encoder transfer."
            )
            payload = torch.load(full_encoder_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            self._pending_full_encoder_state = self._clean_encoder_state_keys(state_dict)
        else:
            raise FileNotFoundError(f"No Robot-DIFT Student weights found under {checkpoint_dir}")

    def _load_dift_components(self, checkpoint_dir: str) -> None:
        """Load a lightweight DIFT-format checkpoint saved as separate components."""
        unet_path = _student_unet_weight_file(checkpoint_dir)
        if unet_path is None:
            raise FileNotFoundError(f"Missing CleanDIFT UNet component under {checkpoint_dir}/unet")
        if unet_path.endswith(".safetensors"):
            unet_state = load_file(unet_path)
        else:
            unet_state = torch.load(unet_path, map_location="cpu", weights_only=False)
        # The Student is the released artifact: a partial load must fail.
        self.model.unet_feature_extractor_cleandift.load_state_dict(unet_state, strict=True)

        component_specs = [
            ("adapters", "adapters.bin"),
            ("mapping", "mapping_network.bin"),
            ("time_emb", "time_emb.bin"),
            ("time_in_proj", "time_in_proj.bin"),
        ]
        loaded = {"unet": len(unet_state)}
        for attr_name, filename in component_specs:
            module = getattr(self.model, attr_name, None)
            path = os.path.join(checkpoint_dir, filename)
            if module is None or not os.path.exists(path):
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            missing, unexpected = module.load_state_dict(state, strict=False)
            loaded[attr_name] = len(state)
            if missing or unexpected:
                warnings.warn(
                    f"CleanDIFT {attr_name} component load: missing={missing[:5]}, unexpected={unexpected[:5]}"
                )

        timestep_path = os.path.join(checkpoint_dir, "timestep.bin")
        if not os.path.exists(timestep_path):
            raise FileNotFoundError(f"Missing learned Student timestep: {timestep_path}")
        timestep_state = torch.load(timestep_path, map_location="cpu", weights_only=False)
        timestep = timestep_state.get("timestep", timestep_state)
        with torch.no_grad():
            self.model.timestep.copy_(timestep.to(device=self.model.timestep.device, dtype=self.model.timestep.dtype))
        loaded["timestep"] = 1

        self._logger.info(
            "Loaded lightweight CleanDIFT component checkpoint from %s: %s",
            checkpoint_dir,
            loaded,
        )

    def _clean_encoder_state_keys(self, state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
        """Normalize common wrapper prefixes for CleanDIFTImgEncoder checkpoints."""
        cleaned = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        return cleaned

    def _load_pending_full_encoder_state(self):
        if self._pending_full_encoder_state is None:
            return
        missing, unexpected = self.load_state_dict(self._pending_full_encoder_state, strict=False)
        missing_student = [key for key in missing if key.startswith("model.unet_feature_extractor_cleandift.")]
        if missing_student:
            raise RuntimeError(
                "Robot-DIFT full encoder state lacks Student U-Net tensors "
                f"(e.g. {missing_student[:3]}); it was not written by CleanDIFTImgEncoder"
            )
        if missing:
            warnings.warn(f"Missing Robot-DIFT encoder keys: {missing[:5]}...")
        if unexpected:
            warnings.warn(f"Unexpected Robot-DIFT encoder keys: {unexpected[:5]}...")
        if self.strict_full_encoder_checkpoint and (missing or unexpected):
            raise RuntimeError(
                "Strict Robot-DIFT full encoder load failed: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self._pending_full_encoder_state = None
