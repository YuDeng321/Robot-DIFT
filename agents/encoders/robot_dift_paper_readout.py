"""Candidate readout for the Robot-DIFT manuscript's S2-FPN interface.

This module implements the manuscript's specified data flow without loading a
diffusion model or downloading CLIP weights. ``views`` is a nonempty list of
per-camera dictionaries mapping coarse-to-fine feature names (normally
``us3``, ``us6``, ``us8``) to ``[B, C_i, H_i, W_i]`` Student feature maps.
``clip_text_tokens`` is the full ``[B, 77, 512]`` hidden-token sequence from a
*frozen CLIP ViT-B/32 text encoder*. The caller owns that encoder; this module
detaches its token input and trains only the lightweight readout adapters.
The caller places all inputs and this module on the same device; incoming
Student/text dtypes are converted to the trainable readout's parameter dtype.
The result is one policy observation embedding ``[B, output_dim]``.

The paper leaves several details open. This candidate makes these choices:

* Lateral projections are 1x1 Conv/GroupNorm/GELU. Each coarse-to-fine fusion
  and the final ``rho_out`` are 3x3 Conv/GroupNorm/GELU blocks. Upsampling is
  bilinear with ``align_corners=False``.
* The Appendix S4.4 adapter order, LayerNorm then Linear, is used. The fixed
  77 CLIP positions preserve a constant MLP input shape. With ``text_mask``,
  positions after EOT are inactive queries, excluded as Transformer keys, and
  zeroed before flattening. Without a mask, all 77 positions remain active.
  No learned queries appear.
* For 2D RoPE, half of every visual key head is rotated by integer column and
  half by integer row at the final feature-map resolution. Text queries have
  no image coordinate and are left unrotated (equivalent to coordinate 0,0).
  Visual values are unrotated. RoPE is applied after key projection, not added
  to the visual token tensor.
* Cross-attention runs independently per view. Its outputs are combined by
  elementwise max before one or more pre-norm Transformer self-attention
  blocks. The 77 output tokens are flattened and passed through a configurable
  MLP. No camera embeddings or view-order-dependent operation is used.

These choices define a reproducible candidate, not an assertion that the
manuscript uniquely specifies these hyperparameters or operator order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLIP_TOKEN_COUNT = 77
CLIP_TOKEN_DIM = 512


def _group_norm(channels: int, max_groups: int) -> nn.GroupNorm:
    groups = min(channels, max_groups)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ConvNormGelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, norm_groups: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            _group_norm(out_channels, norm_groups),
            nn.GELU(),
        )


def _rotate_2d_keys(keys: Tensor, height: int, width: int, base: float) -> Tensor:
    """Apply 2D RoPE to ``[B, heads, height*width, head_dim]`` visual keys."""
    if keys.ndim != 4 or keys.shape[-2] != height * width:
        raise ValueError("keys must have shape [B, heads, height*width, head_dim]")
    head_dim = keys.shape[-1]
    if head_dim % 4:
        raise ValueError("RoPE head_dim must be divisible by four")

    axis_dim = head_dim // 2
    frequencies = base ** (-torch.arange(0, axis_dim, 2, device=keys.device, dtype=torch.float32) / axis_dim)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=keys.device, dtype=torch.float32),
        torch.arange(width, device=keys.device, dtype=torch.float32),
        indexing="ij",
    )
    column_angles = columns.reshape(-1, 1) * frequencies
    row_angles = rows.reshape(-1, 1) * frequencies

    def rotate(axis_keys: Tensor, angles: Tensor) -> Tensor:
        pairs = axis_keys.reshape(*axis_keys.shape[:-1], axis_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        cosine = angles.cos().to(dtype=keys.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=keys.dtype)[None, None, :, :]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    return torch.cat(
        (rotate(keys[..., :axis_dim], column_angles), rotate(keys[..., axis_dim:], row_angles)),
        dim=-1,
    )


class _TextToVisualAttention(nn.Module):
    """CLIP token queries attending to one camera's RoPE-encoded visual keys."""

    def __init__(self, dim: int, num_heads: int, dropout: float, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_base = rope_base
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        text_queries: Tensor,
        visual_tokens: Tensor,
        height: int,
        width: int,
        text_mask: Tensor,
    ) -> Tensor:
        batch, text_count, dim = text_queries.shape

        def split_heads(x: Tensor) -> Tensor:
            return x.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.q_proj(text_queries))
        keys = split_heads(self.k_proj(visual_tokens))
        values = split_heads(self.v_proj(visual_tokens))
        keys = _rotate_2d_keys(keys, height, width, self.rope_base)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, values)
        attended = attended.transpose(1, 2).reshape(batch, text_count, dim)
        return self.out_proj(attended).masked_fill(~text_mask.unsqueeze(-1), 0)


class RobotDIFTPaperReadout(nn.Module):
    """Standalone candidate manuscript readout with an explicit tensor interface.

    Args:
        feature_channels: Channels for each key in ``feature_keys``.
        feature_keys: Coarse-to-fine Student feature keys; default is the paper's
            ``us3/us6/us8`` selection. Equal adjacent resolutions are allowed.
        fpn_dim: Shared lateral and fused-map channel count.
        model_dim: Shared text/visual attention and Transformer dimension.
        output_dim: Width of the final policy observation embedding.
        num_heads: Attention heads. ``model_dim / num_heads`` must be a multiple
            of four for the two-axis rotary convention.
        transformer_layers: Number of self-attention blocks after view pooling.
        mlp_hidden_dims: Hidden widths before the final observation projection.
        norm_groups: Maximum GroupNorm group count, adjusted to divide channels.
        dropout: Attention, Transformer, and MLP dropout probability.
        rope_base: Rotary frequency base.

    ``clip_text_tokens`` must always be ``[B, 77, 512]``. Optional boolean
    ``text_mask`` has shape ``[B, 77]`` and marks positions through EOT as
    True; padding after EOT is False. The CLIP model is intentionally external
    and frozen; this module never downloads weights.
    """

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        *,
        feature_keys: Sequence[str] = ("us3", "us6", "us8"),
        fpn_dim: int = 256,
        model_dim: int = 256,
        output_dim: int = 512,
        num_heads: int = 8,
        transformer_layers: int = 1,
        mlp_hidden_dims: Sequence[int] = (1024, 512),
        norm_groups: int = 32,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        token_pooling: str = "flatten",
    ) -> None:
        super().__init__()
        self.feature_keys = tuple(feature_keys)
        if not self.feature_keys or len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("feature_keys must be nonempty and unique")
        if set(feature_channels) != set(self.feature_keys):
            raise ValueError("feature_channels must contain exactly feature_keys")
        dimensions = (fpn_dim, model_dim, output_dim, num_heads, transformer_layers, norm_groups)
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("dimensions, num_heads, transformer_layers, and norm_groups must be positive integers")
        if any(not isinstance(channel, int) or channel <= 0 for channel in feature_channels.values()):
            raise ValueError("feature channel counts must be positive integers")
        if any(not isinstance(width, int) or width <= 0 for width in mlp_hidden_dims):
            raise ValueError("MLP hidden dimensions must be positive integers")
        if model_dim % num_heads or (model_dim // num_heads) % 4:
            raise ValueError("model_dim must divide num_heads with a head dimension divisible by four")
        if not 0 <= dropout < 1 or rope_base <= 0:
            raise ValueError("dropout must be in [0, 1) and rope_base must be positive")
        if token_pooling not in {"flatten", "eot_attention"}:
            raise ValueError("token_pooling must be 'flatten' or 'eot_attention'")

        self.feature_channels = dict(feature_channels)
        self.token_pooling = token_pooling
        self.lateral = nn.ModuleDict(
            {key: _ConvNormGelu(self.feature_channels[key], fpn_dim, 1, norm_groups) for key in self.feature_keys}
        )
        self.fusions = nn.ModuleList(
            [_ConvNormGelu(2 * fpn_dim, fpn_dim, 3, norm_groups) for _ in self.feature_keys[1:]]
        )
        self.output_fusion = _ConvNormGelu(fpn_dim, fpn_dim, 3, norm_groups)
        self.visual_adapter = nn.Sequential(nn.LayerNorm(fpn_dim), nn.Linear(fpn_dim, model_dim))
        self.text_adapter = nn.Sequential(nn.LayerNorm(CLIP_TOKEN_DIM), nn.Linear(CLIP_TOKEN_DIM, model_dim))
        self.cross_attention = _TextToVisualAttention(model_dim, num_heads, dropout, rope_base)
        self.transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * model_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(transformer_layers)
            ]
        )
        if token_pooling == "eot_attention":
            self.token_pooler = nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.eot_norm = nn.LayerNorm(model_dim)
            self.mean_norm = nn.LayerNorm(model_dim)
            mlp_input_dim = 2 * model_dim
        else:
            mlp_input_dim = CLIP_TOKEN_COUNT * model_dim
        mlp_widths = [mlp_input_dim, *mlp_hidden_dims, output_dim]
        mlp_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(mlp_widths, mlp_widths[1:])):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if index < len(mlp_widths) - 2:
                mlp_layers.extend((nn.GELU(), nn.Dropout(dropout)))
        self.observation_mlp = nn.Sequential(*mlp_layers)

    def _fuse_view(self, features: Mapping[str, Tensor]) -> Tensor:
        fused: Tensor | None = None
        for index, key in enumerate(self.feature_keys):
            # Frozen Student maps may be fp16 while the trainable readout is fp32.
            input_dtype = self.lateral[key][0].weight.dtype
            projected = self.lateral[key](features[key].to(dtype=input_dtype))
            if fused is None:
                fused = projected
            else:
                fused = F.interpolate(fused, size=projected.shape[-2:], mode="bilinear", align_corners=False)
                fused = self.fusions[index - 1](torch.cat((fused, projected), dim=1))
        assert fused is not None
        return self.output_fusion(fused)

    def _validate_inputs(
        self,
        views: Sequence[Mapping[str, Tensor]],
        clip_text_tokens: Tensor,
        text_mask: Tensor | None,
    ) -> None:
        if clip_text_tokens.ndim != 3 or clip_text_tokens.shape[1:] != (CLIP_TOKEN_COUNT, CLIP_TOKEN_DIM):
            raise ValueError("clip_text_tokens must have shape [B, 77, 512]")
        if not views:
            raise ValueError("views must contain at least one camera")
        batch = clip_text_tokens.shape[0]
        if text_mask is not None:
            if text_mask.shape != (batch, CLIP_TOKEN_COUNT) or text_mask.dtype != torch.bool:
                raise ValueError("text_mask must be a boolean tensor of shape [B, 77]")
            if text_mask.device != clip_text_tokens.device:
                raise ValueError("text_mask and clip_text_tokens must be on the same device")
            if not bool(text_mask.any(dim=1).all()):
                raise ValueError("each text_mask row must contain at least one active token")
            if bool((text_mask[:, 1:] & ~text_mask[:, :-1]).any()):
                raise ValueError("text_mask must be a True prefix through EOT followed by False padding")
        for view_index, features in enumerate(views):
            if set(features) != set(self.feature_keys):
                raise ValueError(f"view {view_index} must contain exactly {self.feature_keys}")
            previous_hw = (0, 0)
            for key in self.feature_keys:
                feature = features[key]
                if feature.ndim != 4 or feature.shape[:2] != (batch, self.feature_channels[key]):
                    raise ValueError(
                        f"view {view_index} feature {key} must have shape "
                        f"[B, {self.feature_channels[key]}, H, W]"
                    )
                height, width = feature.shape[-2:]
                if height <= 0 or width <= 0 or height < previous_hw[0] or width < previous_hw[1]:
                    raise ValueError("feature maps must be ordered coarse to fine in both spatial dimensions")
                previous_hw = (height, width)

    def encode_views(
        self,
        views: Sequence[Mapping[str, Tensor]],
        clip_text_tokens: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        """Return pre-pooling cross-attention tokens ``[B, V, 77, model_dim]``."""
        self._validate_inputs(views, clip_text_tokens, text_mask)
        if text_mask is None:
            text_mask = torch.ones(
                clip_text_tokens.shape[:2], dtype=torch.bool, device=clip_text_tokens.device
            )
        text_dtype = self.text_adapter[1].weight.dtype
        text_queries = self.text_adapter(clip_text_tokens.detach().to(dtype=text_dtype))
        per_view = []
        for features in views:
            fused_map = self._fuse_view(features)
            height, width = fused_map.shape[-2:]
            visual_tokens = self.visual_adapter(fused_map.flatten(2).transpose(1, 2))
            per_view.append(self.cross_attention(text_queries, visual_tokens, height, width, text_mask))
        return torch.stack(per_view, dim=1)

    @staticmethod
    def pool_views(per_view_tokens: Tensor) -> Tensor:
        """Elementwise max over cameras: ``[B,V,77,C]`` to ``[B,77,C]``."""
        if per_view_tokens.ndim != 4 or per_view_tokens.shape[1] == 0:
            raise ValueError("per_view_tokens must have shape [B, V, N, C] with V > 0")
        return per_view_tokens.amax(dim=1)

    def forward(
        self,
        views: Sequence[Mapping[str, Tensor]],
        clip_text_tokens: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        """Read Student feature maps and frozen CLIP tokens into ``[B, output_dim]``."""
        attended = self.pool_views(self.encode_views(views, clip_text_tokens, text_mask))
        if text_mask is None:
            text_mask = torch.ones(
                clip_text_tokens.shape[:2], dtype=torch.bool, device=clip_text_tokens.device
            )
        for layer in self.transformer:
            attended = layer(attended, src_key_padding_mask=~text_mask)
            attended = attended.masked_fill(~text_mask.unsqueeze(-1), 0)
        if self.token_pooling == "flatten":
            summary = attended.flatten(start_dim=1)
        else:
            eot_index = text_mask.sum(dim=1) - 1
            eot = attended[torch.arange(attended.shape[0], device=attended.device), eot_index]
            pooled, _ = self.token_pooler(
                eot.unsqueeze(1), attended, attended,
                key_padding_mask=~text_mask, need_weights=False,
            )
            mean = attended.sum(dim=1) / text_mask.sum(dim=1, keepdim=True)
            summary = torch.cat(
                (self.eot_norm(eot + pooled.squeeze(1)), self.mean_norm(mean)), dim=-1
            )
        return self.observation_mlp(summary)
