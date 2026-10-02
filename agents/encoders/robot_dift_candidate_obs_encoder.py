"""Multi-camera Robot-DIFT readout candidate with frozen CLIP text queries.

This Stage-II interface loads the frozen Stage-I Student features. Its
language readout follows ``robot_dift_paper_readout.py``, the same readout
Stage I trains. The Stage-I S2-FPN fusion (the paper readout, or the legacy
head with ``fusion_mode="global_to_fine"``) can initialize the matching
fusion layers; other legacy fusion modes cannot be loaded into them.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import math

import hydra
import torch
from torch import Tensor, nn

from agents.encoders.multi_image_obs_encoder import TensorNormalize, TensorResize
from agents.encoders.robot_dift_deploy_head import load_stage1_global_to_fine_fusion, load_stage1_paper_readout
from agents.encoders.frozen_clip_text import FrozenCLIPTextTokens
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout


class RobotDIFTCandidateObsEncoder(nn.Module):
    """Fuse all RGB views before returning one token per observation frame.

    Inputs are the same RGB observation dictionary used by BaseAgent. Each
    camera is resized to 256×256 and normalized to [-1,1]. The frozen Student
    extracts ``us3/us6/us8`` maps per view. The readout attends from frozen
    CLIP text tokens to each map, max-pools view tokens, and projects them to a
    policy embedding. Output shape is ``[B,1,1,output_dim]`` so BaseAgent's
    temporal reshape yields exactly ``obs_seq_len`` tokens.
    """

    def __init__(
        self,
        shape_meta: Mapping,
        rgb_model,
        clip_model_path: str,
        *,
        output_dim: int = 512,
        feature_keys: Sequence[str] = ("us3", "us6", "us8"),
        resize_shape: Sequence[int] = (256, 256),
        fpn_dim: int = 256,
        model_dim: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 1,
        mlp_hidden_dims: Sequence[int] = (1024, 512),
        token_pooling: str = "flatten",
        feature_cache_max_gib: float = 0.0,
        pretrained_fusion_checkpoint: str | None = None,
        pretrained_readout_scope: str = "fusion",
        text_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if not isinstance(rgb_model, nn.Module):
            rgb_model = hydra.utils.instantiate(rgb_model)
        self.rgb_model = rgb_model
        feature_cache_max_gib = float(feature_cache_max_gib)
        if not math.isfinite(feature_cache_max_gib) or feature_cache_max_gib < 0:
            raise ValueError("feature_cache_max_gib must be finite and nonnegative")
        self.feature_cache_max_bytes = int(feature_cache_max_gib * (1024 ** 3))
        self._feature_cache: OrderedDict[tuple, dict[str, Tensor]] = OrderedDict()
        self._feature_cache_bytes = 0
        self.feature_cache_hits = 0
        self.feature_cache_misses = 0
        self.text_encoder = text_encoder if text_encoder is not None else FrozenCLIPTextTokens(clip_model_path)

        obs_meta = shape_meta["obs"]
        self.rgb_keys = sorted(key for key, item in obs_meta.items() if item.get("type") == "rgb")
        if not self.rgb_keys:
            raise ValueError("RobotDIFTCandidateObsEncoder requires at least one RGB view")
        self.input_shapes = {key: tuple(obs_meta[key]["shape"]) for key in self.rgb_keys}
        for key, shape in self.input_shapes.items():
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"RGB view {key} must have shape [3,H,W]")
        self.transforms = nn.ModuleDict(
            {
                key: nn.Sequential(
                    TensorResize(tuple(resize_shape)),
                    TensorNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                )
                for key in self.rgb_keys
            }
        )
        channels = {key: int(self.rgb_model.feature_dims[key]) for key in feature_keys}
        self.readout = RobotDIFTPaperReadout(
            channels,
            feature_keys=feature_keys,
            fpn_dim=fpn_dim,
            model_dim=model_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            mlp_hidden_dims=mlp_hidden_dims,
            token_pooling=token_pooling,
        )
        if pretrained_readout_scope not in {"fusion", "full"}:
            raise ValueError("pretrained_readout_scope must be 'fusion' or 'full'")
        if pretrained_fusion_checkpoint:
            # "fusion" initializes the S2-FPN only; "full" also copies the CLIP
            # cross-attention, Transformer and MLP trained with the Stage-I policy.
            if pretrained_readout_scope == "full":
                load_stage1_paper_readout(self.readout, pretrained_fusion_checkpoint)
            else:
                load_stage1_global_to_fine_fusion(self.readout, pretrained_fusion_checkpoint)
        self.output_dim = output_dim
        self.alignment_cfg = {"weight": 0.0}
        self.alignment_views = "all"

    @staticmethod
    def _captions(lang_cond, batch: int) -> list[str]:
        if isinstance(lang_cond, str):
            return [lang_cond] * batch
        if isinstance(lang_cond, (tuple, list)) and lang_cond:
            captions = [str(value) for value in lang_cond]
            if len(captions) == batch:
                return captions
            if batch % len(captions) == 0:
                frames = batch // len(captions)
                return [caption for caption in captions for _ in range(frames)]
        raise ValueError(f"Expected one language instruction per sample or trajectory (batch={batch})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def encode_language_goal(self, captions: Sequence[str]) -> Tensor:
        """Use the readout's frozen CLIP tower for the policy's pooled goal."""
        if isinstance(captions, str):
            captions = [captions]
        return self.text_encoder.encode_goal(captions)

    def _features_for_view(
        self,
        key: str,
        image: Tensor,
        captions: list[str],
        frame_ids: Tensor | None,
    ) -> dict[str, Tensor]:
        if self.feature_cache_max_bytes <= 0 or frame_ids is None:
            return self.rgb_model._encode_backbone(image, captions)
        if frame_ids.shape != (image.shape[0], 3):
            raise ValueError("_robot_dift_frame_ids must flatten to [B*T,3]")
        if any(parameter.requires_grad for parameter in self.rgb_model.parameters()):
            raise ValueError("Feature caching requires a frozen Student")
        ids = frame_ids.detach().cpu().tolist()
        cache_keys = [
            (key, int(triple[0]), int(triple[1]), int(triple[2]), caption)
            for triple, caption in zip(ids, captions)
        ]
        # Keep references for the entire batch before LRU insertion/eviction.
        # Otherwise a full cache can evict an earlier hit that this batch
        # still needs when later misses are inserted.
        batch_items = {
            cache_key: self._feature_cache[cache_key]
            for cache_key in cache_keys if cache_key in self._feature_cache
        }
        missing = [index for index, cache_key in enumerate(cache_keys) if cache_key not in self._feature_cache]
        self.feature_cache_hits += len(cache_keys) - len(missing)
        self.feature_cache_misses += len(missing)
        if missing:
            selected = torch.as_tensor(missing, dtype=torch.long, device=image.device)
            computed = self.rgb_model._encode_backbone(
                image.index_select(0, selected), [captions[index] for index in missing]
            )
            # The Student executes in bf16/fp16 on CUDA but exposes float32
            # maps. Those float32 values are exact expansions of the original
            # low-precision activations, so storing the native dtype halves
            # the CPU cache without changing the readout input values.
            cache_dtype = getattr(self.rgb_model, "model_dtype", torch.float32)
            if cache_dtype not in (torch.bfloat16, torch.float16):
                cache_dtype = torch.float32
            for position, index in enumerate(missing):
                item = {
                    name: computed[name][position].detach().to(device="cpu", dtype=cache_dtype).contiguous()
                    for name in self.readout.feature_keys
                }
                batch_items[cache_keys[index]] = item
                item_bytes = sum(value.numel() * value.element_size() for value in item.values())
                if item_bytes > self.feature_cache_max_bytes:
                    continue
                while self._feature_cache_bytes + item_bytes > self.feature_cache_max_bytes:
                    _, old = self._feature_cache.popitem(last=False)
                    self._feature_cache_bytes -= sum(value.numel() * value.element_size() for value in old.values())
                old = self._feature_cache.pop(cache_keys[index], None)
                if old is not None:
                    self._feature_cache_bytes -= sum(value.numel() * value.element_size() for value in old.values())
                self._feature_cache[cache_keys[index]] = item
                self._feature_cache_bytes += item_bytes
        output = {
            name: torch.stack([batch_items[cache_key][name] for cache_key in cache_keys]).to(
                device=image.device, dtype=torch.float32
            )
            for name in self.readout.feature_keys
        }
        for cache_key in cache_keys:
            if cache_key in self._feature_cache:
                self._feature_cache.move_to_end(cache_key)
        return output

    def forward(self, obs_dict: Mapping[str, Tensor], lang_cond=None, alignment_context=None):
        if alignment_context is not None and float(alignment_context.get("weight", 0.0) or 0.0) > 0:
            raise ValueError("Stage-II candidate readout does not compute Teacher alignment")
        first = obs_dict[self.rgb_keys[0]]
        if first.ndim != 4:
            raise ValueError("RGB views must have shape [B,3,H,W]")
        batch = first.shape[0]
        captions = self._captions(lang_cond, batch)
        text_tokens, text_mask = self.text_encoder(captions)
        frame_ids = obs_dict.get("_robot_dift_frame_ids")
        if frame_ids is not None:
            frame_ids = frame_ids.reshape(-1, 3)
        views: list[Mapping[str, Tensor]] = []
        for key in self.rgb_keys:
            image = obs_dict[key]
            if image.ndim != 4 or tuple(image.shape[1:]) != self.input_shapes[key] or image.shape[0] != batch:
                raise ValueError(f"RGB view {key} shape differs from configured {self.input_shapes[key]}")
            image = self.transforms[key](image)
            maps = self._features_for_view(key, image, captions, frame_ids)
            views.append({name: maps[name] for name in self.readout.feature_keys})
        embedding = self.readout(views, text_tokens, text_mask)
        return embedding[:, None, None, :], None

    def output_shape(self):
        return (1, 1, self.output_dim)
