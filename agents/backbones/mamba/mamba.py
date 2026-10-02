from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from agents.backbones.transformer.blocks import RMSNorm


def _patch_transformers_generation_exports() -> None:
    """Backfill names imported by mamba-ssm 2.2.x for newer transformers."""
    try:
        import transformers.generation as generation
        from transformers.generation.utils import GenerateDecoderOnlyOutput
    except Exception:
        return

    for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation, name):
            setattr(generation, name, GenerateDecoderOnlyOutput)


_patch_transformers_generation_exports()

try:
    from mamba_ssm.modules.mamba_simple import Mamba as Mamba1
except ImportError:  # pragma: no cover - depends on installed package version
    from mamba_ssm import Mamba as Mamba1

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except Exception:  # pragma: no cover - Mamba2 is optional and may probe CUDA at import time
    Mamba2 = None


def _build_mamba_layer(
    layer_name: str,
    d_model: int,
    ssm_cfg: dict[str, Any],
    device: str | None,
):
    kwargs = dict(ssm_cfg)
    kwargs.pop("layer", None)
    if device is not None:
        kwargs.setdefault("device", device)

    layer_name = layer_name.lower()
    if layer_name in {"mamba", "mamba1"}:
        return Mamba1(d_model=d_model, **kwargs)
    if layer_name == "mamba2":
        if Mamba2 is None:
            raise ImportError("Mamba2 was requested, but the installed mamba_ssm package does not expose it.")
        return Mamba2(d_model=d_model, **kwargs)
    raise ValueError(f"Unsupported Mamba layer type: {layer_name}")


class _MixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ssm_cfg: dict[str, Any],
        d_intermediate: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        layer_name = str(ssm_cfg.get("layer", "Mamba1"))
        self.norm = RMSNorm(d_model, eps=1e-6)
        self.mixer = _build_mamba_layer(layer_name, d_model=d_model, ssm_cfg=ssm_cfg, device=device)

        hidden_dim = int(d_intermediate or 0)
        if hidden_dim > 0:
            self.norm_mlp = RMSNorm(d_model, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            self.norm_mlp = None
            self.mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x))
        if self.mlp is not None and self.norm_mlp is not None:
            x = x + self.mlp(self.norm_mlp(x))
        return x


class MixerModel(nn.Module):
    def __init__(
        self,
        ssm_cfg: dict[str, Any],
        d_model: int,
        n_layer: int,
        d_intermediate: int | None = None,
        device: str | None = None,
        **_: Any,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _MixerBlock(
                    d_model=d_model,
                    ssm_cfg=ssm_cfg,
                    d_intermediate=d_intermediate,
                    device=device,
                )
                for _ in range(int(n_layer))
            ]
        )
        self.norm_f = RMSNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)
