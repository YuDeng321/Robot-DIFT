#!/usr/bin/env python3
"""Check that a saved Stage-I readout transfers exactly into the Stage-II readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.encoders.cleandift_img_encoder import S2FPNGlobalToFineFusion
from agents.encoders.robot_dift_deploy_head import (
    load_stage1_global_to_fine_fusion,
    load_stage1_paper_readout,
    validate_deploy_head,
)
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout

PAPER_READOUT_ARGS = ("fpn_dim", "model_dim", "output_dim", "num_heads", "transformer_layers", "token_pooling")


def _probe_maps(keys, channels):
    torch.manual_seed(0)
    # us3/us6/us8 of a 256x256 image (32x32 latent): 8x8, 16x16, 16x16.
    sizes = ((8, 8), (16, 16), (16, 16))
    if len(keys) != len(sizes):
        raise ValueError("Parity probe expects the three released Student feature maps")
    return {key: torch.randn(1, channels[key], *sizes[index]) for index, key in enumerate(keys)}


def check_paper(checkpoint: Path, metadata: dict, keys: tuple, channels: dict) -> dict:
    config = metadata.get("readout_config") or {}
    kwargs = {name: config[name] for name in PAPER_READOUT_ARGS if name in config}
    kwargs["mlp_hidden_dims"] = tuple(config.get("mlp_hidden_dims", (1024, 512)))
    stage1 = RobotDIFTPaperReadout(channels, feature_keys=keys, **kwargs)
    stage1_tensors = load_stage1_paper_readout(stage1, checkpoint)
    fusion_only = RobotDIFTPaperReadout(channels, feature_keys=keys, fpn_dim=kwargs.get("fpn_dim", 256))
    fusion_tensors = load_stage1_global_to_fine_fusion(fusion_only, checkpoint)

    maps = _probe_maps(keys, channels)
    text = torch.randn(1, 77, 512)
    mask = torch.zeros(1, 77, dtype=torch.bool)
    mask[:, :6] = True
    stage1.eval()
    fusion_only.eval()
    with torch.no_grad():
        fused_source = stage1._fuse_view(maps)
        fused_target = fusion_only._fuse_view(maps)
        output = stage1([maps], text, mask)
    if not torch.isfinite(output).all() or not torch.isfinite(fused_target).all():
        raise AssertionError("Transferred readout produced nonfinite features")
    max_abs_diff = float((fused_source - fused_target).abs().max())
    if max_abs_diff > 1e-6:
        raise AssertionError(f"Stage-I/II fusion output differs: max_abs_diff={max_abs_diff}")
    return {
        "readout": "paper",
        "readout_tensors_transferred": stage1_tensors,
        "fusion_tensors_transferred": fusion_tensors,
        "fused_shape": list(fused_source.shape),
        "policy_token_shape": list(output.shape),
        "max_abs_diff": max_abs_diff,
    }


def check_global_to_fine(checkpoint: Path, metadata: dict, keys: tuple, channels: dict) -> dict:
    config = json.loads(metadata["training_config"])
    fpn_dim = int(config["observation"]["encoder"]["rgb"]["core_kwargs"]["backbone_kwargs"]["fpn_dim"])
    stage1 = S2FPNGlobalToFineFusion(channels, list(keys), fpn_dim, device="cpu")
    fusion = {
        key.removeprefix("feature_fusion."): value
        for key, value in validate_deploy_head(checkpoint).items()
        if key.startswith("feature_fusion.")
    }
    stage1.load_state_dict(fusion, strict=True)
    stage2 = RobotDIFTPaperReadout(channels, feature_keys=keys, fpn_dim=fpn_dim)
    transferred = load_stage1_global_to_fine_fusion(stage2, checkpoint)
    if transferred != len(fusion):
        raise AssertionError("Transferred fusion tensor count differs from saved fusion")
    maps = _probe_maps(keys, channels)
    stage1.eval()
    stage2.eval()
    with torch.no_grad():
        source = stage1(maps)
        target = stage2._fuse_view(maps)
    max_abs_diff = float((source - target).abs().max())
    if not torch.allclose(source, target, atol=1e-6, rtol=0):
        raise AssertionError(f"Stage-I/II fusion output differs: max_abs_diff={max_abs_diff}")
    return {
        "readout": "legacy_global_to_fine",
        "fusion_tensors_transferred": transferred,
        "fused_shape": list(source.shape),
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Stage-I component checkpoint directory")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    keys = tuple(metadata["feature_key"])
    channels = {key: int(metadata["feature_dims"][key]) for key in keys}
    if metadata.get("readout") == "paper":
        report = check_paper(checkpoint, metadata, keys, channels)
    elif metadata.get("fusion_mode") == "global_to_fine":
        report = check_global_to_fine(checkpoint, metadata, keys, channels)
    else:
        raise ValueError("Stage-I checkpoint has neither a paper readout nor a global_to_fine fusion")
    print(json.dumps({"checkpoint": str(checkpoint), **report, "finite": True}, indent=2))


if __name__ == "__main__":
    main()
