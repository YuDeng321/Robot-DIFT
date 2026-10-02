"""Compact Stage-I readout export and checked transfer into Stage II.

The frozen SD2.1 Teacher, VAE and alignment projection heads are training
dependencies, not part of the deployable Student/readout artifact.

Two readouts can be exported:

* ``readout="paper"``: the S2-FPN + CLIP readout trained in Stage I by
  :class:`agents.encoders.robot_dift_stage1_encoder.RobotDIFTStage1Encoder`.
  Tensors are stored as ``readout.<name>`` and match
  :class:`RobotDIFTPaperReadout` exactly.
* ``readout="legacy"``: the ``feature_fusion``/query head of
  :class:`CleanDIFTImgEncoder`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn


DEPLOY_HEAD_FILE = "deploy_head.safetensors"
_HEAD_MODULES = ("feature_fusion", "queries_pooling", "final_proj", "query_proj")
_PAPER_PREFIX = "readout."
_FUSION_PREFIXES = ("lateral.", "fusions.", "output_fusion.")


def collect_deploy_head_state(encoder: nn.Module) -> dict[str, Tensor]:
    """Collect only trained, policy-facing visual readout parameters."""
    readout_state = getattr(encoder, "readout_state", None)
    if callable(readout_state):
        selected = readout_state()
        if not selected or not all(key.startswith(_PAPER_PREFIX) for key in selected):
            raise ValueError("Paper readout export must contain only readout.* tensors")
        return selected
    if not isinstance(getattr(encoder, "feature_fusion", None), nn.Module):
        raise ValueError("Robot-DIFT encoder has no feature_fusion module")
    selected: dict[str, Tensor] = {}
    for name in _HEAD_MODULES:
        module = getattr(encoder, name, None)
        if module is None:
            continue
        if not isinstance(module, nn.Module):
            raise TypeError(f"{name} is not a torch module")
        for key, value in module.state_dict().items():
            selected[f"{name}.{key}"] = value.detach().cpu().contiguous()
    if not any(key.startswith("feature_fusion.") for key in selected):
        raise ValueError("Robot-DIFT feature_fusion has no saved tensors")
    return selected


def _read_metadata(checkpoint_dir: Path) -> dict:
    with (checkpoint_dir / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_deploy_head(checkpoint_dir: str | Path) -> dict[str, Tensor]:
    """Read a compact head and reject missing, unexpected or nonfinite tensors."""
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    metadata = _read_metadata(checkpoint_dir)
    if metadata.get("components", {}).get("deploy_head") != DEPLOY_HEAD_FILE:
        raise ValueError("Checkpoint metadata lacks the compact deploy head")
    state = load_file(str(checkpoint_dir / DEPLOY_HEAD_FILE), device="cpu")
    if metadata.get("readout") == "paper":
        allowed = (_PAPER_PREFIX,)
        if not any(key.startswith(_PAPER_PREFIX + "lateral.") for key in state):
            raise ValueError("Paper deploy head has no S2-FPN lateral tensors")
    else:
        allowed = tuple(f"{name}." for name in _HEAD_MODULES)
        if not state or not any(key.startswith("feature_fusion.") for key in state):
            raise ValueError("Compact deploy head has no fusion tensors")
    for key, value in state.items():
        if not key.startswith(allowed):
            raise ValueError(f"Unexpected deploy-head tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite deploy-head tensor: {key}")
    return state


def _stage1_fusion_tensors(checkpoint_dir: Path, metadata: dict) -> dict[str, Tensor]:
    source = validate_deploy_head(checkpoint_dir)
    if metadata.get("readout") == "paper":
        return {
            key.removeprefix(_PAPER_PREFIX): value
            for key, value in source.items()
            if key.removeprefix(_PAPER_PREFIX).startswith(_FUSION_PREFIXES)
        }
    if metadata.get("fusion_mode") != "global_to_fine":
        raise ValueError("Stage-I fusion transfer requires a paper or global_to_fine checkpoint")
    return {
        key.removeprefix("feature_fusion."): value
        for key, value in source.items()
        if key.startswith("feature_fusion.")
    }


def load_stage1_global_to_fine_fusion(readout: nn.Module, checkpoint_dir: str | Path) -> int:
    """Strictly transfer a trained Stage-I S2-FPN fusion into the Stage-II readout.

    Accepts paper-readout and legacy ``global_to_fine`` checkpoints. Only
    the lateral/coarse-to-fine/output fusion tensors are copied.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata = _read_metadata(checkpoint_dir)
    if tuple(metadata.get("feature_key", ())) != tuple(readout.feature_keys):
        raise ValueError("Stage-I and Stage-II fusion feature keys differ")
    fusion = _stage1_fusion_tensors(checkpoint_dir, metadata)
    target = readout.state_dict()
    target_fusion_keys = {key for key in target if key.startswith(_FUSION_PREFIXES)}
    if set(fusion) != target_fusion_keys:
        raise ValueError("Stage-I and Stage-II fusion parameter names differ")
    for key, value in fusion.items():
        if target[key].shape != value.shape:
            raise ValueError(f"Stage-I fusion tensor shape differs: {key}")
        target[key] = value
    readout.load_state_dict(target, strict=True)
    return len(fusion)


def load_stage1_paper_readout(readout: nn.Module, checkpoint_dir: str | Path) -> int:
    """Strictly copy the complete Stage-I paper readout into ``readout``."""
    checkpoint_dir = Path(checkpoint_dir)
    metadata = _read_metadata(checkpoint_dir)
    if metadata.get("readout") != "paper":
        raise ValueError("Full readout transfer requires a Stage-I paper-readout checkpoint")
    if tuple(metadata.get("feature_key", ())) != tuple(readout.feature_keys):
        raise ValueError("Stage-I and Stage-II readout feature keys differ")
    state = {key.removeprefix(_PAPER_PREFIX): value for key, value in validate_deploy_head(checkpoint_dir).items()}
    readout.load_state_dict(state, strict=True)
    return len(state)
