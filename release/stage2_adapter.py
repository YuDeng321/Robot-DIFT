"""Portable Stage-II policy adapters referencing one shared frozen encoder.

The adapter contains the trainable readout/policy and small agent state. The
frozen Student and CLIP weights are supplied by their separately versioned
artifacts. This module only converts trusted, locally produced full training
checkpoints; loading the release artifact uses safetensors and JSON.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

from agents.utils.scaler import ActionScaler


FROZEN_PREFIXES = ("img_encoder.rgb_model.", "img_encoder.text_encoder.")
SCALER_KEYS = ("y_mean", "y_std", "y_min", "y_max", "y_bounds_tensor")


class _TrustedCPUUnpickler(pickle.Unpickler):
    """Map tensors in our own CUDA-saved legacy scaler pickle onto CPU."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _student_weight(checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / "unet" / name
        for name in ("diffusion_pytorch_model.bin", "model.safetensors", "diffusion_pytorch_model.safetensors")
        if (checkpoint_dir / "unet" / name).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one Student weight file under {checkpoint_dir / 'unet'}")
    return candidates[0]


def _frozen_key(key: str) -> bool:
    return key.startswith(FROZEN_PREFIXES)


def _frozen_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash exactly the frozen tensors omitted from an adapter."""
    digest = hashlib.sha256()
    for key in sorted(name for name in state if _frozen_key(name)):
        value = state[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _tensor_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected a nonempty tensor state dictionary")
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def export_stage2_adapter(
    full_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    stage1_checkpoint: str | Path,
    clip_model: str | Path,
) -> dict:
    """Convert a trusted full training save into a smaller release adapter."""
    full_checkpoint = Path(full_checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    stage1_checkpoint = Path(stage1_checkpoint).expanduser().resolve()
    clip_model = Path(clip_model).expanduser().resolve()
    metadata_path = stage1_checkpoint / "metadata.json"
    if not all(path.is_file() for path in (full_checkpoint, metadata_path, clip_model)):
        raise FileNotFoundError("Full checkpoint, Stage-I metadata, and CLIP model must exist")
    student_weight = _student_weight(stage1_checkpoint)
    with metadata_path.open(encoding="utf-8") as stream:
        stage1 = json.load(stream)
    if stage1.get("readout") != "paper" and stage1.get("fusion_mode") != "global_to_fine":
        raise ValueError("Stage-II candidate adapter requires a paper-readout or global_to_fine Stage-I encoder")

    state = torch.load(full_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping):
        raise ValueError("Full checkpoint does not contain an agent state dictionary")
    frozen = {key for key in state if _frozen_key(key)}
    if not any(key.startswith(FROZEN_PREFIXES[0]) for key in frozen) or not any(
        key.startswith(FROZEN_PREFIXES[1]) for key in frozen
    ):
        raise ValueError("Full checkpoint lacks expected frozen Student or CLIP weights")
    adapter = _tensor_state({key: value for key, value in state.items() if key not in frozen})
    if not any(key.startswith("img_encoder.readout.") for key in adapter) or not any(
        key.startswith("model.") for key in adapter
    ):
        raise ValueError("Full checkpoint lacks Stage-II readout or action-policy weights")
    pooling = "eot_attention" if any(".token_pooler." in key for key in adapter) else "flatten"

    # This pickle comes only from a trusted local training run. Its release
    # replacement is a tensor-only safetensors file plus a JSON scale flag.
    scaler_path = full_checkpoint.parent / "model_scaler.pkl"
    with scaler_path.open("rb") as stream:
        scaler = _TrustedCPUUnpickler(stream).load()
    if type(scaler) is not ActionScaler:
        raise TypeError("Expected the standard RoboCasa ActionScaler in the local training save")
    scaler_state = _tensor_state({key: getattr(scaler, key) for key in SCALER_KEYS})

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "policy_adapter.safetensors"
    scaler_output = output_dir / "action_scaler.safetensors"
    save_file(adapter, str(weights_path))
    save_file(scaler_state, str(scaler_output))
    manifest = {
        "schema_version": 1,
        "kind": "robot_dift_stage2_policy_adapter",
        "status": "candidate; downstream success not established by export",
        "token_pooling": pooling,
        "stage1": {
            "metadata_sha256": _sha256(metadata_path),
            "student_weight_name": student_weight.name,
            "student_weight_sha256": _sha256(student_weight),
            "vae_latent_mode": stage1.get("vae_latent_mode", "sample"),
            "fusion_mode": stage1["fusion_mode"],
            "epoch": stage1.get("epoch"),
            "ema": stage1.get("ema"),
        },
        "clip_model_sha256": _sha256(clip_model),
        "weights_sha256": _sha256(weights_path),
        "scaler_sha256": _sha256(scaler_output),
        "scale_data": bool(scaler.scale_data),
        "adapter_tensor_count": len(adapter),
        "adapter_tensor_bytes": sum(value.numel() * value.element_size() for value in adapter.values()),
        "excluded_frozen_tensor_count": len(frozen),
        "frozen_state_sha256": _frozen_state_sha256(state),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_stage2_adapter(
    agent: torch.nn.Module,
    artifact_dir: str | Path,
    *,
    stage1_checkpoint: str | Path,
    clip_model: str | Path,
) -> dict:
    """Strictly load an adapter into an agent built from the referenced assets."""
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    stage1_checkpoint = Path(stage1_checkpoint).expanduser().resolve()
    clip_model = Path(clip_model).expanduser().resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "robot_dift_stage2_policy_adapter":
        raise ValueError("Unsupported Stage-II adapter manifest")
    expected_hashes = {
        stage1_checkpoint / "metadata.json": manifest["stage1"]["metadata_sha256"],
        _student_weight(stage1_checkpoint): manifest["stage1"]["student_weight_sha256"],
        clip_model: manifest["clip_model_sha256"],
        artifact_dir / "policy_adapter.safetensors": manifest["weights_sha256"],
        artifact_dir / "action_scaler.safetensors": manifest["scaler_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"Stage-II adapter asset hash mismatch: {path}")
    with (stage1_checkpoint / "metadata.json").open(encoding="utf-8") as stream:
        stage1 = json.load(stream)
    if stage1.get("vae_latent_mode", "sample") != manifest["stage1"]["vae_latent_mode"]:
        raise ValueError("Stage-I VAE mode differs from adapter manifest")

    adapter = load_file(str(artifact_dir / "policy_adapter.safetensors"), device="cpu")
    current = agent.state_dict()
    frozen_digest = manifest.get("frozen_state_sha256")
    if frozen_digest is not None and _frozen_state_sha256(current) != frozen_digest:
        raise ValueError("Stage-II adapter frozen Student/CLIP state differs from its training checkpoint")
    expected = {key for key in current if not _frozen_key(key)}
    # BaseAgent registers robot-state buffers lazily; restore them separately.
    robot_stats = {key: adapter.pop(key) for key in ("robot_states_min", "robot_states_max") if key in adapter}
    expected -= set(robot_stats)
    if set(adapter) != expected:
        raise ValueError(
            f"Stage-II adapter key mismatch: missing={sorted(expected - set(adapter))[:10]}, "
            f"unexpected={sorted(set(adapter) - expected)[:10]}"
        )
    for key, value in adapter.items():
        if value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError(f"Stage-II adapter shape/dtype mismatch: {key}")
    incompatible = agent.load_state_dict(adapter, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {key for key in current if _frozen_key(key)}:
        raise ValueError("Stage-II adapter could not be loaded while preserving the frozen encoder")
    if robot_stats:
        if set(robot_stats) != {"robot_states_min", "robot_states_max"}:
            raise ValueError("Incomplete robot-state bounds in Stage-II adapter")
        agent.robot_states_min = robot_stats["robot_states_min"].to(next(agent.parameters()).device)
        agent.robot_states_max = robot_stats["robot_states_max"].to(next(agent.parameters()).device)
        agent._robot_state_stats_loaded_from_checkpoint = True

    scaler_state = load_file(str(artifact_dir / "action_scaler.safetensors"), device="cpu")
    if set(scaler_state) != set(SCALER_KEYS):
        raise ValueError("Stage-II action scaler has incomplete tensor state")
    device = next(agent.parameters()).device
    scaler = ActionScaler.__new__(ActionScaler)
    scaler.scale_data = bool(manifest["scale_data"])
    scaler.device = str(device)
    for key, value in scaler_state.items():
        setattr(scaler, key, value.to(device))
    scaler.y_bounds = scaler.y_bounds_tensor.detach().cpu().numpy()
    scaler.tensor_y_bounds = scaler.y_bounds_tensor
    agent.scaler = scaler
    agent._scaler_loaded_from_checkpoint = True
    return manifest
