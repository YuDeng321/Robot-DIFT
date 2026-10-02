#!/usr/bin/env python3
"""Interpolate a trained Stage-I Student with its initialization (WiSE-FT).

For each ``alpha`` the U-Net weights become ``(1 - alpha) * init + alpha * student``
and the learned Student timestep is interpolated the same way, so ``alpha = 1`` is
the trained Student and ``alpha = 0`` its initialization. Each output is a Stage-I
checkpoint directory (Student U-Net, timestep, deploy head, metadata) that the
probes, check_student_artifact.py, export_encoder_release.py, and Stage II load
like any other. The deploy head is copied from the trained checkpoint; the
training-only alignment adapters are not written.

Example:
    python scripts/release/interpolate_student.py \\
        --checkpoint /path/to/encoder/checkpoint-300000-ema \\
        --model-repo "$ROBOT_DIFT_MODEL_DIR" \\
        --alpha 0.25 0.5 0.75 1.0 \\
        --output-root /path/to/interpolated
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

UNET_WEIGHT_NAMES = (
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)
DEPLOY_HEAD_FILE = "deploy_head.safetensors"
# Beyond this relative weight change the initialization is probably not the one
# the Student was trained from (for example a different SD2.1 snapshot).
LARGE_RELATIVE_CHANGE = 0.5


def sd_teacher_initial_timestep() -> float:
    """Learned-timestep initialization of a Student copied from SD2.1."""
    from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner

    return float(inspect.signature(StableFeatureAligner.__init__).parameters["t_init"].default)


def _read_state(path: Path) -> "OrderedDict[str, torch.Tensor]":
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError(f"Invalid U-Net state dictionary in {path}")
    return OrderedDict(state)


def read_student_unet(checkpoint: Path) -> "OrderedDict[str, torch.Tensor]":
    unet_dir = checkpoint / "unet"
    files = [unet_dir / name for name in UNET_WEIGHT_NAMES if (unet_dir / name).is_file()]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Student U-Net weight file under {unet_dir}; found {[f.name for f in files]}"
        )
    return _read_state(files[0])


def read_timestep(checkpoint: Path) -> float:
    path = checkpoint / "timestep.bin"
    state = torch.load(path, map_location="cpu", weights_only=True)
    value = state.get("timestep") if isinstance(state, dict) else state
    if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError(f"Invalid Student timestep in {path}")
    return float(value.detach().float().reshape(()))


def read_metadata(checkpoint: Path) -> dict:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "cleandift" or metadata.get("sd_version") != "sd21":
        raise ValueError(f"Expected an SD2.1 Robot-DIFT Student checkpoint: {checkpoint}")
    components = metadata.get("components", {})
    if str(components.get("student_unet", "")).strip("/") != "unet":
        raise ValueError(f"Checkpoint metadata does not identify its Student U-Net: {checkpoint}")
    if components.get("deploy_head") != DEPLOY_HEAD_FILE or not (checkpoint / DEPLOY_HEAD_FILE).is_file():
        raise ValueError(f"Checkpoint has no deploy head: {checkpoint}")
    return metadata


def read_initialization(metadata: dict, model_repo: Path | None, init_checkpoint: Path | None):
    """Return ``(unet_state, timestep, description)`` of the Student initialization."""
    if init_checkpoint is not None:
        return read_student_unet(init_checkpoint), read_timestep(init_checkpoint), f"checkpoint:{init_checkpoint}"
    student_init = metadata.get("student_init")
    if student_init != "sd_teacher":
        raise ValueError(
            f"This Student was initialized with student_init={student_init!r}, not from SD2.1; "
            "pass that initialization with --init-checkpoint"
        )
    if model_repo is None:
        raise ValueError("--model-repo (or ROBOT_DIFT_MODEL_DIR) is required for an SD2.1-initialized Student")
    weights = model_repo / "unet" / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing SD2.1 U-Net weights: {weights}")
    return _read_state(weights), sd_teacher_initial_timestep(), f"sd21:{model_repo}"


def check_compatible(init: dict, student: dict) -> None:
    if set(init) != set(student):
        missing = sorted(set(student) - set(init))[:5]
        extra = sorted(set(init) - set(student))[:5]
        raise ValueError(f"Initialization and Student U-Net keys differ (missing {missing}, extra {extra})")
    for key, value in student.items():
        if init[key].shape != value.shape:
            raise ValueError(f"Shape of {key} differs: {tuple(init[key].shape)} vs {tuple(value.shape)}")
        if not value.is_floating_point() and not torch.equal(init[key], value):
            raise ValueError(f"Non-floating tensor {key} differs between initialization and Student")


def relative_change(init: dict, student: dict) -> float:
    """``||student - init|| / ||init||`` over all floating-point tensors."""
    delta = norm = 0.0
    for key, value in student.items():
        if value.is_floating_point():
            base = init[key].double()
            delta += float((value.double() - base).pow(2).sum())
            norm += float(base.pow(2).sum())
    return math.sqrt(delta / norm) if norm > 0 else float("inf")


def interpolate(init: dict, student: dict, alpha: float) -> "OrderedDict[str, torch.Tensor]":
    """``(1 - alpha) * init + alpha * student`` in float32; exact at alpha 0 and 1."""
    output = OrderedDict()
    for key, value in student.items():
        if value.is_floating_point():
            output[key] = (init[key].float() * (1.0 - alpha) + value.float() * alpha).contiguous()
        else:
            output[key] = value.clone()
    return output


def write_checkpoint(destination: Path, source: Path, metadata: dict, unet_state: dict,
                     timestep: float, provenance: dict) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        (temporary / "unet").mkdir()
        torch.save(unet_state, temporary / "unet" / "diffusion_pytorch_model.bin")
        torch.save({"timestep": torch.tensor(timestep, dtype=torch.float32)}, temporary / "timestep.bin")
        shutil.copy2(source / DEPLOY_HEAD_FILE, temporary / DEPLOY_HEAD_FILE)
        output_metadata = dict(metadata)
        components = dict(metadata.get("components", {}))
        components.update(
            student_unet="unet/", timestep="timestep.bin", deploy_head=DEPLOY_HEAD_FILE,
            adapters=None, mapping_network=None, full_state_dict=None, robot_dift_encoder_state=None,
        )
        output_metadata["components"] = components
        output_metadata["student_timestep"] = timestep
        output_metadata["weight_interpolation"] = provenance
        (temporary / "metadata.json").write_text(json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="trained Stage-I Student directory, e.g. checkpoint-N-ema")
    parser.add_argument("--model-repo", type=Path, default=os.environ.get("ROBOT_DIFT_MODEL_DIR"),
                        help="SD2.1 snapshot the Student was copied from (default: ROBOT_DIFT_MODEL_DIR)")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Stage-I checkpoint directory of the initialization when it was not SD2.1")
    parser.add_argument("--alpha", type=float, nargs="+", required=True,
                        help="weight on the trained Student, each in [0, 1]")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without reading weights")
    args = parser.parse_args(argv)

    alphas = list(args.alpha)
    if len(set(alphas)) != len(alphas) or not all(math.isfinite(a) and 0.0 <= a <= 1.0 for a in alphas):
        parser.error("--alpha values must be distinct and within [0, 1]")
    source = args.checkpoint.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root == source or source in output_root.parents:
        parser.error("--output-root cannot be inside the source checkpoint")
    destinations = {alpha: output_root / f"{source.name}-alpha{alpha:.2f}" for alpha in alphas}
    if len({path.name for path in destinations.values()}) != len(alphas):
        parser.error("--alpha values must differ at two decimals")
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Outputs already exist: {existing}")

    metadata = read_metadata(source)
    model_repo = args.model_repo.expanduser().resolve() if args.model_repo else None
    init_checkpoint = args.init_checkpoint.expanduser().resolve(strict=True) if args.init_checkpoint else None
    if args.dry_run:
        init = f"checkpoint:{init_checkpoint}" if init_checkpoint else f"sd21:{model_repo}"
        print(json.dumps({"student": str(source), "init": init,
                          "outputs": {f"{a:.2f}": str(p) for a, p in destinations.items()}}, indent=2))
        return 0

    student = read_student_unet(source)
    student_timestep = read_timestep(source)
    init, init_timestep, init_description = read_initialization(metadata, model_repo, init_checkpoint)
    check_compatible(init, student)
    change = relative_change(init, student)
    if change > LARGE_RELATIVE_CHANGE:
        print(f"warning: the Student differs from its initialization by {change:.1%} (relative L2); "
              "check that --model-repo/--init-checkpoint is the snapshot it was trained from", file=sys.stderr)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for alpha, destination in destinations.items():
        timestep = (1.0 - alpha) * init_timestep + alpha * student_timestep
        write_checkpoint(destination, source, metadata, interpolate(init, student, alpha), timestep, {
            "alpha": alpha,
            "student_checkpoint": str(source),
            "student_timestep": student_timestep,
            "init": init_description,
            "init_timestep": init_timestep,
            "relative_change_student_vs_init": change,
            "deploy_head": "copied from the student checkpoint",
        })
        outputs.append({"alpha": alpha, "path": str(destination), "student_timestep": timestep})
    print(json.dumps({"student": str(source), "init": init_description,
                      "relative_change_student_vs_init": change, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
