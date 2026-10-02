#!/usr/bin/env python3
"""Strictly reload a saved Stage-I Student and run one finite forward pass.

This is a deployment-interface check, not a correspondence or robot-success
measurement. Supply the same SD2.1 snapshot used by Stage-I training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor  # noqa: E402
from agents.encoders.robot_dift_deploy_head import validate_deploy_head  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Stage-I encoder checkpoint directory")
    parser.add_argument("--model-repo", type=Path, required=True, help="local SD2.1 Diffusers snapshot")
    parser.add_argument("--device", default="cuda", help="torch device for the forward check")
    parser.add_argument("--latent-mode", choices=("auto", "sample", "mode"), default="auto",
                        help="auto follows Stage-I metadata; legacy snapshots imply sample")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    model_repo = args.model_repo.expanduser().resolve()
    model = RobotDIFTStudentFeatureExtractor(
        checkpoint_dir=str(checkpoint),
        model_repo=str(model_repo),
        device=args.device,
        vae_latent_mode=args.latent_mode,
    )
    deploy_head_state = validate_deploy_head(checkpoint)
    image = torch.full((1, 3, 256, 256), 0.25, device=args.device, dtype=model.model_dtype)
    with torch.no_grad():
        maps = model._encode_backbone(image, ["press the button"])
    expected = ("us3", "us6", "us8")
    if tuple(maps) != expected:
        raise RuntimeError(f"Student feature keys differ from {expected}: {tuple(maps)}")
    if any(not torch.isfinite(maps[key]).all().item() for key in expected):
        raise RuntimeError("Student produced nonfinite feature values")
    report = {
        "checkpoint": str(checkpoint),
        "model_repo": str(model_repo),
        "device": str(args.device),
        "vae_latent_mode": model.vae_latent_mode,
        "student_only_strict_load": True,
        "deploy_head_tensors": len(deploy_head_state),
        "feature_shapes": {key: list(maps[key].shape) for key in expected},
        "finite": True,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Stage-I Student strict reload and finite forward: PASS")
        for key, shape in report["feature_shapes"].items():
            print(f"  {key}: {shape}")


if __name__ == "__main__":
    main()
