#!/usr/bin/env python3
"""Evaluate a saved Robot-DIFT Stage-II policy without retraining it.

Load the exact Hydra ``.hydra/config.yaml`` from training. The caller must
provide the same Stage-I Student, SD2.1 snapshot and CLIP model through the
``ROBOT_DIFT_*`` environment variables referenced by that config. A separate
adapter artifact may replace the trusted local full checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from release.stage2_adapter import load_stage2_adapter  # noqa: E402
from scripts.release.compare_stage2_transfer_probe import read_episodes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training .hydra/config.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--full-checkpoint", type=Path)
    source.add_argument("--adapter", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--style-id", type=int, choices=tuple(range(12)),
                        help="RoboCasa style 0..11; styles 9/10 are the development/final presets")
    parser.add_argument("--training-epoch", type=int,
                        help="Epoch represented by the checkpoint; used in rollout records")
    parser.add_argument("--scheduler", choices=("ddpm", "ddim"), help="explicit inference scheduler ablation")
    parser.add_argument("--sampling-steps", type=int, help="explicit inference step-count ablation")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL episode file")
    parser.add_argument("--save-videos", action="store_true",
                        help="Record simulator camera frames for trajectory diagnosis")
    parser.add_argument("--video-dir", type=Path,
                        help="Base directory for sim_videos, used with --save-videos")
    parser.add_argument("--check-only", action="store_true", help="instantiate/load without simulator rollout")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    if args.episodes < 1 or args.start_episode < 0:
        parser.error("episodes must be positive and start episode nonnegative")
    if args.training_epoch is not None and args.training_epoch < 0:
        parser.error("training epoch must be nonnegative")
    if args.video_dir is not None and not args.save_videos:
        parser.error("--video-dir requires --save-videos")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("sampling steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-II simulator evaluation requires one CUDA GPU")
    for variable in ("WORK", "ROBOT_DIFT_STAGE1_ENCODER", "ROBOT_DIFT_MODEL_DIR", "ROBOT_DIFT_CLIP_MODEL"):
        if not os.environ.get(variable):
            raise ValueError(f"Missing required environment variable {variable}")
    os.environ.setdefault("ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT", os.environ["ROBOT_DIFT_STAGE1_ENCODER"])
    os.environ["ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE"] = "1"
    args.output = args.output.expanduser().resolve()
    if not args.check_only and args.output.exists():
        raise FileExistsError(f"Episode output already exists: {args.output}")
    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
    config_path = args.config.expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    cfg.simulation.num_episode = args.episodes
    cfg.simulation.start_episode = args.start_episode
    if args.style_id is not None:
        cfg.simulation.style_ids = [args.style_id]
    if args.scheduler is not None:
        cfg.agents.model.scheduler_type = args.scheduler
    if args.sampling_steps is not None:
        cfg.agents.model.num_inference_timesteps = args.sampling_steps
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    agent = hydra.utils.instantiate(cfg.agents)
    if args.full_checkpoint is not None:
        checkpoint = args.full_checkpoint.expanduser().resolve()
        agent.load_pretrained_model(str(checkpoint))
        artifact_digest = _sha256(checkpoint)
        artifact_kind = "trusted_full_checkpoint"
    else:
        artifact = args.adapter.expanduser().resolve()
        manifest = load_stage2_adapter(
            agent, artifact,
            stage1_checkpoint=os.environ["ROBOT_DIFT_STAGE1_ENCODER"],
            clip_model=os.environ["ROBOT_DIFT_CLIP_MODEL"],
        )
        artifact_digest = manifest["weights_sha256"]
        artifact_kind = "release_adapter"
    if agent.scaler is None or any(p.requires_grad for p in agent.img_encoder.rgb_model.parameters()):
        raise RuntimeError("Loaded policy has no action scaler or its Student is not frozen")
    agent.eval()
    info = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "artifact_kind": artifact_kind,
        "artifact_sha256": artifact_digest,
        "student": os.environ["ROBOT_DIFT_STAGE1_ENCODER"],
        "task": list(cfg.simulation.env_name) if not isinstance(cfg.simulation.env_name, str) else [cfg.simulation.env_name],
        "seed": seed,
        "style_ids": list(cfg.simulation.style_ids) if cfg.simulation.style_ids is not None else None,
        "episodes": args.episodes,
        "start_episode": args.start_episode,
        "training_epoch": int(args.training_epoch if args.training_epoch is not None else cfg.epoch),
        "check_only": args.check_only,
        "inference_scheduler": str(cfg.agents.model.scheduler_type),
        "inference_steps": int(cfg.agents.model.num_inference_timesteps),
        "inference_override": args.scheduler is not None or args.sampling_steps is not None,
        "save_videos": args.save_videos,
        "video_dir": str(args.video_dir.resolve()) if args.video_dir is not None else None,
    }
    if args.check_only:
        print(json.dumps(info, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_DIFT_EPISODE_LOG_PATH"] = str(args.output)
    simulation = hydra.utils.instantiate(cfg.simulation)
    with torch.inference_mode():
        metrics = simulation.test_agent(
            agent, step=info["training_epoch"], save_videos=args.save_videos,
            video_dir=str(args.video_dir.resolve()) if args.video_dir is not None else None,
        )
    episodes = read_episodes(args.output, args.episodes)
    info["successes"] = sum(int(row["success"]) for row in episodes.values())
    info["metrics"] = metrics
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
