#!/usr/bin/env python3
"""Held-out DROID probe of frozen Students: robot-state decoding and camera use.

Every checkpoint encodes the same frames, and two questions are measured:

1. Do the deployed us3/us6/us8 maps carry control-relevant information? The
   maps are average-pooled to a grid per camera, and a ridge probe (dual form,
   one lambda per target dimension chosen by grouped inner CV) predicts the
   end-effector position and gripper at frame t, and the end-effector
   displacement and gripper ``--horizon`` steps later. R^2 is out-of-fold
   over episodes, for each camera alone and for all cameras together.
2. How does the Stage-I readout use the cameras (checkpoints trained with the
   paper readout)? The report gives the share of max-pooled query channels
   each camera wins (ties go to the first camera), overall and by gripper
   state; how far the policy token moves without each camera, and how far
   the single-camera token is from the full one, both in units of the
   token's spread across frames; and the R^2 of the same probe on the token.

Frames come from DROID failure episodes by default, which Stage-I training
(success episodes only) never sees. The first --checkpoint is the reference:
use the SD2.1 initialization written by
``scripts/release/interpolate_student.py --alpha 0`` as the DIFT baseline.
Paired R^2 differences are bootstrapped over episodes. This is a frozen-feature
diagnostic; robot success still needs Stage II.

Example:
    python scripts/probes/droid_multiview_probe.py \\
        --data-root "$ROBOT_DIFT_DATA_ROOT" --samples /path/to/droid_probe_frames.npz \\
        --checkpoint sd21_dift=/path/to/interpolated/checkpoint-300000-ema-alpha0.00 \\
        --checkpoint robot_dift=/path/to/encoder/checkpoint-300000-ema \\
        --output-dir /path/to/droid_multiview
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import linalg as scipy_linalg
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAMERAS = ("wrist_image_left", "exterior_image_1_left", "exterior_image_2_left")
FEATURE_KEYS = ("us3", "us6", "us8")
TARGET_GROUPS = {
    "ee_position": (0, 3),
    "gripper": (3, 4),
    "ee_displacement": (4, 7),
    "future_gripper": (7, 8),
}
SPLIT_PATTERNS = {"failure": r".*/failure/.*", "success": r".*/success/.*", "all": None}
LAMBDA_SCALES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
READOUT_ARGS = ("fpn_dim", "model_dim", "output_dim", "num_heads", "transformer_layers", "token_pooling")


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------

@dataclass
class Samples:
    images: dict[str, np.ndarray]  # camera -> uint8 [N, H, W, 3]
    targets: np.ndarray             # float64 [N, 8], columns follow TARGET_GROUPS
    episodes: np.ndarray            # int [N], index into episode_names
    frames: np.ndarray              # int [N]
    instructions: list[str]
    episode_names: list[str]

    def __len__(self) -> int:
        return len(self.episodes)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, targets=self.targets, episodes=self.episodes, frames=self.frames,
            instructions=np.asarray(self.instructions, dtype=str),
            episode_names=np.asarray(self.episode_names, dtype=str),
            cameras=np.asarray(list(self.images), dtype=str),
            **{f"image__{camera}": images for camera, images in self.images.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "Samples":
        data = np.load(path, allow_pickle=False)
        return cls(
            images={str(camera): data[f"image__{camera}"] for camera in data["cameras"]},
            targets=data["targets"], episodes=data["episodes"], frames=data["frames"],
            instructions=[str(text) for text in data["instructions"]],
            episode_names=[str(name) for name in data["episode_names"]],
        )


def select_frames(num_steps: int, per_episode: int, horizon: int) -> list[int]:
    """Evenly spaced frames ``t`` with ``t + horizon`` inside the episode."""
    last = num_steps - 1 - horizon
    if last < 0 or per_episode < 1:
        return []
    count = min(per_episode, last + 1)
    return sorted({int(round(value)) for value in np.linspace(0, last, count)})


def _decode(image) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, np.bytes_)):
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {array.shape}")
    return array


def samples_from_episodes(
    episodes: Iterable[Mapping],
    *,
    cameras: Sequence[str],
    per_episode: int,
    horizon: int,
) -> Samples:
    """Collect probe frames from episodes given as dicts of per-step arrays.

    Each episode has ``episode_id``, ``instruction``, ``images`` (camera -> list of
    encoded bytes or HxWx3 arrays), ``cartesian_position`` [T, 6], and
    ``gripper_position`` [T, 1].
    """
    images = {camera: [] for camera in cameras}
    targets, episode_index, frames, instructions, names = [], [], [], [], []
    for episode in episodes:
        position = np.asarray(episode["cartesian_position"], dtype=np.float64)[:, :3]
        gripper = np.asarray(episode["gripper_position"], dtype=np.float64).reshape(len(position), -1)[:, :1]
        chosen = select_frames(len(position), per_episode, horizon)
        if not chosen:
            continue
        names.append(str(episode["episode_id"]))
        for frame in chosen:
            future = frame + horizon
            targets.append(np.concatenate([
                position[frame], gripper[frame], position[future] - position[frame], gripper[future],
            ]))
            for camera in cameras:
                images[camera].append(_decode(episode["images"][camera][frame]))
            episode_index.append(len(names) - 1)
            frames.append(frame)
            instructions.append(str(episode.get("instruction", "")))
    if not names:
        raise ValueError("No episode was long enough for the requested horizon")
    return Samples(
        images={camera: np.stack(values) for camera, values in images.items()},
        targets=np.stack(targets), episodes=np.asarray(episode_index), frames=np.asarray(frames),
        instructions=instructions, episode_names=names,
    )


def read_droid_episodes(data_root: str, dataset: str, split: str, cameras: Sequence[str], limit: int):
    """Yield DROID RLDS episodes whose file path matches ``split`` as plain arrays."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder(dataset, data_dir=data_root)
    decoders = {"steps": {"observation": {camera: tfds.decode.SkipDecoding() for camera in cameras}}}
    pattern = SPLIT_PATTERNS[split]
    matched = 0
    for episode in builder.as_dataset(split="train", decoders=decoders, shuffle_files=False):
        path = episode["episode_metadata"]["file_path"].numpy().decode()
        if pattern and not re.fullmatch(pattern, path):
            continue
        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        instruction = next(
            (steps[0][key].decode().strip() for key in ("language_instruction", "language_instruction_2",
                                                        "language_instruction_3")
             if key in steps[0] and steps[0][key].decode().strip()),
            "",
        )
        yield {
            "episode_id": path,
            "instruction": instruction,
            "images": {camera: [step["observation"][camera] for step in steps] for camera in cameras},
            "cartesian_position": np.stack([step["observation"]["cartesian_position"] for step in steps]),
            "gripper_position": np.stack([step["observation"]["gripper_position"] for step in steps]),
        }
        matched += 1
        if matched >= limit:
            return


# ----------------------------------------------------------------------------
# Features and camera statistics
# ----------------------------------------------------------------------------

def student_input(images: np.ndarray, size: int, device: str) -> torch.Tensor:
    """uint8 [B,H,W,3] -> [-1, 1] float [B,3,size,size], as the Stage-II encoder feeds the Student."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return tensor * 2.0 - 1.0


def grid_pool(maps: Mapping[str, torch.Tensor], grid: int) -> torch.Tensor:
    return torch.cat([F.adaptive_avg_pool2d(maps[key].float(), grid).flatten(1) for key in FEATURE_KEYS], dim=1)


@torch.no_grad()
def camera_statistics(readout, views, text_tokens, text_mask) -> dict[str, torch.Tensor]:
    """Per-sample camera use of the paper readout for ``views`` (one map dict per camera)."""
    per_view = readout.encode_views(views, text_tokens, text_mask)  # [B, V, Q, C]
    count = per_view.shape[1]
    winners = per_view.argmax(dim=1)
    active = text_mask[:, :, None].expand_as(winners)
    wins = torch.stack([((winners == view) & active).flatten(1).sum(1) for view in range(count)], dim=1)
    share = wins.float() / active.flatten(1).sum(1, keepdim=True).clamp_min(1)
    token = readout(views, text_tokens, text_mask).float()
    result = {"win_share": share, "token": token}
    result["single_distance"] = torch.stack([
        (token - readout([views[view]], text_tokens, text_mask).float()).norm(dim=-1)
        for view in range(count)
    ], dim=1)
    if count > 1:
        result["drop_distance"] = torch.stack([
            (token - readout(views[:view] + views[view + 1:], text_tokens, text_mask).float()).norm(dim=-1)
            for view in range(count)
        ], dim=1)
    return result


def load_paper_readout(checkpoint: Path, channels: Mapping[str, int]):
    """Return ``(readout, reason_if_unavailable)`` for a checkpoint's trained paper readout."""
    from agents.encoders.robot_dift_deploy_head import load_stage1_paper_readout
    from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("readout") != "paper":
        return None, "checkpoint has no paper readout"
    interpolation = metadata.get("weight_interpolation")
    if interpolation and float(interpolation.get("alpha", 1.0)) != 1.0:
        return None, "the readout was trained on a different Student (weight interpolation)"
    config = metadata.get("readout_config") or {}
    kwargs = {name: config[name] for name in READOUT_ARGS if name in config}
    kwargs["mlp_hidden_dims"] = tuple(config.get("mlp_hidden_dims", (1024, 512)))
    readout = RobotDIFTPaperReadout(
        dict(channels), feature_keys=tuple(config.get("feature_keys", FEATURE_KEYS)), **kwargs
    )
    load_stage1_paper_readout(readout, checkpoint)
    return readout.eval(), None


def encode_checkpoint(samples: Samples, extractor, *, cameras, image_size, grid, batch_size, device,
                      prompt_mode="instruction", readout=None, text_encoder=None) -> dict:
    """Grid-pooled features per camera and, with a readout, per-sample camera statistics."""
    pooled = {camera: [] for camera in cameras}
    stats: dict[str, list] = {}
    for start in range(0, len(samples), batch_size):
        rows = slice(start, start + batch_size)
        captions = samples.instructions[rows] if prompt_mode == "instruction" else [""] * len(
            samples.instructions[rows])
        views = []
        for camera in cameras:
            maps = extractor._encode_backbone(
                student_input(samples.images[camera][rows], image_size, device), list(captions)
            )
            pooled[camera].append(grid_pool(maps, grid).cpu().to(torch.float16))
            views.append(maps)
        if readout is not None:
            tokens, mask = text_encoder(list(samples.instructions[rows]))
            batch_stats = camera_statistics(readout, views, tokens.to(device), mask.to(device))
            for name, value in batch_stats.items():
                stats.setdefault(name, []).append(value.cpu())
    result = {"features": {camera: torch.cat(values).numpy() for camera, values in pooled.items()}}
    if stats:
        result["camera_stats"] = {name: torch.cat(values).numpy() for name, values in stats.items()}
    return result


# ----------------------------------------------------------------------------
# Probe and statistics
# ----------------------------------------------------------------------------

def grouped_folds(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds that keep every episode on one side."""
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"Need at least {folds} episodes for {folds}-fold grouped CV, got {len(unique)}")
    order = np.random.default_rng(seed).permutation(unique)
    fold_of_group = {group: index % folds for index, group in enumerate(order)}
    fold = np.asarray([fold_of_group[group] for group in groups])
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(folds)]


def _standardize(train: np.ndarray, *others: np.ndarray):
    train = train.astype(np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    return [(train - mean) / std] + [(other.astype(np.float64) - mean) / std for other in others]


def _ridge_path(k_train: np.ndarray, y_train: np.ndarray, k_eval: np.ndarray, lambdas) -> np.ndarray:
    """Dual ridge predictions ``[L, n_eval, dims]`` for centered ``y_train``."""
    eigenvalues, eigenvectors = scipy_linalg.eigh(k_train, check_finite=True)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ y_train
    left = k_eval @ eigenvectors
    return np.stack([left @ (projected / (eigenvalues + lam)[:, None]) for lam in lambdas])


def ridge_oof_predictions(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, *,
                          folds: int = 5, inner_folds: int = 3, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions; lambda per target dimension from grouped inner CV."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.zeros_like(targets)
    for train, test in grouped_folds(groups, folds, seed):
        x_train, x_test = _standardize(features[train], features[test])
        k_train = x_train @ x_train.T
        k_test = x_test @ x_train.T
        scale = max(float(np.trace(k_train)) / len(train), 1e-12)
        lambdas = [factor * scale for factor in LAMBDA_SCALES]
        y_scale = targets[train].std(axis=0) + 1e-12
        errors = np.zeros((len(lambdas), targets.shape[1]))
        for inner_train, inner_val in grouped_folds(groups[train], inner_folds, seed + 1):
            y_inner = targets[train][inner_train] / y_scale
            mean = y_inner.mean(axis=0)
            path = _ridge_path(k_train[np.ix_(inner_train, inner_train)], y_inner - mean,
                               k_train[np.ix_(inner_val, inner_train)], lambdas) + mean
            errors += ((path - targets[train][inner_val] / y_scale) ** 2).sum(axis=1)
        best = errors.argmin(axis=0)
        y_train = targets[train] / y_scale
        mean = y_train.mean(axis=0)
        path = _ridge_path(k_train, y_train - mean, k_test, lambdas) + mean
        predictions[test] = path[best, :, np.arange(targets.shape[1])].T * y_scale
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite ridge predictions; DROID probe cannot be compared")
    return predictions


def r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Mean over target dimensions of 1 - SS_res / SS_tot."""
    residual = ((targets - predictions) ** 2).sum(axis=0)
    total = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    valid = total > 0
    return float(np.mean(1.0 - residual[valid] / total[valid])) if valid.any() else float("nan")


def _episode_rows(groups: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups == group) for group in np.unique(groups)]


def bootstrap_r2(targets, predictions, groups, *, draws, seed) -> list[float]:
    rows_by_episode = _episode_rows(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        rows = np.concatenate([rows_by_episode[i] for i in rng.integers(len(rows_by_episode), size=len(rows_by_episode))])
        values.append(r2(targets[rows], predictions[rows]))
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def paired_r2_difference(targets, reference, candidate, groups, *, draws, seed) -> dict:
    """R^2(candidate) - R^2(reference) with an episode bootstrap."""
    rows_by_episode = _episode_rows(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        rows = np.concatenate([rows_by_episode[i] for i in rng.integers(len(rows_by_episode), size=len(rows_by_episode))])
        deltas.append(r2(targets[rows], candidate[rows]) - r2(targets[rows], reference[rows]))
    deltas = np.asarray(deltas)
    return {
        "delta_r2": r2(targets, candidate) - r2(targets, reference),
        "ci95": np.nanpercentile(deltas, [2.5, 97.5]).tolist(),
        "p_positive": float(np.mean(deltas > 0)),
    }


def _groups_r2(targets, predictions, groups, *, draws, seed) -> dict:
    return {
        name: {"r2": r2(targets[:, lo:hi], predictions[:, lo:hi]),
               "ci95": bootstrap_r2(targets[:, lo:hi], predictions[:, lo:hi], groups, draws=draws, seed=seed)}
        for name, (lo, hi) in TARGET_GROUPS.items()
    }


def analyze(samples: Samples, encoded: Mapping[str, dict], *, cameras, folds, inner_folds, draws, seed) -> dict:
    """Probe every checkpoint and camera set, then compare each checkpoint to the first."""
    labels = list(encoded)
    camera_sets = {camera: (camera,) for camera in cameras}
    if len(cameras) > 1:
        camera_sets["all_cameras"] = tuple(cameras)
    targets, groups = samples.targets, samples.episodes
    predictions: dict[str, dict[str, np.ndarray]] = {}
    report = {"raw_maps": {}, "paired_vs_reference": {}, "readout": {}}
    for label in labels:
        predictions[label] = {}
        report["raw_maps"][label] = {}
        for set_name, members in camera_sets.items():
            features = np.concatenate([encoded[label]["features"][camera] for camera in members], axis=1)
            predictions[label][set_name] = ridge_oof_predictions(
                features, targets, groups, folds=folds, inner_folds=inner_folds, seed=seed)
            report["raw_maps"][label][set_name] = _groups_r2(
                targets, predictions[label][set_name], groups, draws=draws, seed=seed)
        stats = encoded[label].get("camera_stats")
        if stats is not None:
            closed = targets[:, 3] > 0.5
            tokens = stats["token"].astype(np.float64)
            spread = max(float(np.sqrt(((tokens - tokens.mean(axis=0)) ** 2).sum(axis=1).mean())), 1e-12)
            token_predictions = ridge_oof_predictions(stats["token"], targets, groups, folds=folds,
                                                      inner_folds=inner_folds, seed=seed)
            readout = {
                "win_share": dict(zip(cameras, stats["win_share"].mean(axis=0).tolist())),
                "win_share_gripper_closed": dict(zip(cameras, stats["win_share"][closed].mean(axis=0).tolist()))
                if closed.any() else None,
                "win_share_gripper_open": dict(zip(cameras, stats["win_share"][~closed].mean(axis=0).tolist()))
                if (~closed).any() else None,
                "token_spread": spread,
                "single_camera_token_distance": dict(zip(cameras, (stats["single_distance"].mean(axis=0) / spread).tolist())),
                "policy_token_probe": _groups_r2(targets, token_predictions, groups, draws=draws, seed=seed),
            }
            if "drop_distance" in stats:
                readout["token_change_without_camera"] = dict(zip(cameras, (stats["drop_distance"].mean(axis=0) / spread).tolist()))
            report["readout"][label] = readout
    reference = labels[0]
    for label in labels[1:]:
        report["paired_vs_reference"][label] = {
            set_name: {
                name: paired_r2_difference(targets[:, lo:hi], predictions[reference][set_name][:, lo:hi],
                                           predictions[label][set_name][:, lo:hi], groups, draws=draws, seed=seed)
                for name, (lo, hi) in TARGET_GROUPS.items()
            }
            for set_name in camera_sets
        }
    report["reference"] = reference
    return report


def markdown_summary(report: Mapping) -> str:
    reference = report["reference"]
    lines = ["# DROID multi-camera probe", "",
             f"Reference: `{reference}`. Frames: {report['samples']['frames']} from "
             f"{report['samples']['episodes']} held-out episodes ({report['samples']['split']}).", "",
             "## Probe R^2 on raw us3/us6/us8 (all cameras)", "",
             "| checkpoint | " + " | ".join(TARGET_GROUPS) + " |", "|---" * (len(TARGET_GROUPS) + 1) + "|"]
    for label, sets in report["raw_maps"].items():
        values = sets.get("all_cameras") or next(iter(sets.values()))
        lines.append(f"| {label} | " + " | ".join(
            f"{values[name]['r2']:.3f} [{values[name]['ci95'][0]:.3f}, {values[name]['ci95'][1]:.3f}]"
            for name in TARGET_GROUPS) + " |")
    for label, sets in report["paired_vs_reference"].items():
        lines += ["", f"## {label} minus {reference}: ΔR^2 [95% CI] (P>0)", "",
                  "| camera set | " + " | ".join(TARGET_GROUPS) + " |", "|---" * (len(TARGET_GROUPS) + 1) + "|"]
        for set_name, targets in sets.items():
            lines.append(f"| {set_name} | " + " | ".join(
                f"{targets[name]['delta_r2']:+.3f} [{targets[name]['ci95'][0]:+.3f}, {targets[name]['ci95'][1]:+.3f}] "
                f"({targets[name]['p_positive']:.2f})" for name in TARGET_GROUPS) + " |")
    for label, readout in report["readout"].items():
        lines += ["", f"## Stage-I readout camera use: {label}", "",
                  "Token distances are in units of the token's spread across frames.", "",
                  "| camera | win share | gripper closed | gripper open | token change without it | single-camera token distance |",
                  "|---|---|---|---|---|---|"]
        for camera, share in readout["win_share"].items():
            closed = (readout.get("win_share_gripper_closed") or {}).get(camera)
            opened = (readout.get("win_share_gripper_open") or {}).get(camera)
            drop = (readout.get("token_change_without_camera") or {}).get(camera)
            fmt = lambda value: "n/a" if value is None else f"{value:.3f}"  # noqa: E731
            lines.append(f"| {camera} | {share:.3f} | {fmt(closed)} | {fmt(opened)} | {fmt(drop)} | "
                         f"{readout['single_camera_token_distance'][camera]:.3f} |")
        lines.append("")
        lines.append("Policy-token probe R^2: " + ", ".join(
            f"{name} {value['r2']:.3f}" for name, value in readout["policy_token_probe"].items()))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def _parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    checkpoints = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"--checkpoint expects LABEL=PATH, got {value!r}")
        if label in checkpoints:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        checkpoints[label] = Path(path).expanduser().resolve(strict=True)
    return checkpoints


def main(argv: Sequence[str] | None = None, *, extractor_factory: Callable | None = None,
         text_encoder_factory: Callable | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH of a Stage-I Student directory; the first is the reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, help="frame cache (.npz); read if present, else written")
    parser.add_argument("--data-root", default=os.environ.get("ROBOT_DIFT_DATA_ROOT"),
                        help="TFDS data dir containing droid/ (default: ROBOT_DIFT_DATA_ROOT)")
    parser.add_argument("--dataset", default="droid")
    parser.add_argument("--split", choices=tuple(SPLIT_PATTERNS), default="failure")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8, help="future offset in steps (15 Hz)")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument("--model-repo", default=os.environ.get("ROBOT_DIFT_MODEL_DIR"))
    parser.add_argument("--clip-model", default=os.environ.get("ROBOT_DIFT_CLIP_MODEL"),
                        help="CLIP ViT-B/32 file for the readout queries; readout analysis is skipped without it")
    parser.add_argument("--prompt", choices=("instruction", "empty"), default="instruction",
                        help="Student text conditioning; 'empty' removes the instruction")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    checkpoints = _parse_checkpoints(args.checkpoint)
    cameras = tuple(args.cameras)
    if args.samples is not None and args.samples.is_file():
        samples = Samples.load(args.samples)
        missing = [camera for camera in cameras if camera not in samples.images]
        if missing:
            parser.error(f"--samples lacks cameras {missing}")
    else:
        if not args.data_root:
            parser.error("--data-root (or ROBOT_DIFT_DATA_ROOT) is required without a --samples cache")
        samples = samples_from_episodes(
            read_droid_episodes(args.data_root, args.dataset, args.split, cameras, args.episodes),
            cameras=cameras, per_episode=args.frames_per_episode, horizon=args.horizon,
        )
        if args.samples is not None:
            samples.save(args.samples)

    if extractor_factory is None:
        from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor

        def extractor_factory(path):
            return RobotDIFTStudentFeatureExtractor(str(path), model_repo=args.model_repo, device=args.device)
    text_encoder = None
    if text_encoder_factory is not None:
        text_encoder = text_encoder_factory()
    elif args.clip_model:
        from agents.encoders.frozen_clip_text import FrozenCLIPTextTokens

        text_encoder = FrozenCLIPTextTokens(args.clip_model).to(args.device).eval()

    encoded, readout_notes = {}, {}
    for label, path in checkpoints.items():
        extractor = extractor_factory(path)
        readout, reason = (None, "no CLIP model given") if text_encoder is None else load_paper_readout(
            path, extractor.feature_dims)
        if readout is not None:
            readout = readout.to(args.device)
        readout_notes[label] = reason or "analyzed"
        encoded[label] = encode_checkpoint(
            samples, extractor, cameras=cameras, image_size=args.image_size, grid=args.grid,
            batch_size=args.batch_size, device=args.device, prompt_mode=args.prompt,
            readout=readout, text_encoder=text_encoder,
        )
        del extractor, readout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = analyze(samples, encoded, cameras=cameras, folds=args.folds, inner_folds=args.inner_folds,
                     draws=args.bootstrap, seed=args.seed)
    report["samples"] = {"episodes": len(samples.episode_names), "frames": len(samples), "split": args.split,
                         "horizon": args.horizon, "cameras": list(cameras)}
    report["protocol"] = {"image_size": args.image_size, "grid": args.grid, "prompt": args.prompt,
                          "folds": args.folds, "inner_folds": args.inner_folds, "bootstrap": args.bootstrap,
                          "seed": args.seed, "targets": {name: list(span) for name, span in TARGET_GROUPS.items()}}
    report["checkpoints"] = {label: str(path) for label, path in checkpoints.items()}
    report["readout_analysis"] = readout_notes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = markdown_summary(report)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
