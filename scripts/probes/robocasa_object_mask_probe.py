#!/usr/bin/env python3
"""Leave-one-demo-out target-object segmentation from frozen Student maps.

Simulator geom masks provide an object label in recorded RoboCasa eye-camera
frames. Foreground and background feature prototypes come only from other
demos. The default empty prompt probes visual object separation; an optional
fixed prompt tests conditioning sensitivity. Neither proves instruction
grounding or policy success. Location and RGB controls expose shortcuts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor
from scripts.probes.robocasa_contact_point_probe import FEATURE_KEYS, extract_maps


def read_rows(image_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((image_dir / "manifest.json").read_text())
    rows = []
    for row in manifest["rows"]:
        mask_name = row.get("object_mask")
        if not row.get("in_frame") or not mask_name or (row.get("object_mask_pixels") or 0) < 32:
            continue
        for name, digest in ((row["image"], row["sha256"]),
                             (mask_name, row["object_mask_sha256"])):
            if hashlib.sha256((image_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed frame or mask: {image_dir / name}")
        rows.append(row)
    if len(rows) < 8 or len({row["camera"] for row in rows}) != 1:
        raise ValueError("Need at least eight labeled rows from one camera")
    return manifest, sorted(rows, key=lambda row: row["demo"])


def mask_labels(image_dir: Path, rows: list[dict], grid_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    labels, soft_masks, fallback_demos = [], [], []
    for row in rows:
        with Image.open(image_dir / row["object_mask"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        soft = F.interpolate(torch.from_numpy(mask)[None, None], size=(grid_size, grid_size),
                             mode="area")[0, 0].flatten()
        label = soft >= 0.5
        if not label.any():
            if not bool((soft > 0).any()):
                raise ValueError(f"Visible mask vanished at {grid_size}x{grid_size}: {row['demo']}")
            # Tiny objects can cover less than half of every coarse token.
            # Retain the one cell with maximal overlap and report the fallback.
            label[soft.argmax()] = True
            fallback_demos.append(row["demo"])
        if label.all():
            raise ValueError(f"Mask fills the entire {grid_size}x{grid_size} grid: {row['demo']}")
        labels.append(label)
        soft_masks.append(soft)
    return labels, soft_masks, fallback_demos


def rgb_patch_maps(image_dir: Path, rows: list[dict], grid_size: int) -> list[torch.Tensor]:
    maps = []
    for row in rows:
        with Image.open(image_dir / row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        small = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1)[None],
                              size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        patches = F.unfold(F.pad(small, (1, 1, 1, 1), mode="replicate"), kernel_size=3)
        maps.append(patches.reshape(27, grid_size, grid_size))
    return maps


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def summarize(scores_by_demo: list[np.ndarray], labels: list[torch.Tensor], rows: list[dict]) -> dict:
    details = []
    for scores, truth, row in zip(scores_by_demo, labels, rows):
        y = truth.numpy()
        details.append({
            "demo": row["demo"],
            "average_precision": average_precision(scores, y),
            "roc_auc": roc_auc(scores, y),
            "foreground_fraction": float(y.mean()),
        })
    return {
        "demos": len(details),
        "mean_average_precision": float(np.mean([item["average_precision"] for item in details])),
        "mean_roc_auc": float(np.mean([item["roc_auc"] for item in details])),
        "mean_foreground_fraction": float(np.mean([item["foreground_fraction"] for item in details])),
        "by_demo": details,
    }


def evaluate_prototypes(maps: list[torch.Tensor], labels: list[torch.Tensor],
                        rows: list[dict]) -> dict:
    tokens = [F.normalize(value.float().flatten(1).T, dim=1) for value in maps]
    scores_by_demo = []
    for heldout in range(len(rows)):
        foreground, background = [], []
        for index, (features, truth) in enumerate(zip(tokens, labels)):
            if index == heldout:
                continue
            foreground.append(F.normalize(features[truth].mean(0), dim=0))
            background.append(F.normalize(features[~truth].mean(0), dim=0))
        positive = F.normalize(torch.stack(foreground).mean(0), dim=0)
        negative = F.normalize(torch.stack(background).mean(0), dim=0)
        score = tokens[heldout] @ positive - tokens[heldout] @ negative
        scores_by_demo.append(score.numpy())
    return summarize(scores_by_demo, labels, rows)


def evaluate_position_prior(soft_masks: list[torch.Tensor], labels: list[torch.Tensor],
                            rows: list[dict]) -> dict:
    scores_by_demo = [torch.stack([mask for index, mask in enumerate(soft_masks)
                                   if index != heldout]).mean(0).numpy()
                      for heldout in range(len(rows))]
    return summarize(scores_by_demo, labels, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", default="", help="One fixed Student conditioning prompt for every frame")
    parser.add_argument("--prompt-mode", choices=("fixed", "recorded", "shifted"), default="fixed",
                        help="Use one fixed prompt, each demo's instruction, or a different demo's instruction")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    manifest, rows = read_rows(args.image_dir)
    if args.prompt_mode == "fixed":
        prompts: str | list[str] = args.prompt
    else:
        if args.prompt:
            parser.error("--prompt and --prompt-mode other than fixed cannot be combined")
        instructions = [str(row.get("instruction") or "").strip() for row in rows]
        if not all(instructions):
            parser.error("Every selected manifest row needs a nonempty recorded instruction")
        if args.prompt_mode == "recorded":
            prompts = instructions
        else:
            if len(set(instructions)) < 2:
                parser.error("Shifted prompts require at least two distinct instructions")
            prompts = [next(instructions[(index + offset) % len(rows)]
                            for offset in range(1, len(rows))
                            if instructions[(index + offset) % len(rows)] != instruction)
                       for index, instruction in enumerate(instructions)]
    labels = {}
    report = {
        "scope": "One-task target-object segmentation; fixed prompt and controlled scene",
        "manifest": str((args.image_dir / "manifest.json").resolve()),
        "manifest_sha256": hashlib.sha256((args.image_dir / "manifest.json").read_bytes()).hexdigest(),
        "valid_rows": len(rows),
        "prompt": args.prompt if args.prompt_mode == "fixed" else f"<manifest:{args.prompt_mode}>",
        "prompt_mode": args.prompt_mode,
        "prompts_by_demo": ({row["demo"]: prompt for row, prompt in zip(rows, prompts)}
                            if not isinstance(prompts, str) else None),
        "controls": {},
        "label_protocol": "foreground if area overlap >= 0.5; otherwise one maximum-overlap token",
        "low_occupancy_fallback_demos": {},
        "checkpoint_results": {},
    }
    for size in (8, 16):
        truth, soft, fallback_demos = mask_labels(args.image_dir, rows, size)
        labels[size] = truth
        report["low_occupancy_fallback_demos"][f"{size}x{size}"] = fallback_demos
        report["controls"][f"{size}x{size}"] = {
            "position_prior": evaluate_position_prior(soft, truth, rows),
            "rgb_3x3_patch": evaluate_prototypes(rgb_patch_maps(args.image_dir, rows, size), truth, rows),
        }
    for checkpoint in args.checkpoint:
        model = RobotDIFTStudentFeatureExtractor(str(checkpoint), model_repo=args.model_repo,
                                                device=args.device, feature_keys=FEATURE_KEYS)
        maps = extract_maps(model, args.image_dir, rows, args.batch_size, prompt=prompts)
        report["checkpoint_results"][str(checkpoint.resolve())] = {
            "vae_latent_mode": model.vae_latent_mode,
            "maps": {key: evaluate_prototypes(maps[key], labels[maps[key][0].shape[-1]], rows)
                     for key in FEATURE_KEYS},
        }
        del model, maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "valid_rows": report["valid_rows"],
        "controls": {size: {name: {key: value[key] for key in
                                     ("mean_average_precision", "mean_roc_auc", "mean_foreground_fraction")}
                             for name, value in controls.items()}
                     for size, controls in report["controls"].items()},
        "checkpoint_results": {
            name: {key: {metric: value[metric] for metric in
                         ("mean_average_precision", "mean_roc_auc", "mean_foreground_fraction")}
                   for key, value in result["maps"].items()}
            for name, result in report["checkpoint_results"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
