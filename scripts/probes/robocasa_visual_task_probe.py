#!/usr/bin/env python3
"""Small frozen-feature probe for RoboCasa task/scene separability.

The 24-task/4-demo image set was selected for a spatial descriptor diagnostic.
Here each held-out demo image is ranked against prototypes built from the
other three demos of each task. All Student calls use an empty prompt so the
task label cannot leak through the text conditioner. This is a visual proxy,
not a test of language grounding, contact geometry, or robot success.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor


FEATURE_KEYS = ("us3", "us6", "us8")


def load_fixed_images(image_dir: Path) -> list[dict]:
    manifest = json.loads((image_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["images"]
    if len(rows) != 96 or len({row["task"] for row in rows}) != 24:
        raise ValueError("Expected the fixed 96-image, 24-task RoboCasa set")
    if any(Counter(row["demo"] for row in rows if row["task"] == task)
           != Counter({f"demo_{index}": 1 for index in range(1, 5)})
           for task in {row["task"] for row in rows}):
        raise ValueError("Each task must have exactly one image from each of four demos")
    for row in rows:
        path = image_dir / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image digest differs from fixed manifest: {path}")
    return sorted(rows, key=lambda row: (row["task"], row["demo"]))


def rank_heldout_demos(descriptors: torch.Tensor, tasks: list[str], demos: list[str]) -> dict:
    """Rank cosine similarity to each task prototype, holding one demo out."""
    if descriptors.ndim != 2 or descriptors.shape[0] != len(tasks) or len(tasks) != len(demos):
        raise ValueError("Descriptors, tasks, and demos must have matching sample counts")
    classes = sorted(set(tasks))
    folds = sorted(set(demos))
    if len(classes) < 2 or len(folds) < 2:
        raise ValueError("Need multiple classes and demonstration folds")
    descriptors = F.normalize(descriptors.float(), dim=1)
    ranks, margins = [], []
    by_fold = {}
    for fold in folds:
        train_indices = [index for index, demo in enumerate(demos) if demo != fold]
        test_indices = [index for index, demo in enumerate(demos) if demo == fold]
        prototypes = []
        for task in classes:
            matching = [index for index in train_indices if tasks[index] == task]
            if not matching:
                raise ValueError(f"Missing training image for task {task} in fold {fold}")
            prototypes.append(F.normalize(descriptors[matching].mean(dim=0), dim=0))
        scores = descriptors[test_indices] @ torch.stack(prototypes).T
        fold_ranks = []
        for row_index, source_index in enumerate(test_indices):
            target = classes.index(tasks[source_index])
            row_scores = scores[row_index]
            # Stable rank even when scores tie exactly.
            rank = 1 + sum(
                float(row_scores[index]) > float(row_scores[target])
                or (float(row_scores[index]) == float(row_scores[target]) and index < target)
                for index in range(len(classes)) if index != target
            )
            fold_ranks.append(rank)
            margins.append(float(row_scores[target] - row_scores[[i for i in range(len(classes)) if i != target]].max()))
        ranks.extend(fold_ranks)
        by_fold[fold] = {"top1": sum(rank == 1 for rank in fold_ranks) / len(fold_ranks),
                         "top5": sum(rank <= 5 for rank in fold_ranks) / len(fold_ranks)}
    return {
        "samples": len(ranks),
        "classes": len(classes),
        "chance_top1": 1.0 / len(classes),
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_true_minus_best_other_cosine": sum(margins) / len(margins),
        "by_demo_fold": by_fold,
    }


def extract_rgb_baseline(image_dir: Path, rows: list[dict]) -> torch.Tensor:
    """Cheap color/layout control to expose task-specific scene shortcuts."""
    descriptors = []
    for row in rows:
        with Image.open(image_dir / row["image"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
        descriptors.append(F.interpolate(tensor, size=(8, 8), mode="bilinear", align_corners=False).flatten())
    return torch.stack(descriptors)


@torch.inference_mode()
def extract_descriptors(model: RobotDIFTStudentFeatureExtractor, image_dir: Path,
                        rows: list[dict], batch_size: int) -> dict[str, torch.Tensor]:
    result = {key: [] for key in FEATURE_KEYS}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        arrays = []
        for row in batch_rows:
            with Image.open(image_dir / row["image"]) as image:
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0)
        images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(model.student_timestep.device)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        maps = model._encode_backbone(images * 2.0 - 1.0, [""] * len(batch_rows))
        for key in FEATURE_KEYS:
            pooled = maps[key].float().mean(dim=(-2, -1))
            if not torch.isfinite(pooled).all():
                raise ValueError(f"Nonfinite {key} descriptor")
            result[key].append(F.normalize(pooled, dim=1).cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows = load_fixed_images(args.image_dir)
    tasks = [row["task"] for row in rows]
    demos = [row["demo"] for row in rows]
    report = {
        "scope": "Exploratory visual task/scene separability only; no label text is supplied to the Student",
        "image_manifest": str((args.image_dir / "manifest.json").resolve()),
        "image_manifest_sha256": hashlib.sha256((args.image_dir / "manifest.json").read_bytes()).hexdigest(),
        "prompt": "",
        "pooling": "mean of raw spatial map, then L2 normalize",
        "protocol": "four leave-one-demo-out folds; nearest cosine task prototype",
        "rgb_8x8_control": rank_heldout_demos(extract_rgb_baseline(args.image_dir, rows), tasks, demos),
        "checkpoint_results": {},
    }
    for checkpoint in args.checkpoint:
        model = RobotDIFTStudentFeatureExtractor(
            str(checkpoint), model_repo=args.model_repo, device=args.device,
            feature_keys=FEATURE_KEYS,
        )
        descriptors = extract_descriptors(model, args.image_dir, rows, args.batch_size)
        report["checkpoint_results"][str(checkpoint.resolve())] = {
            "vae_latent_mode": model.vae_latent_mode,
            "maps": {key: rank_heldout_demos(descriptors[key], tasks, demos) for key in FEATURE_KEYS},
        }
        del model, descriptors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
