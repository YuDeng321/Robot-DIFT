#!/usr/bin/env python3
"""Check whether a frozen Student changes object localization with the noun.

Each RoboCasa frame has two visible, simulator-labeled objects. The two
instructions differ only in the named object. A leave-one-demo-out prototype
scores the named versus other object's spatial tokens. A prompt-blind map has
zero paired switch margin by construction. This is a diagnostic of frozen
features; it does not measure policy success or prove language grounding.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor
from scripts.probes.robocasa_contact_point_probe import FEATURE_KEYS, extract_maps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(image_dir: Path) -> tuple[dict, list[dict]]:
    manifest_path = image_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("contrast_geom_prefix"):
        raise ValueError("Manifest does not contain same-image contrast masks")
    rows = []
    for row in manifest["rows"]:
        if min(row["object_mask_pixels"], row["contrast_object_mask_pixels"]) < 32:
            continue
        if not row["in_frame"] or not row["contrast_in_frame"]:
            continue
        if not row.get("instruction") or not row.get("contrast_instruction"):
            raise ValueError(f"Missing paired instructions for {row['demo']}")
        if row["instruction"] == row["contrast_instruction"]:
            raise ValueError(f"Identical paired instructions for {row['demo']}")
        for name_key, digest_key in (
            ("image", "sha256"),
            ("object_mask", "object_mask_sha256"),
            ("contrast_object_mask", "contrast_object_mask_sha256"),
        ):
            if sha256(image_dir / row[name_key]) != row[digest_key]:
                raise ValueError(f"Changed input {row[name_key]}")
        rows.append(row)
    rows.sort(key=lambda row: int(row["demo"].split("_")[-1]))
    if len(rows) < 8 or len({row["camera"] for row in rows}) != 1:
        raise ValueError("Need at least eight dual-visible demos from one camera")
    return manifest, rows


def area_weights(image_dir: Path, rows: list[dict], key: str, size: int) -> list[torch.Tensor]:
    weights = []
    for row in rows:
        with Image.open(image_dir / row[key]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        small = F.interpolate(torch.from_numpy(mask)[None, None], size=(size, size), mode="area")
        weight = small.flatten().float()
        if weight.sum() <= 0:
            raise ValueError(f"Visible mask vanished on {size}x{size} grid: {row['demo']}")
        weights.append(weight / weight.sum())
    return weights


def mean_score(scores: torch.Tensor, weight: torch.Tensor) -> float:
    return float((scores * weight).sum())


def evaluate_map_pair(maps: list[torch.Tensor], target: list[torch.Tensor],
                      contrast: list[torch.Tensor], rows: list[dict]) -> dict:
    if len(maps) != 2 * len(rows):
        raise ValueError("Need two prompt-conditioned maps per labeled demo")
    tokens = [F.normalize(value.float().flatten(1).T, dim=-1, eps=1e-6) for value in maps]
    named, other = [], []
    for index in range(len(rows)):
        target_tokens, contrast_tokens = tokens[2 * index:2 * index + 2]
        named.append((F.normalize((target_tokens * target[index][:, None]).sum(0), dim=0),
                      F.normalize((contrast_tokens * contrast[index][:, None]).sum(0), dim=0)))
        other.append((F.normalize((target_tokens * contrast[index][:, None]).sum(0), dim=0),
                      F.normalize((contrast_tokens * target[index][:, None]).sum(0), dim=0)))
    details = []
    for heldout, row in enumerate(rows):
        positive = F.normalize(torch.stack([named[j][p] for j in range(len(rows))
                                            if j != heldout for p in range(2)]).mean(0), dim=0)
        negative = F.normalize(torch.stack([other[j][p] for j in range(len(rows))
                                            if j != heldout for p in range(2)]).mean(0), dim=0)
        score_target = tokens[2 * heldout] @ positive - tokens[2 * heldout] @ negative
        score_contrast = tokens[2 * heldout + 1] @ positive - tokens[2 * heldout + 1] @ negative
        margin_target = (mean_score(score_target, target[heldout])
                         - mean_score(score_target, contrast[heldout]))
        margin_contrast = (mean_score(score_contrast, contrast[heldout])
                           - mean_score(score_contrast, target[heldout]))
        prompt_cosine = float((tokens[2 * heldout] * tokens[2 * heldout + 1]).sum(-1).mean())
        details.append({
            "demo": row["demo"],
            "target_category": row["target_category"],
            "contrast_category": row["contrast_category"],
            "target_prompt_margin": margin_target,
            "contrast_prompt_margin": margin_contrast,
            "paired_switch_margin": (margin_target + margin_contrast) / 2,
            "both_prompts_correct": bool(margin_target > 0 and margin_contrast > 0),
            "mean_spatial_prompt_cosine": prompt_cosine,
        })
    return {
        "mean_paired_switch_margin": float(np.mean([row["paired_switch_margin"] for row in details])),
        "both_prompts_correct_fraction": float(np.mean([row["both_prompts_correct"] for row in details])),
        "mean_spatial_prompt_cosine": float(np.mean([row["mean_spatial_prompt_cosine"] for row in details])),
        "by_demo": details,
    }


def paired_interval(reference: list[dict], candidate: list[dict], seed: int = 0) -> dict:
    if [row["demo"] for row in reference] != [row["demo"] for row in candidate]:
        raise ValueError("Checkpoint evaluations use different demos")
    difference = np.asarray([b["paired_switch_margin"] - a["paired_switch_margin"]
                             for a, b in zip(reference, candidate)], dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.integers(len(difference), size=(10000, len(difference)))
    limits = np.quantile(difference[samples].mean(axis=1), [0.025, 0.975])
    return {"mean_change": float(difference.mean()),
            "paired_demo_bootstrap_95pct": [float(limits[0]), float(limits[1])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    manifest, rows = load_rows(args.image_dir)
    dual_rows = [row for row in rows for _ in range(2)]
    prompts = [prompt for row in rows for prompt in
               (row["instruction"], row["contrast_instruction"])]
    report = {
        "scope": "Same-image two-object noun switch on frozen Student maps",
        "manifest": str((args.image_dir / "manifest.json").resolve()),
        "manifest_sha256": sha256(args.image_dir / "manifest.json"),
        "source_file": manifest["source_file"],
        "valid_dual_visible_demos": len(rows),
        "protocol": "Leave-one-demo-out named/other object prototypes from both prompts;"
                    " area-weighted object scores; image and action wording fixed within pair",
        "prompt_blind_paired_switch_margin": 0.0,
        "checkpoint_results": {},
        "paired_checkpoint_changes_from_first": {},
    }
    weight_cache = {}
    for checkpoint in args.checkpoint:
        checkpoint_path = str(checkpoint.resolve())
        model = RobotDIFTStudentFeatureExtractor(checkpoint_path, model_repo=args.model_repo,
                                                device=args.device, feature_keys=FEATURE_KEYS)
        with torch.inference_mode():
            maps = extract_maps(model, args.image_dir, dual_rows, args.batch_size, prompt=prompts)
        result = {"vae_latent_mode": model.vae_latent_mode, "maps": {}}
        for key in FEATURE_KEYS:
            size = maps[key][0].shape[-1]
            if size not in weight_cache:
                weight_cache[size] = (
                    area_weights(args.image_dir, rows, "object_mask", size),
                    area_weights(args.image_dir, rows, "contrast_object_mask", size),
                )
            result["maps"][key] = evaluate_map_pair(maps[key], *weight_cache[size], rows)
        report["checkpoint_results"][checkpoint_path] = result
        del model, maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    first = str(args.checkpoint[0].resolve())
    for checkpoint in args.checkpoint[1:]:
        name = str(checkpoint.resolve())
        report["paired_checkpoint_changes_from_first"][name] = {
            key: paired_interval(report["checkpoint_results"][first]["maps"][key]["by_demo"],
                                 report["checkpoint_results"][name]["maps"][key]["by_demo"])
            for key in FEATURE_KEYS
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid_dual_visible_demos": len(rows),
        "checkpoint_results": {name: {key: {k: value[k] for k in
                                               ("mean_paired_switch_margin",
                                                "both_prompts_correct_fraction",
                                                "mean_spatial_prompt_cosine")}
                                       for key, value in result["maps"].items()}
                               for name, result in report["checkpoint_results"].items()},
        "paired_checkpoint_changes_from_first": report["paired_checkpoint_changes_from_first"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
