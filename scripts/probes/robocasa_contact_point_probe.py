#!/usr/bin/env python3
"""Cross-demo contact-point matching on saved RoboCasa keypoint projections.

The default empty prompt measures visual descriptors without instruction
leakage; an optional fixed prompt probes conditioning sensitivity. A projected
collision-geom center is a proxy for the contact point. This test does not
validate language grounding or downstream policy performance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor


FEATURE_KEYS = ("us3", "us6", "us8")


def load_rows(image_dir: Path, max_surface_gap: float,
              min_front_score: float) -> tuple[dict, list[dict]]:
    manifest_path = image_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for row in manifest["rows"]:
        path = image_dir / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Frame changed after manifest creation: {path}")
        gap = row["target_minus_surface_m"]
        if (row["in_frame"] and gap is not None and abs(gap) <= max_surface_gap
                and row.get("surface_same_fixture") is True
                and row.get("button_front_facing_score") is not None
                and row["button_front_facing_score"] >= min_front_score):
            rows.append(row)
    counts = Counter(row["camera"] for row in rows)
    if any(counts[camera] < 4 for camera in manifest["cameras"]):
        raise ValueError(f"Need at least four visible keypoints per camera: {dict(counts)}")
    return manifest, sorted(rows, key=lambda row: (row["camera"], row["demo"]))


def sample_point(feature: torch.Tensor, xy: tuple[float, float], size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    x, y = xy
    grid = torch.tensor([[[[(x + 0.5) * 2.0 / width - 1.0,
                            (y + 0.5) * 2.0 / height - 1.0]]]],
                        device=feature.device, dtype=feature.dtype)
    value = F.grid_sample(feature[None], grid, mode="bilinear",
                          padding_mode="border", align_corners=False)[0, :, 0, 0]
    return F.normalize(value.float(), dim=0)


def nearest_point(source: torch.Tensor, target: torch.Tensor,
                  source_xy: tuple[float, float], size: tuple[int, int]) -> tuple[float, float]:
    query = sample_point(source, source_xy, size)
    height, width = size
    _, map_height, map_width = target.shape
    tokens = F.normalize(target.float().flatten(1).T, dim=1)
    best = int(torch.argmax(tokens @ query))
    y, x = divmod(best, map_width)
    return ((x + 0.5) * width / map_width - 0.5,
            (y + 0.5) * height / map_height - 0.5)


def summarize(pairs: list[dict]) -> dict:
    errors = np.asarray([pair["error_px"] for pair in pairs], dtype=np.float64)
    if len(errors) == 0:
        raise ValueError("No cross-demo pairs")
    return {
        "pairs": len(errors),
        "pck4": float(np.mean(errors <= 4)),
        "pck8": float(np.mean(errors <= 8)),
        "pck16": float(np.mean(errors <= 16)),
        "median_error_px": float(np.median(errors)),
        "mean_error_px": float(np.mean(errors)),
    }


def match_all(rows: list[dict], maps: dict[str, list[torch.Tensor]]) -> dict:
    report = {}
    for key, feature_maps in maps.items():
        pairs = []
        for source_index, source in enumerate(rows):
            source_xy = tuple(source["target_xy"])
            for target_index, target in enumerate(rows):
                if source["camera"] != target["camera"] or source["demo"] == target["demo"]:
                    continue
                size = tuple(target["image_size"])
                if tuple(source["image_size"]) != size:
                    raise ValueError("Mixed image sizes are unsupported in one camera")
                target_xy = tuple(target["target_xy"])
                predicted = nearest_point(feature_maps[source_index], feature_maps[target_index],
                                          source_xy, size)
                pairs.append({
                    "camera": source["camera"],
                    "source_demo": source["demo"],
                    "target_demo": target["demo"],
                    "error_px": float(np.hypot(predicted[0] - target_xy[0],
                                               predicted[1] - target_xy[1])),
                    "predicted_xy": list(predicted),
                    "target_xy": list(target_xy),
                })
        report[key] = {
            "all": summarize(pairs),
            "by_camera": {camera: summarize([pair for pair in pairs if pair["camera"] == camera])
                          for camera in sorted({pair["camera"] for pair in pairs})},
            "pairs": pairs,
        }
    return report


def position_baseline(rows: list[dict]) -> dict:
    pairs = []
    for source in rows:
        for target in rows:
            if source["camera"] != target["camera"] or source["demo"] == target["demo"]:
                continue
            source_xy = tuple(source["target_xy"])
            target_xy = tuple(target["target_xy"])
            pairs.append({
                "camera": source["camera"],
                "source_demo": source["demo"],
                "target_demo": target["demo"],
                "error_px": float(np.hypot(source_xy[0] - target_xy[0],
                                           source_xy[1] - target_xy[1])),
            })
    return {
        "all": summarize(pairs),
        "by_camera": {camera: summarize([pair for pair in pairs if pair["camera"] == camera])
                      for camera in sorted({pair["camera"] for pair in pairs})},
        "pairs": pairs,
    }


def rgb_patch_control(image_dir: Path, rows: list[dict], grid_size: int) -> dict:
    """Match local low-resolution RGB patches on the same dense grid."""
    feature_maps = []
    for row in rows:
        with Image.open(image_dir / row["image"]) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1)[None]
        small = F.interpolate(tensor, size=(grid_size, grid_size),
                              mode="bilinear", align_corners=False)
        padded = F.pad(small, (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(padded, kernel_size=3).reshape(27, grid_size, grid_size)
        feature_maps.append(patches)
    return match_all(rows, {f"rgb_{grid_size}x{grid_size}_3x3": feature_maps})


@torch.inference_mode()
def extract_maps(model: RobotDIFTStudentFeatureExtractor, image_dir: Path,
                 rows: list[dict], batch_size: int,
                 feature_keys: tuple[str, ...] = FEATURE_KEYS,
                 prompt: str | list[str] = "") -> dict[str, list[torch.Tensor]]:
    if not isinstance(prompt, str) and len(prompt) != len(rows):
        raise ValueError("Per-image prompt list must match the selected image rows")
    maps = {key: [] for key in feature_keys}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        frames = []
        for row in batch_rows:
            with Image.open(image_dir / row["image"]) as image:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0)
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(model.student_timestep.device)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        captions = [prompt] * len(batch_rows) if isinstance(prompt, str) else prompt[start:start + len(batch_rows)]
        batch_maps = model._encode_backbone(images * 2.0 - 1.0, captions)
        for key in feature_keys:
            value = batch_maps[key].float().cpu()
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} map")
            maps[key].extend(list(value))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-surface-gap", type=float, default=0.08)
    parser.add_argument("--min-front-score", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-us10", action="store_true",
                        help="Also report the non-deployed 32x32 us10 map as a geometry diagnostic")
    parser.add_argument("--prompt", default="", help="One fixed Student conditioning prompt for every frame")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_surface_gap <= 0 or not -1 <= args.min_front_score <= 1:
        parser.error("batch-size and max-surface-gap must be positive; min-front-score in [-1,1]")
    manifest, rows = load_rows(args.image_dir, args.max_surface_gap, args.min_front_score)
    feature_keys = FEATURE_KEYS + (("us10",) if args.include_us10 else ())
    report = {
        "scope": "Projected contact-point cross-demo descriptor diagnostic only",
        "manifest": str((args.image_dir / "manifest.json").resolve()),
        "manifest_sha256": hashlib.sha256((args.image_dir / "manifest.json").read_bytes()).hexdigest(),
        "source_file": manifest["source_file"],
        "valid_rows": len(rows),
        "visible_surface_gap_limit_m": args.max_surface_gap,
        "minimum_front_facing_score": args.min_front_score,
        "prompt": args.prompt,
        "feature_keys": list(feature_keys),
        "position_baseline": position_baseline(rows),
        "rgb_patch_controls": {
            "8x8": rgb_patch_control(args.image_dir, rows, 8),
            "16x16": rgb_patch_control(args.image_dir, rows, 16),
        },
        "checkpoint_results": {},
    }
    for checkpoint in args.checkpoint:
        model = RobotDIFTStudentFeatureExtractor(str(checkpoint), model_repo=args.model_repo,
                                                device=args.device, feature_keys=feature_keys)
        maps = extract_maps(model, args.image_dir, rows, args.batch_size, feature_keys,
                            args.prompt)
        report["checkpoint_results"][str(checkpoint.resolve())] = {
            "vae_latent_mode": model.vae_latent_mode,
            "maps": match_all(rows, maps),
        }
        del model, maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    brief = {
        "valid_rows": report["valid_rows"],
        "position_baseline": report["position_baseline"]["all"],
        "rgb_patch_controls": {
            name: next(iter(result.values()))["all"]
            for name, result in report["rgb_patch_controls"].items()
        },
        "checkpoint_results": {
            name: {key: value["all"] for key, value in result["maps"].items()}
            for name, result in report["checkpoint_results"].items()
        },
    }
    print(json.dumps(brief, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
