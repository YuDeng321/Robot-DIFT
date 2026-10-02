#!/usr/bin/env python3
"""Controlled dense-correspondence evaluation under known image warps.

The evaluator deliberately separates local discriminability from repeatability:
it queries a source feature at a physical/image point, retrieves its nearest
neighbour in a geometrically warped target image, and compares the retrieval to
the known target location.

Model integration is configuration-driven. A JSON model spec contains a Python
class target, constructor kwargs, and a dense extraction method. This keeps the
metric independent of Robot-DIFT/DINO checkpoint layout while allowing the
paper encoders to be evaluated without copying model code into this script.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class WarpSpec:
    name: str
    tx_px: float = 0.0
    ty_px: float = 0.0
    rotation_deg: float = 0.0
    brightness: float = 1.0


def _resolve_target(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def _replace_placeholders(value: Any, checkpoint: Optional[str], device: str) -> Any:
    if isinstance(value, str):
        resolved = value.replace("${checkpoint}", checkpoint or "").replace("${device}", device)
        if "${model_repo}" in resolved:
            model_repo = os.environ.get("ROBOT_DIFT_MODEL_DIR")
            if not model_repo:
                raise ValueError("${model_repo} requires ROBOT_DIFT_MODEL_DIR")
            resolved = resolved.replace("${model_repo}", model_repo)
        return resolved
    if isinstance(value, list):
        return [_replace_placeholders(item, checkpoint, device) for item in value]
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, checkpoint, device) for key, item in value.items()}
    return value


class DenseExtractor:
    """Normalize repository encoders to a BCHW dense-feature interface."""

    def __init__(
        self,
        name: str,
        model: Optional[torch.nn.Module],
        method: str,
        device: str,
        imagenet_norm: bool = False,
        normalization: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
        language: Optional[str] = None,
        feature_key: Optional[str] = None,
        expected_feature_size: Optional[Tuple[int, int]] = None,
        checkpoint: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.method = method
        self.device = torch.device(device)
        self.normalization = str(normalization or ("imagenet" if imagenet_norm else "none")).lower()
        if self.normalization not in {"none", "imagenet", "dift"}:
            raise ValueError(
                f"Unsupported normalization={self.normalization}; expected none, imagenet, or dift"
            )
        self.imagenet_norm = self.normalization == "imagenet"
        self.resize = resize
        self.language = language
        self.feature_key = feature_key
        self.expected_feature_size = expected_feature_size
        self.checkpoint = checkpoint
        self._observed_feature_size: Optional[Tuple[int, int]] = None

    @classmethod
    def from_spec(
        cls,
        spec_path: Optional[str],
        checkpoint: Optional[str],
        device: str,
    ) -> "DenseExtractor":
        if spec_path is None:
            return cls(name="rgb", model=None, method="rgb", device=device)
        with open(spec_path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        if bool(spec.get("requires_checkpoint", False)):
            if not checkpoint:
                raise ValueError(f"{spec_path} requires --checkpoint; refusing to run a public fallback")
            if not Path(checkpoint).expanduser().exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        kwargs = _replace_placeholders(spec.get("kwargs", {}), checkpoint, device)
        model_cls = _resolve_target(spec["target"])
        model = model_cls(**kwargs).to(device)
        model.eval()
        resize = spec.get("resize")
        return cls(
            name=str(spec.get("name", Path(spec_path).stem)),
            model=model,
            method=str(spec.get("method", "extract_feature_map")),
            device=device,
            imagenet_norm=bool(spec.get("imagenet_norm", False)),
            normalization=spec.get("normalization"),
            resize=tuple(resize) if resize else None,
            language=spec.get("language"),
            feature_key=spec.get("feature_key"),
            expected_feature_size=(
                tuple(spec["expected_feature_size"])
                if spec.get("expected_feature_size")
                else None
            ),
            checkpoint=str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
        )

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32)
        if self.resize:
            images = F.interpolate(images, size=self.resize, mode="bilinear", align_corners=False)
        if self.normalization == "imagenet":
            mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            images = (images - mean) / std
        elif self.normalization == "dift":
            images = (images - 0.5) / 0.5
        return images

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = self._prepare(images)
        if self.method == "rgb":
            features = images
        elif self.method == "extract_feature_map":
            features = self.model.extract_feature_map(images, lang_cond=self.language)
        elif self.method == "dino_fused":
            maps = self.model._extract_feature_maps(images)
            features = self.model.s2fpn_fusion(maps)
        elif self.method == "dino_map":
            maps = self.model._extract_feature_maps(images)
            feature_key = self.feature_key or self.model.feature_keys[-1]
            features = maps[feature_key]
        elif self.method == "cleandift_map":
            maps = self.model._encode_backbone(images, self.language)
            feature_key = self.feature_key or self.model.primary_feature_key
            features = maps[feature_key]
        elif self.method == "call":
            features = self.model(images)
            if isinstance(features, (tuple, list)):
                features = features[0]
        else:
            method = getattr(self.model, self.method)
            features = method(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            shape = getattr(features, "shape", None)
            raise RuntimeError(f"{self.name} did not return BCHW dense features; got {shape}")
        feature_size = (int(features.shape[-2]), int(features.shape[-1]))
        if self.expected_feature_size and feature_size != self.expected_feature_size:
            raise RuntimeError(
                f"{self.name} returned a {feature_size[0]}x{feature_size[1]} feature map; "
                f"expected {self.expected_feature_size[0]}x{self.expected_feature_size[1]}"
            )
        if self._observed_feature_size and feature_size != self._observed_feature_size:
            raise RuntimeError(
                f"{self.name} changed feature-map size from {self._observed_feature_size} "
                f"to {feature_size} within one evaluation"
            )
        self._observed_feature_size = feature_size
        return F.normalize(features.float(), dim=1, eps=1e-6)

    def protocol_metadata(self) -> Dict[str, Any]:
        metadata = {
            "checkpoint": self.checkpoint,
            "expected_feature_size": (
                list(self.expected_feature_size) if self.expected_feature_size else None
            ),
            "feature_key": self.feature_key,
            "normalization": self.normalization,
            "input_resize": list(self.resize) if self.resize else None,
            "language": self.language,
            "method": self.method,
            "observed_feature_size": (
                list(self._observed_feature_size) if self._observed_feature_size else None
            ),
        }
        model_metadata = getattr(self.model, "protocol_metadata", None)
        if callable(model_metadata):
            metadata["model_protocol"] = model_metadata()
        return metadata


def source_to_target_homography(
    height: int,
    width: int,
    tx_px: float,
    ty_px: float,
    rotation_deg: float,
    device: torch.device,
) -> torch.Tensor:
    angle = math.radians(rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    translate_to_origin = torch.tensor(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
        device=device,
    )
    rotate = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    translate_back = torch.tensor(
        [[1.0, 0.0, cx + tx_px], [0.0, 1.0, cy + ty_px], [0.0, 0.0, 1.0]],
        device=device,
    )
    return translate_back @ rotate @ translate_to_origin


def _pixel_to_normalized(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = 2.0 * points[..., 0] / max(width - 1, 1) - 1.0
    y = 2.0 * points[..., 1] / max(height - 1, 1) - 1.0
    return torch.stack((x, y), dim=-1)


def warp_images(images: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Warp source images into target coordinates using source->target H."""
    batch, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32),
        torch.arange(width, device=images.device, dtype=torch.float32),
        indexing="ij",
    )
    target = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(-1, 3).T
    source = torch.linalg.inv(homography) @ target
    source_xy = (source[:2] / source[2:].clamp_min(1e-8)).T.reshape(height, width, 2)
    grid = _pixel_to_normalized(source_xy, height, width)[None].expand(batch, -1, -1, -1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def transform_points(points_xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points_xy, torch.ones_like(points_xy[:, :1])), dim=1)
    transformed = (homography @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:].clamp_min(1e-8)


def sample_query_points(
    height: int,
    width: int,
    stride: int,
    border: int,
    device: torch.device,
) -> torch.Tensor:
    xs = torch.arange(border, width - border, stride, device=device, dtype=torch.float32)
    ys = torch.arange(border, height - border, stride, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=1)


def _image_to_feature_points(
    points_xy: torch.Tensor,
    image_hw: Tuple[int, int],
    feature_hw: Tuple[int, int],
) -> torch.Tensor:
    image_h, image_w = image_hw
    feature_h, feature_w = feature_hw
    x = points_xy[:, 0] * max(feature_w - 1, 1) / max(image_w - 1, 1)
    y = points_xy[:, 1] * max(feature_h - 1, 1) / max(image_h - 1, 1)
    return torch.stack((x, y), dim=1)


def _sample_features(feature_map: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    _, _, height, width = feature_map.shape
    grid = _pixel_to_normalized(points_xy, height, width)[None, None]
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
    return F.normalize(sampled[0, :, 0].T, dim=1, eps=1e-6)


def match_feature_maps(
    source_map: torch.Tensor,
    target_map: torch.Tensor,
    source_points_image: torch.Tensor,
    target_points_image: torch.Tensor,
    image_hw: Tuple[int, int],
    pck_thresholds: Sequence[float],
    exclusion_radius_px: float,
) -> Dict[str, float]:
    if source_map.shape[0] != 1 or target_map.shape[0] != 1:
        raise ValueError("match_feature_maps currently expects batch size one")
    source_hw, target_hw = source_map.shape[-2:], target_map.shape[-2:]
    source_points_feature = _image_to_feature_points(source_points_image, image_hw, source_hw)
    target_points_feature = _image_to_feature_points(target_points_image, image_hw, target_hw)
    queries = _sample_features(source_map, source_points_feature)
    target_tokens = F.normalize(target_map[0].flatten(1).T, dim=1, eps=1e-6)
    similarities = queries @ target_tokens.T
    best_indices = similarities.argmax(dim=1)
    target_h, target_w = target_hw
    pred_feature = torch.stack(
        ((best_indices % target_w).float(), (best_indices // target_w).float()), dim=1
    )
    pred_image = torch.stack(
        (
            pred_feature[:, 0] * max(image_hw[1] - 1, 1) / max(target_w - 1, 1),
            pred_feature[:, 1] * max(image_hw[0] - 1, 1) / max(target_h - 1, 1),
        ),
        dim=1,
    )
    errors = torch.linalg.vector_norm(pred_image - target_points_image, dim=1)

    true_features = _sample_features(target_map, target_points_feature)
    true_similarity = (queries * true_features).sum(dim=1)
    yy, xx = torch.meshgrid(
        torch.arange(target_h, device=target_map.device),
        torch.arange(target_w, device=target_map.device),
        indexing="ij",
    )
    token_image_xy = torch.stack(
        (
            xx.flatten() * max(image_hw[1] - 1, 1) / max(target_w - 1, 1),
            yy.flatten() * max(image_hw[0] - 1, 1) / max(target_h - 1, 1),
        ),
        dim=1,
    )
    distances_to_truth = torch.cdist(target_points_image, token_image_xy.float())
    incorrect_mask = distances_to_truth > exclusion_radius_px
    negative_similarity = similarities.masked_fill(~incorrect_mask, -torch.inf).max(dim=1).values
    valid_negative = torch.isfinite(negative_similarity)
    margins = true_similarity[valid_negative] - negative_similarity[valid_negative]

    result: Dict[str, float] = {
        "num_points": float(errors.numel()),
        "mean_error_px": float(errors.mean().item()),
        "median_error_px": float(errors.median().item()),
        "same_point_cosine": float(true_similarity.mean().item()),
        "negative_margin": float(margins.mean().item()) if margins.numel() else float("nan"),
    }
    for threshold in pck_thresholds:
        result[f"pck@{threshold:g}px"] = float((errors <= threshold).float().mean().item())
    return result


def _load_image(path: Path, image_size: Optional[Tuple[int, int]]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size:
            image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"model", "image", "warp"} and isinstance(value, (float, int))
    ]
    result: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(np.nanmean(values))
        result[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def _aggregate_image_level_with_ci(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, float]:
    """Aggregate by independent image, then bootstrap images (not query points)."""
    images = sorted({str(row["image"]) for row in rows})
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"model", "image", "warp", "num_points"}
        and isinstance(value, (float, int))
    ]
    per_image: Dict[str, Dict[str, float]] = {}
    for image_name in images:
        image_rows = [row for row in rows if str(row["image"]) == image_name]
        per_image[image_name] = {
            key: float(np.nanmean([float(row[key]) for row in image_rows]))
            for key in numeric_keys
        }
    matrix = np.asarray(
        [[per_image[image_name][key] for key in numeric_keys] for image_name in images],
        dtype=np.float64,
    )
    result: Dict[str, float] = {"num_images": float(len(images))}
    rng = np.random.default_rng(seed)
    if bootstrap_samples > 0 and len(images) > 1:
        indices = rng.integers(0, len(images), size=(bootstrap_samples, len(images)))
        bootstrap_means = matrix[indices].mean(axis=1)
    else:
        bootstrap_means = matrix.mean(axis=0, keepdims=True)
    for idx, key in enumerate(numeric_keys):
        values = matrix[:, idx]
        result[f"{key}_mean"] = float(np.nanmean(values))
        result[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"{key}_ci95_low"] = float(np.nanpercentile(bootstrap_means[:, idx], 2.5))
        result[f"{key}_ci95_high"] = float(np.nanpercentile(bootstrap_means[:, idx], 97.5))
    return result


def evaluate(
    extractor: DenseExtractor,
    image_paths: Sequence[Path],
    warps: Sequence[WarpSpec],
    device: str,
    query_stride: int,
    query_border: int,
    pck_thresholds: Sequence[float],
    exclusion_radius_px: float,
    image_size: Optional[Tuple[int, int]],
    match_feature_size: Optional[Tuple[int, int]],
    warp_batch_size: int,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    torch_device = torch.device(device)
    for image_path in image_paths:
        source = _load_image(image_path, image_size)[None].to(torch_device)
        _, _, height, width = source.shape
        source_map = extractor(source)
        if match_feature_size:
            source_map = F.interpolate(
                source_map, size=match_feature_size, mode="bilinear", align_corners=False
            )
        source_points = sample_query_points(
            height, width, query_stride, query_border, torch_device
        )
        for chunk_start in range(0, len(warps), max(1, warp_batch_size)):
            warp_chunk = warps[chunk_start : chunk_start + max(1, warp_batch_size)]
            homographies = [
                source_to_target_homography(
                    height,
                    width,
                    warp.tx_px,
                    warp.ty_px,
                    warp.rotation_deg,
                    torch_device,
                )
                for warp in warp_chunk
            ]
            targets = torch.cat(
                [
                    (warp_images(source, homography) * warp.brightness).clamp(0.0, 1.0)
                    for warp, homography in zip(warp_chunk, homographies)
                ],
                dim=0,
            )
            target_maps = extractor(targets)
            if match_feature_size:
                target_maps = F.interpolate(
                    target_maps, size=match_feature_size, mode="bilinear", align_corners=False
                )
            for index, (warp, homography) in enumerate(zip(warp_chunk, homographies)):
                target_points = transform_points(source_points, homography)
                valid = (
                    (target_points[:, 0] >= query_border)
                    & (target_points[:, 0] < width - query_border)
                    & (target_points[:, 1] >= query_border)
                    & (target_points[:, 1] < height - query_border)
                )
                metrics = match_feature_maps(
                    source_map,
                    target_maps[index : index + 1],
                    source_points[valid],
                    target_points[valid],
                    (height, width),
                    pck_thresholds,
                    exclusion_radius_px,
                )
                rows.append(
                    {
                        "model": extractor.name,
                        "image": image_path.name,
                        "warp": warp.name,
                        **metrics,
                    }
                )
    image_summary = _aggregate_image_level_with_ci(rows, bootstrap_samples, seed)
    summary = {"model": extractor.name, "num_pairs": len(rows), **image_summary}
    summary["protocol"] = {
        "exclusion_radius_px": exclusion_radius_px,
        "image_size": list(image_size) if image_size else None,
        "match_feature_size": list(match_feature_size) if match_feature_size else None,
        "model_preprocessing": extractor.protocol_metadata(),
        "pck_thresholds": list(pck_thresholds),
        "query_border": query_border,
        "query_stride": query_stride,
        "seed": seed,
    }
    summary["by_warp"] = {
        warp.name: _aggregate_image_level_with_ci(
            [row for row in rows if row["warp"] == warp.name], bootstrap_samples, seed
        )
        for warp in warps
    }
    return rows, summary


def _parse_warp(text: str) -> WarpSpec:
    # name,tx,ty,rotation,brightness
    fields = text.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "warp must be name,tx_px,ty_px,rotation_deg,brightness"
        )
    return WarpSpec(fields[0], *[float(value) for value in fields[1:]])


def _parse_hw(text: str) -> Tuple[int, int]:
    fields = text.lower().split("x")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("image size must be HxW")
    return int(fields[0]), int(fields[1])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-spec", help="JSON dense-extractor specification; omit for RGB sanity check")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--image-size", type=_parse_hw)
    parser.add_argument("--query-stride", type=int, default=16)
    parser.add_argument("--query-border", type=int, default=16)
    parser.add_argument("--pck-thresholds", type=float, nargs="+", default=[4.0, 8.0, 16.0])
    parser.add_argument("--exclusion-radius-px", type=float, default=8.0)
    parser.add_argument("--match-feature-size", type=_parse_hw)
    parser.add_argument("--warp-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--warp",
        action="append",
        type=_parse_warp,
        help="name,tx_px,ty_px,rotation_deg,brightness; repeat as needed",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    image_paths = sorted(
        path for path in Path(args.image_dir).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    warps = args.warp or [
        WarpSpec("translate_8px", tx_px=8.0),
        WarpSpec("rotate_5deg", rotation_deg=5.0),
        WarpSpec("translate_brightness", tx_px=8.0, ty_px=4.0, brightness=0.8),
    ]
    extractor = DenseExtractor.from_spec(args.model_spec, args.checkpoint, args.device)
    rows, summary = evaluate(
        extractor,
        image_paths,
        warps,
        args.device,
        args.query_stride,
        args.query_border,
        args.pck_thresholds,
        args.exclusion_radius_px,
        args.image_size,
        args.match_feature_size,
        args.warp_batch_size,
        args.bootstrap_samples,
        args.seed,
    )
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "per_pair.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
