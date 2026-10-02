#!/usr/bin/env python3
"""Paired, image-level bootstrap comparison of correspondence result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_rows(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["image"], row["warp"])] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"model", "image", "warp"}
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference = load_rows(args.reference)
    candidate = load_rows(args.candidate)
    keys = sorted(set(reference) & set(candidate))
    if not keys or len(keys) != len(reference) or len(keys) != len(candidate):
        raise RuntimeError(
            f"Pair mismatch: reference={len(reference)}, candidate={len(candidate)}, shared={len(keys)}"
        )
    image_differences: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = sorted(set(reference[keys[0]]) & set(candidate[keys[0]]))
    for image_warp in keys:
        image_name, _ = image_warp
        for metric in metrics:
            image_differences[image_name][metric].append(
                candidate[image_warp][metric] - reference[image_warp][metric]
            )
    images = sorted(image_differences)
    matrix = np.asarray(
        [
            [np.mean(image_differences[image][metric]) for metric in metrics]
            for image in images
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(images), size=(args.bootstrap_samples, len(images)))
    bootstrap_means = matrix[indices].mean(axis=1)
    result = {
        "difference": "candidate_minus_reference",
        "num_images": len(images),
        "num_pairs": len(keys),
        "metrics": {
            metric: {
                "mean": float(matrix[:, idx].mean()),
                "ci95_low": float(np.percentile(bootstrap_means[:, idx], 2.5)),
                "ci95_high": float(np.percentile(bootstrap_means[:, idx], 97.5)),
            }
            for idx, metric in enumerate(metrics)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
