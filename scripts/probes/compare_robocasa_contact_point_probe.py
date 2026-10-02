#!/usr/bin/env python3
"""Paired checkpoint comparison for RoboCasa cross-demo contact-point matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_KEYS = ("us3", "us6", "us8", "us10")


def find_checkpoint(report: dict, substring: str) -> tuple[str, dict]:
    hits = [(name, value) for name, value in report["checkpoint_results"].items()
            if substring in name]
    if len(hits) != 1:
        raise ValueError(f"Expected one checkpoint matching {substring!r}, found {len(hits)}")
    return hits[0]


def paired_rows(reference: list[dict], candidate: list[dict]) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    def keyed(rows: list[dict]) -> dict[tuple, dict]:
        result = {}
        for row in rows:
            key = row["camera"], row["source_demo"], row["target_demo"]
            if key in result:
                raise ValueError(f"Duplicate source-target pair: {key}")
            result[key] = row
        return result
    ref = keyed(reference)
    cand = keyed(candidate)
    if set(ref) != set(cand):
        raise ValueError("Checkpoint pair sets differ")
    keys = sorted(ref)
    if any(ref[key]["target_xy"] != cand[key]["target_xy"] for key in keys):
        raise ValueError("Target keypoints differ between checkpoints")
    return keys, np.array([ref[key]["error_px"] for key in keys]), np.array([cand[key]["error_px"] for key in keys])


def paired_bootstrap(keys: list[tuple], delta: np.ndarray, *, draws: int, seed: int) -> list[float]:
    demos = sorted({key[1] for key in keys} | {key[2] for key in keys})
    demo_index = {name: index for index, name in enumerate(demos)}
    source = np.array([demo_index[key[1]] for key in keys])
    target = np.array([demo_index[key[2]] for key in keys])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(len(demos), size=len(demos)), minlength=len(demos))
        weights = counts[source] * counts[target]
        if weights.sum():
            estimates.append(float(np.average(delta, weights=weights)))
    if not estimates:
        raise ValueError("Bootstrap produced no valid pair weights")
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path,
                        help="Second report with the same image manifest, e.g. another fixed prompt")
    parser.add_argument("--reference", required=True, help="Unique substring of reference checkpoint path")
    parser.add_argument("--candidate", required=True, help="Unique substring of candidate checkpoint path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("Need at least 100 bootstrap draws")
    report = json.loads(args.report.read_text())
    candidate_report_path = args.candidate_report or args.report
    candidate_report = json.loads(candidate_report_path.read_text())
    if report["manifest_sha256"] != candidate_report["manifest_sha256"]:
        raise ValueError("Image manifests differ between reports")
    reference_name, reference = find_checkpoint(report, args.reference)
    candidate_name, candidate = find_checkpoint(candidate_report, args.candidate)
    if reference_name == candidate_name and args.report.resolve() == candidate_report_path.resolve():
        parser.error("Reference and candidate must differ")
    output = {
        "scope": "Exploratory paired cross-demo point matching; demo-resampled uncertainty",
        "source_report": str(args.report.resolve()),
        "candidate_report": str(candidate_report_path.resolve()),
        "manifest_sha256": report["manifest_sha256"],
        "reference_prompt": report["prompt"],
        "candidate_prompt": candidate_report["prompt"],
        "reference": reference_name,
        "candidate": candidate_name,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "bootstrap_unit": "resample demos; weight each ordered source-target pair by both demo counts",
        "maps": {},
    }
    for layer in FEATURE_KEYS:
        if layer not in reference["maps"] or layer not in candidate["maps"]:
            continue
        keys, ref_error, cand_error = paired_rows(reference["maps"][layer]["pairs"],
                                                 candidate["maps"][layer]["pairs"])
        metrics = {}
        for threshold in (4, 8, 16):
            delta = (cand_error <= threshold).astype(float) - (ref_error <= threshold).astype(float)
            metrics[f"pck{threshold}_difference"] = {
                "estimate": float(delta.mean()),
                "confidence_interval_95": paired_bootstrap(keys, delta, draws=args.bootstrap_draws,
                                                             seed=args.seed),
            }
        error_delta = cand_error - ref_error
        metrics["mean_error_px_difference"] = {
            "estimate": float(error_delta.mean()),
            "confidence_interval_95": paired_bootstrap(keys, error_delta,
                                                         draws=args.bootstrap_draws, seed=args.seed),
        }
        output["maps"][layer] = {"pairs": len(keys), "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
