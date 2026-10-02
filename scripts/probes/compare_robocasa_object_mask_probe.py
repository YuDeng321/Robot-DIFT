#!/usr/bin/env python3
"""Paired held-out-demo comparison of RoboCasa target-object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_checkpoint(report: dict, substring: str) -> tuple[str, dict]:
    matches = [(name, result) for name, result in report["checkpoint_results"].items()
               if substring in name]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint matching {substring!r}; found {len(matches)}")
    return matches[0]


def compare_layer(reference: dict, candidate: dict, draws: int, seed: int) -> dict:
    ref = {row["demo"]: row for row in reference["by_demo"]}
    cand = {row["demo"]: row for row in candidate["by_demo"]}
    if set(ref) != set(cand):
        raise ValueError("Held-out demo sets differ")
    demos = sorted(ref)
    rng = np.random.default_rng(seed)
    metrics = {}
    for key in ("average_precision", "roc_auc"):
        deltas = np.array([cand[demo][key] - ref[demo][key] for demo in demos])
        sample_indices = rng.integers(len(demos), size=(draws, len(demos)))
        estimates = deltas[sample_indices].mean(axis=1)
        metrics[f"{key}_difference"] = {
            "estimate": float(deltas.mean()),
            "confidence_interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        }
    return {"demos": len(demos), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path,
                        help="Second report with the same image manifest, e.g. another fixed prompt")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
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
    ref_name, ref = find_checkpoint(report, args.reference)
    cand_name, cand = find_checkpoint(candidate_report, args.candidate)
    if ref_name == cand_name and args.report.resolve() == candidate_report_path.resolve():
        parser.error("Reference and candidate must differ")
    result = {
        "scope": "Exploratory single-task visual object-mask comparison",
        "source_report": str(args.report.resolve()),
        "candidate_report": str(candidate_report_path.resolve()),
        "manifest_sha256": report["manifest_sha256"],
        "reference_prompt": report["prompt"],
        "candidate_prompt": candidate_report["prompt"],
        "reference": ref_name,
        "candidate": cand_name,
        "bootstrap_unit": "held-out demo; support sets overlap across folds",
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "maps": {key: compare_layer(ref["maps"][key], cand["maps"][key],
                                    args.bootstrap_draws, args.seed)
                 for key in ("us3", "us6", "us8")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
