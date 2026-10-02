#!/usr/bin/env python3
"""Gate a staged Stage-I continuation using fixed paired descriptor results.

This gate authorizes more representation training. Robot success still needs a
separate matched policy and rollout evaluation before a release claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _protocol_without_checkpoint(summary: dict) -> dict:
    protocol = json.loads(json.dumps(summary["protocol"]))
    protocol["model_preprocessing"].pop("checkpoint", None)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--early-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--paired-comparison", type=Path, required=True)
    parser.add_argument(
        "--deployed-map", action="append", nargs=4, required=True,
        metavar=("KEY", "REFERENCE_SUMMARY", "CANDIDATE_SUMMARY", "PAIRED_COMPARISON"),
        help="required us3 and us8 measurements on the same fixed image set",
    )
    parser.add_argument(
        "--diagnostic-map", action="append", nargs=4, default=[],
        metavar=("KEY", "REFERENCE_SUMMARY", "CANDIDATE_SUMMARY", "PAIRED_COMPARISON"),
        help="optional non-deployed map measurement, currently us10",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _read(args.reference_summary)
    early = _read(args.early_summary)
    candidate = _read(args.candidate_summary)
    comparison = _read(args.paired_comparison)
    summaries = (reference, early, candidate)
    if any(int(item["num_images"]) != 96 or int(item["num_pairs"]) != 1056 for item in summaries):
        raise ValueError("Descriptor gate requires the fixed 96-image, 1,056-pair RoboCasa set")
    protocols = [_protocol_without_checkpoint(item) for item in summaries]
    if not (protocols[0] == protocols[1] == protocols[2]):
        raise ValueError("Descriptor preprocessing or matching protocol changed")
    if reference["protocol"]["model_preprocessing"]["feature_key"] != "us6":
        raise ValueError("Primary descriptor gate requires the us6 map")
    if comparison["num_images"] != 96 or comparison["num_pairs"] != 1056:
        raise ValueError("Paired comparison does not cover the fixed image/warp pairs")

    deployed_inputs = {item[0]: item[1:] for item in args.deployed_map}
    if len(deployed_inputs) != len(args.deployed_map) or set(deployed_inputs) != {"us3", "us8"}:
        raise ValueError("Descriptor gate requires exactly one us3 and one us8 deployed-map comparison")
    diagnostic_inputs = {item[0]: item[1:] for item in args.diagnostic_map}
    if len(diagnostic_inputs) != len(args.diagnostic_map) or set(diagnostic_inputs) - {"us10"}:
        raise ValueError("Only one optional us10 diagnostic-map comparison is supported")
    map_results = {}
    for key, (reference_path, candidate_path, comparison_path) in {
        **deployed_inputs, **diagnostic_inputs,
    }.items():
        map_reference = _read(Path(reference_path))
        map_candidate = _read(Path(candidate_path))
        map_comparison = _read(Path(comparison_path))
        if any(int(item["num_images"]) != 96 or int(item["num_pairs"]) != 1056
               for item in (map_reference, map_candidate)):
            raise ValueError(f"{key} does not cover the fixed image/warp pairs")
        if _protocol_without_checkpoint(map_reference) != _protocol_without_checkpoint(map_candidate):
            raise ValueError(f"{key} preprocessing or matching protocol changed")
        if any(item["protocol"]["model_preprocessing"]["feature_key"] != key
               for item in (map_reference, map_candidate)):
            raise ValueError(f"Map summary is not for {key}")
        if map_reference["protocol"]["model_preprocessing"]["checkpoint"] != reference["protocol"]["model_preprocessing"]["checkpoint"]:
            raise ValueError(f"{key} reference checkpoint differs from us6")
        if map_candidate["protocol"]["model_preprocessing"]["checkpoint"] != candidate["protocol"]["model_preprocessing"]["checkpoint"]:
            raise ValueError(f"{key} candidate checkpoint differs from us6")
        if map_comparison["num_images"] != 96 or map_comparison["num_pairs"] != 1056:
            raise ValueError(f"{key} paired comparison does not cover the fixed image/warp pairs")
        if map_comparison.get("difference") != "candidate_minus_reference":
            raise ValueError(f"{key} paired comparison has the wrong direction")
        pck8_difference = map_comparison["metrics"]["pck@8px"]
        if not all(math.isfinite(float(pck8_difference[field]))
                   for field in ("mean", "ci95_low", "ci95_high")):
            raise ValueError(f"{key} paired PCK@8 contains a nonfinite metric")
        expected_difference = float(map_candidate["pck@8px_mean"]) - float(map_reference["pck@8px_mean"])
        if abs(float(pck8_difference["mean"]) - expected_difference) > 1e-5:
            raise ValueError(f"{key} paired comparison direction or summary differs")
        map_results[key] = {
            "reference_pck8": float(map_reference["pck@8px_mean"]),
            "candidate_pck8": float(map_candidate["pck@8px_mean"]),
            "candidate_minus_reference_pck8": pck8_difference,
            "role": "deployed" if key in deployed_inputs else "diagnostic_only",
        }

    pck8 = comparison["metrics"]["pck@8px"]
    error = comparison["metrics"]["mean_error_px"]
    candidate_pck8 = float(candidate["pck@8px_mean"])
    early_pck8 = float(early["pck@8px_mean"])
    reference_pck8 = float(reference["pck@8px_mean"])
    if abs(float(pck8["mean"]) - (candidate_pck8 - reference_pck8)) > 1e-5:
        raise ValueError("Primary paired comparison direction or summary differs")
    values = (pck8["mean"], pck8["ci95_low"], error["ci95_high"], candidate_pck8)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Descriptor gate contains a nonfinite metric")

    checks = {
        "pck8_gain_at_least_0.01": float(pck8["mean"]) >= 0.01,
        "paired_pck8_ci_above_zero": float(pck8["ci95_low"]) > 0.0,
        "paired_mean_error_ci_below_zero": float(error["ci95_high"]) < 0.0,
        "pck8_no_more_than_0.005_below_early_100": candidate_pck8 >= early_pck8 - 0.005,
        "pck16_no_more_than_0.005_below_reference": (
            float(candidate["pck@16px_mean"]) >= float(reference["pck@16px_mean"]) - 0.005
        ),
        "us3_pck8_no_more_than_0.002_below_reference": (
            float(map_results["us3"]["candidate_minus_reference_pck8"]["mean"]) >= -0.002
        ),
        "us8_pck8_no_more_than_0.005_below_reference": (
            float(map_results["us8"]["candidate_minus_reference_pck8"]["mean"]) >= -0.005
        ),
    }
    report = {
        "passed": all(checks.values()),
        "scope": "Gate for a 5,000-update exploratory Stage-I continuation only",
        "checks": checks,
        "reference_pck8": reference_pck8,
        "early_100_pck8": early_pck8,
        "candidate_pck8": candidate_pck8,
        "candidate_minus_reference_pck8": pck8,
        "candidate_minus_reference_mean_error_px": error,
        "deployed_maps": {key: map_results[key] for key in ("us3", "us8")},
        "diagnostic_maps": {key: map_results[key] for key in diagnostic_inputs},
        "interpretation": (
            "us3 is the deployed coarse semantic tap, but synthetic 2-D warp PCK "
            "does not test semantic grounding. This gate permits exploratory "
            "training only; matched robot-task success is required for release."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
