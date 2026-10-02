#!/usr/bin/env python3
"""Summarize visual target-mask probes across distinct RoboCasa tasks.

This is a descriptive semantic diagnostic. Average precision and its gap to
position/RGB controls are reported per task and deployed feature map; no
single-task or macro average is treated as a release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FEATURE_KEYS = ("us3", "us6", "us8")


def unique_checkpoint(report: dict, substring: str) -> tuple[str, dict]:
    matches = [(name, value) for name, value in report["checkpoint_results"].items()
               if substring in name]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint containing {substring!r}; found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-report", action="append", type=Path, required=True)
    parser.add_argument("--reference", default="checkpoint-2-ema")
    parser.add_argument("--candidate", default="checkpoint-1000-ema")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.task_report) < 2:
        parser.error("Need reports from at least two distinct tasks")
    tasks = {}
    checkpoint_pair = None
    for path in args.task_report:
        report = json.loads(path.read_text())
        manifest_path = Path(report["manifest"])
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != report["manifest_sha256"]:
            raise ValueError(f"Manifest changed: {manifest_path}")
        manifest = json.loads(manifest_bytes)
        name = manifest["env_name"]
        if name in tasks:
            raise ValueError(f"Duplicate task: {name}")
        reference_name, reference = unique_checkpoint(report, args.reference)
        candidate_name, candidate = unique_checkpoint(report, args.candidate)
        if reference_name == candidate_name:
            raise ValueError("Reference and candidate checkpoint must differ")
        pair = (reference_name, candidate_name)
        if checkpoint_pair is not None and pair != checkpoint_pair:
            raise ValueError("Checkpoint paths differ across task reports")
        checkpoint_pair = pair
        if report["prompt"]:
            raise ValueError(f"Use empty-prompt visual probes for this summary: {path}")
        layer_results = {}
        for key in FEATURE_KEYS:
            size = 8 if key == "us3" else 16
            controls = report["controls"][f"{size}x{size}"]
            ref_ap = float(reference["maps"][key]["mean_average_precision"])
            cand_ap = float(candidate["maps"][key]["mean_average_precision"])
            position_ap = float(controls["position_prior"]["mean_average_precision"])
            rgb_ap = float(controls["rgb_3x3_patch"]["mean_average_precision"])
            layer_results[key] = {
                "reference_ap": ref_ap,
                "candidate_ap": cand_ap,
                "candidate_minus_reference_ap": cand_ap - ref_ap,
                "position_prior_ap": position_ap,
                "rgb_patch_ap": rgb_ap,
                "candidate_minus_position_ap": cand_ap - position_ap,
                "candidate_minus_rgb_ap": cand_ap - rgb_ap,
            }
        tasks[name] = {
            "report": str(path.resolve()),
            "manifest_sha256": report["manifest_sha256"],
            "target_geom_selection": manifest.get("target_geom_prefix", manifest.get("target_geom_substring")),
            "valid_rows": report["valid_rows"],
            "layers": layer_results,
        }
    macro = {
        key: {
            field: sum(task["layers"][key][field] for task in tasks.values()) / len(tasks)
            for field in ("candidate_minus_reference_ap", "candidate_minus_position_ap")
        }
        for key in FEATURE_KEYS
    }
    output = {
        "scope": "Exploratory visual target-object masks across tasks; descriptive only",
        "reference_checkpoint": checkpoint_pair[0],
        "candidate_checkpoint": checkpoint_pair[1],
        "task_count": len(tasks),
        "tasks": tasks,
        "task_macro_means": macro,
        "release_gate": False,
        "interpretation": "Object masks do not establish language grounding, contact precision, or policy success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
