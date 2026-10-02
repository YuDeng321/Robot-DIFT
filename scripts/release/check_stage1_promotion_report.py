#!/usr/bin/env python3
"""Fail closed before a long Stage-I run unless all release feature gates are recorded.

The report is an auditable decision from completed probe artifacts. This
validator checks provenance and coverage; it does not replace analysis of the
underlying metrics or make a release-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_GATES = frozenset({
    "us3_semantics",
    "us6_spatial",
    "us8_spatial",
    "object_part_semantics",
    "contact_localization",
    "matched_robot_transfer",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(report_path: Path, candidate: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("decision") != "continue_stage1":
        raise ValueError("Promotion report must explicitly select continue_stage1 under schema 1")
    candidate = candidate.resolve(strict=True)
    metadata = candidate / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing candidate metadata: {metadata}")
    if report.get("candidate_checkpoint") != str(candidate):
        raise ValueError("Promotion report identifies a different candidate checkpoint")
    if report.get("candidate_metadata_sha256") != sha256(metadata):
        raise ValueError("Candidate metadata changed after promotion review")
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATES or any(
        value is not True for value in gates.values()
    ):
        raise ValueError("All six semantic, spatial, and robot-transfer gates must be true")
    evidence = report.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < len(REQUIRED_GATES):
        raise ValueError("Promotion report lacks evidence for every gate")
    covered = set()
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {"gate", "path", "sha256"}:
            raise ValueError("Each evidence entry needs gate, path, and sha256")
        gate = entry["gate"]
        if gate not in REQUIRED_GATES or gate in covered:
            raise ValueError(f"Unknown or duplicate evidence gate: {gate}")
        path = Path(entry["path"])
        if not path.is_absolute() or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Evidence file missing or changed: {path}")
        covered.add(gate)
    if covered != REQUIRED_GATES:
        raise ValueError(f"Evidence is missing gates: {sorted(REQUIRED_GATES - covered)}")
    for entry in report.get("source_artifacts", []):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("Each source artifact needs path and sha256")
        path = Path(entry["path"])
        if not path.is_absolute() or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Source artifact missing or changed: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.report, args.candidate)
    print(json.dumps({
        "promotion_report": str(args.report.resolve()),
        "candidate_checkpoint": report["candidate_checkpoint"],
        "gates": report["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
