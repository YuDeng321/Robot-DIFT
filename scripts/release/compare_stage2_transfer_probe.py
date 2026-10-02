#!/usr/bin/env python3
"""Compare two RoboCasa policy rollouts with explicit episode matching limits.

Inputs may be the structured ``sim_episodes.jsonl`` from new runs or older
Slurm stdout containing ``[SimScene]`` and ``[SimEpisode]`` records. Text logs
only verify task, episode number, layout, style and object split. Exact reset
state and policy RNG pairing require structured records from both runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path


SCENE = re.compile(
    r"\[SimScene\] env=(\S+) episode=(\d+) layout_id=(\S+) "
    r"style_id=(\S+) obj_instance_split=(\S+)"
)
RESULT = re.compile(
    r"\[SimEpisode\] env=(\S+) episode=(\d+) steps=(\d+) success=(True|False)"
)


def _optional(value: str) -> str | None:
    return None if value == "None" else value


def read_episodes(path: Path, expected_episodes: int | None = None) -> dict[int, dict]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        scenes: dict[tuple[str, int], dict] = {}
        results: dict[tuple[str, int], dict] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            scene = SCENE.search(line)
            if scene:
                task, episode, layout, style, split = scene.groups()
                key = (task, int(episode))
                if key in scenes:
                    raise ValueError(f"Repeated scene in {path}: {key}; select one evaluation")
                scenes[key] = {
                    "task": task,
                    "episode": int(episode),
                    "layout_id": _optional(layout),
                    "style_id": _optional(style),
                    "obj_instance_split": _optional(split),
                    "reset_state_sha256": None,
                    "policy_seed": None,
                }
            result = RESULT.search(line)
            if result:
                task, episode, steps, success = result.groups()
                key = (task, int(episode))
                if key in results:
                    raise ValueError(f"Repeated episode result in {path}: {key}")
                results[key] = {"steps": int(steps), "success": success == "True"}
        if set(scenes) != set(results):
            raise ValueError(f"Incomplete scene/result records in {path}")
        records = [{**scenes[key], **results[key]} for key in sorted(scenes)]
    if expected_episodes is not None and len(records) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes in {path}, found {len(records)}")
    tasks = {record["task"] for record in records}
    if len(tasks) != 1:
        raise ValueError(f"Expected one task in {path}, found {sorted(tasks)}")
    episodes: dict[int, dict] = {}
    for record in records:
        episode = int(record["episode"])
        if episode in episodes:
            raise ValueError(f"Repeated episode {episode} in {path}")
        if not isinstance(record["success"], bool) or int(record["steps"]) < 1:
            raise ValueError(f"Invalid success/steps in {path}, episode {episode}")
        episodes[episode] = record
    return episodes


def read_episode_parts(paths: list[Path], expected_episodes: int) -> dict[int, dict]:
    """Join disjoint rollout chunks before checking exact scene/RNG pairing."""
    episodes: dict[int, dict] = {}
    for path in paths:
        part = read_episodes(path)
        overlap = set(episodes).intersection(part)
        if overlap:
            raise ValueError(f"Repeated episode IDs across parts: {sorted(overlap)}")
        episodes.update(part)
    expected_ids = set(range(expected_episodes))
    if set(episodes) != expected_ids:
        missing = sorted(expected_ids - set(episodes))
        extra = sorted(set(episodes) - expected_ids)
        raise ValueError(f"Episode parts are incomplete: missing={missing}, extra={extra}")
    return episodes


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] * (upper - position) + values[upper] * (position - lower) if lower != upper else values[lower]


def compare(reference: dict[int, dict], candidate: dict[int, dict], bootstrap_samples: int, seed: int) -> dict:
    if set(reference) != set(candidate):
        raise ValueError("Episode IDs differ between runs")
    ids = sorted(reference)
    for episode in ids:
        left, right = reference[episode], candidate[episode]
        for field in ("task", "layout_id", "style_id", "obj_instance_split"):
            if str(left.get(field)) != str(right.get(field)):
                raise ValueError(f"Episode {episode} differs in {field}: {left.get(field)} vs {right.get(field)}")
        if left.get("seed") is not None and right.get("seed") is not None and left["seed"] != right["seed"]:
            raise ValueError(f"Episode {episode} differs in simulator seed")

    exact_state = all(
        reference[i].get("reset_state_sha256")
        and reference[i].get("reset_state_sha256") == candidate[i].get("reset_state_sha256")
        for i in ids
    )
    matched_policy_rng = all(
        reference[i].get("policy_seed") is not None
        and reference[i].get("policy_seed") == candidate[i].get("policy_seed")
        for i in ids
    )
    paired = exact_state and matched_policy_rng
    a = [int(reference[i]["success"]) for i in ids]
    b = [int(candidate[i]["success"]) for i in ids]
    n = len(ids)
    rng = random.Random(seed)
    differences = []
    for _ in range(bootstrap_samples):
        left_indices = [rng.randrange(n) for _ in range(n)]
        right_indices = left_indices if paired else [rng.randrange(n) for _ in range(n)]
        differences.append(
            sum(b[i] for i in right_indices) / n - sum(a[i] for i in left_indices) / n
        )
    differences.sort()
    reference_successes = sum(a)
    candidate_successes = sum(b)
    report = {
        "task": reference[ids[0]]["task"],
        "episodes": n,
        "episode_ids": ids,
        "scene_metadata_matches": True,
        "exact_reset_state_matches": bool(exact_state),
        "policy_rng_seed_matches": bool(matched_policy_rng),
        "difference_interval_method": "paired_episode_bootstrap" if paired else "independent_episode_bootstrap",
        "reference": {
            "successes": reference_successes,
            "success_rate": reference_successes / n,
            "wilson_ci95": wilson(reference_successes, n),
            "mean_steps": sum(int(reference[i]["steps"]) for i in ids) / n,
        },
        "candidate": {
            "successes": candidate_successes,
            "success_rate": candidate_successes / n,
            "wilson_ci95": wilson(candidate_successes, n),
            "mean_steps": sum(int(candidate[i]["steps"]) for i in ids) / n,
        },
        "candidate_minus_reference_success_rate": candidate_successes / n - reference_successes / n,
        "difference_bootstrap_ci95": [_quantile(differences, 0.025), _quantile(differences, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }
    if paired:
        candidate_only = sum(not left and right for left, right in zip(a, b))
        reference_only = sum(left and not right for left, right in zip(a, b))
        discordant = candidate_only + reference_only
        exact_p = min(
            1.0,
            2 * sum(math.comb(discordant, k) for k in range(min(candidate_only, reference_only) + 1))
            / (2 ** discordant),
        )
        report["paired_outcomes"] = {
            "both_success": sum(left and right for left, right in zip(a, b)),
            "candidate_only_success": candidate_only,
            "reference_only_success": reference_only,
            "both_failure": sum(not left and not right for left, right in zip(a, b)),
            "exact_mcnemar_two_sided_p": exact_p,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-part", action="append", type=Path, default=[])
    parser.add_argument("--candidate-part", action="append", type=Path, default=[])
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-exact-state", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_episodes < 1 or args.bootstrap_samples < 100:
        parser.error("expected episodes must be positive and bootstrap samples at least 100")
    reference_paths = [args.reference, *args.reference_part]
    candidate_paths = [args.candidate, *args.candidate_part]
    report = compare(
        read_episode_parts(reference_paths, args.expected_episodes),
        read_episode_parts(candidate_paths, args.expected_episodes),
        args.bootstrap_samples,
        args.seed,
    )
    report["input_files"] = {
        label: [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
        for label, paths in (("reference", reference_paths), ("candidate", candidate_paths))
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_exact_state and report["difference_interval_method"] != "paired_episode_bootstrap":
        raise SystemExit("Exact reset state and policy RNG seed pairing were not verified")


if __name__ == "__main__":
    main()
