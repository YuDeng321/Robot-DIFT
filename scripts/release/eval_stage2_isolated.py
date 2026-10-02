#!/usr/bin/env python3
"""Evaluate a saved Stage-II policy with one simulator process per episode.

Completed episode files are reusable after a failed run. The combined JSONL is
written only after every requested episode passes provenance checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _read_episode(path: Path, episode: int) -> tuple[dict, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1 or records[0].get("episode") != episode:
        raise ValueError(f"Expected only episode {episode} in {path}")
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("episodes") != 1 or summary.get("start_episode") != episode:
        raise ValueError(f"Incorrect episode range in {summary_path}")
    if summary.get("successes") != int(bool(records[0].get("success"))):
        raise ValueError(f"Success mismatch between {path} and {summary_path}")
    return records[0], summary


def merge_episodes(episode_dir: Path, output: Path, start: int, count: int) -> dict:
    rows = []
    summaries = []
    for episode in range(start, start + count):
        row, summary = _read_episode(episode_dir / f"episode_{episode:04d}.jsonl", episode)
        rows.append(row)
        summaries.append(summary)
    provenance = (
        "config_sha256", "artifact_kind", "artifact_sha256", "student", "task",
        "seed", "style_ids", "training_epoch", "inference_scheduler", "inference_steps",
    )
    first = summaries[0]
    for episode, (row, summary) in enumerate(zip(rows, summaries), start):
        if any(summary.get(key) != first.get(key) for key in provenance):
            raise ValueError(f"Checkpoint or protocol changed at episode {episode}")
        if row.get("training_epoch") != first.get("training_epoch") or row.get("seed") != first.get("seed"):
            raise ValueError(f"Episode metadata differs from checkpoint at episode {episode}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(temporary, output)
    aggregate = {
        **{key: first.get(key) for key in provenance},
        "episodes": count,
        "start_episode": start,
        "successes": sum(bool(row["success"]) for row in rows),
        "episode_dir": str(episode_dir.resolve()),
        "output": str(output.resolve()),
        "isolation": "one fresh process per episode",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--full-checkpoint", type=Path)
    source.add_argument("--adapter", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--style-id", type=int, choices=tuple(range(12)))
    parser.add_argument("--training-epoch", type=int)
    parser.add_argument("--scheduler", choices=("ddpm", "ddim"))
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.start_episode < 0:
        parser.error("episodes must be positive and start episode nonnegative")
    if args.output.exists():
        parser.error(f"Combined output already exists: {args.output}")
    episode_dir = args.episode_dir or args.output.with_name(args.output.stem + "_isolated")
    episode_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("eval_stage2_checkpoint.py")
    for episode in range(args.start_episode, args.start_episode + args.episodes):
        path = episode_dir / f"episode_{episode:04d}.jsonl"
        if path.exists():
            _read_episode(path, episode)
            continue
        command = [
            sys.executable, str(runner), "--config", str(args.config),
            "--episodes", "1", "--start-episode", str(episode), "--output", str(path),
        ]
        command.extend(
            ["--full-checkpoint", str(args.full_checkpoint)] if args.full_checkpoint
            else ["--adapter", str(args.adapter)]
        )
        for flag, value in (
            ("--style-id", args.style_id), ("--training-epoch", args.training_epoch),
            ("--scheduler", args.scheduler), ("--sampling-steps", args.sampling_steps),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        print(f"Evaluating episode {episode}", flush=True)
        subprocess.run(command, check=True)
        _read_episode(path, episode)
    print(json.dumps(merge_episodes(episode_dir, args.output, args.start_episode, args.episodes), indent=2))


if __name__ == "__main__":
    main()
