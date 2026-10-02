#!/usr/bin/env python3
"""Check the Stage-I release launcher against the paper protocol.

The launcher is run with ``--dry-run`` to obtain its exact command, and
``train_droid_auto.py`` resolves that command with ``--config_only`` in a
temporary workspace. Every check in configs/release/paper_stage1_protocol.json
is then evaluated on the resolved configuration, so shell defaults, CLI
defaults, and robomimic config defaults are all covered. Nothing is trained.

Examples (from the repository root, inside the training environment):
    python scripts/release/check_paper_protocol.py
    ROBOT_DIFT_GPUS=8 ROBOT_DIFT_PER_GPU_BATCH=32 ROBOT_DIFT_ACCUMULATION_STEPS=1 \\
        python scripts/release/check_paper_protocol.py --json
    python scripts/release/check_paper_protocol.py --check-assets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs/release/paper_stage1_protocol.json"
DEFAULT_SCRIPT = REPO_ROOT / "scripts/release/stage1_paper_protocol.sh"
CONFIG_RELATIVE_PATH = Path("policy/auto_generated_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="paper protocol JSON")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="Stage-I launcher")
    parser.add_argument("--python", default=sys.executable, help="training-environment Python")
    parser.add_argument(
        "--check-assets",
        action="store_true",
        help="also require the DROID root, SD2.1 snapshot, and CLIP file the launcher points to",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable audit")
    return parser.parse_args()


def launcher_command(script: Path) -> list[str]:
    env = os.environ.copy()
    env.setdefault("ROBOT_DIFT_TORCHRUN", "torchrun")
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Launcher dry run failed: {result.stderr.strip() or result.stdout.strip()}")
    return shlex.split(result.stdout)


def split_command(command: list[str]) -> tuple[int, list[str]]:
    """Return ``(processes, train_droid_auto.py arguments)``."""
    entry = [index for index, part in enumerate(command) if Path(part).name == "train_droid_auto.py"]
    if len(entry) != 1:
        raise ValueError("Launcher command does not run train_droid_auto.py exactly once")
    processes = 1
    for part in command[: entry[0]]:
        if part.startswith("--nproc_per_node="):
            processes = int(part.split("=", 1)[1])
    return processes, command[entry[0] + 1:]


def _option(arguments: list[str], name: str) -> str | None:
    values = [arguments[index + 1] for index, part in enumerate(arguments[:-1]) if part == name]
    return values[-1] if values else None


def resolve_config(python: str, arguments: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="robot_dift_protocol_") as work:
        env = os.environ.copy()
        env["WORK"] = work
        env["RANK"] = "0"
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        overrides = ["--checkpoint_dir", os.path.join(work, "policy")]
        if _option(arguments, "--save_cleandift_dir") is not None:
            overrides += ["--save_cleandift_dir", os.path.join(work, "encoder")]
        result = subprocess.run(
            [python, str(REPO_ROOT / "train_droid_auto.py"), *arguments, *overrides, "--config_only"],
            cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"train_droid_auto.py --config_only failed:\n{result.stderr[-4000:]}")
        return json.loads((Path(work) / CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def lookup(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def derived(name: str, config: dict[str, Any], processes: int) -> Any:
    if name == "global_batch":
        return processes * int(config["train"]["batch_size"]) * int(config["train"]["gradient_accumulation_steps"])
    if name == "alignment_final_weight":
        return float(config["algo"]["cleandift_alignment_weight"]) * float(
            config["algo"]["cleandift_alignment_min_decay_factor"]
        )
    raise ValueError(f"Unknown derived value {name}")


def matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return actual is expected or actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(expected, list) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(matches(a, e) for a, e in zip(actual, expected))
    return actual == expected


def asset_checks(arguments: list[str]) -> list[dict[str, Any]]:
    data_root = _option(arguments, "--data_path")
    clip = _option(arguments, "--clip_model")
    model_dir = os.environ.get("ROBOT_DIFT_MODEL_DIR")
    items = [
        ("droid_tfds_root", Path(data_root) / "droid" if data_root else None),
        ("sd21_model_index", Path(model_dir) / "model_index.json" if model_dir else None),
        ("clip_vit_b32", Path(clip) if clip else None),
    ]
    return [
        {"name": name, "path": str(path) if path else None, "status": "PRESENT" if path and path.exists() else "MISSING"}
        for name, path in items
    ]


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 2 or not isinstance(spec.get("checks"), list):
        raise ValueError("Unsupported paper protocol spec")

    command = launcher_command(args.script)
    processes, arguments = split_command(command)
    config = resolve_config(args.python, arguments)

    results = []
    for check in spec["checks"]:
        try:
            actual = derived(check["derived"], config, processes) if "derived" in check else lookup(config, check["path"])
            status = "MATCH" if matches(actual, check["expected"]) else "MISMATCH"
        except KeyError as missing:
            actual, status = f"missing config key {missing}", "MISMATCH"
        results.append({
            "name": check["name"],
            "status": status,
            "actual": actual,
            "expected": check["expected"],
            "source": check.get("source"),
        })
    choices = {}
    for name, choice in spec.get("implementation_choices", {}).items():
        path = choice.get("path")
        try:
            choices[name] = lookup(config, path) if path else choice.get("value")
        except KeyError:
            choices[name] = None
    assets = asset_checks(arguments) if args.check_assets else []

    protocol_match = all(item["status"] == "MATCH" for item in results)
    assets_ok = all(item["status"] == "PRESENT" for item in assets)
    report = {
        "launcher": str(args.script.relative_to(REPO_ROOT) if args.script.is_relative_to(REPO_ROOT) else args.script),
        "command": command,
        "processes": processes,
        "protocol_match": protocol_match,
        "checks": results,
        "implementation_choices": choices,
        "assets": assets,
        "limitations": spec.get("limitations", []),
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Paper protocol: {args.spec}")
        print(f"Launcher: {report['launcher']} ({processes} process(es))")
        for item in results:
            print(f"  [{item['status']}] {item['name']}: actual={item['actual']!s} paper={item['expected']!s}")
        print("Implementation choices (not stated by the paper):")
        for name, value in choices.items():
            print(f"  {name}: {value}")
        for item in assets:
            print(f"  [{item['status']}] {item['name']}: {item['path']}")
        matched = sum(item["status"] == "MATCH" for item in results)
        print(f"{matched}/{len(results)} paper checks match")
    return 0 if protocol_match and assets_ok else 1


if __name__ == "__main__":
    sys.exit(main())
