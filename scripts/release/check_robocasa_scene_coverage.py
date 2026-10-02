#!/usr/bin/env python3
"""Reset each release task in proposed held-out RoboCasa scene styles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TASKS = ROOT / "configs/release/robocasa_24_tasks.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles", nargs="+", type=int, default=[9, 10])
    parser.add_argument("--task", help="check one task first")
    parser.add_argument("--output", type=Path, help="write structured results")
    args = parser.parse_args()
    if not args.styles or len(set(args.styles)) != len(args.styles):
        parser.error("--styles must contain unique style IDs")

    # Import only after CLI parsing so --help works without simulator assets.
    from robocasa.utils.env_utils import create_env
    from environments.utils.robocasa_aliases import normalize_env_name

    manifest = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = [item["name"] for item in manifest["tasks"]]
    if args.task:
        if args.task not in tasks:
            parser.error(f"Unknown release task: {args.task}")
        tasks = [args.task]

    records = []
    for task in tasks:
        for style in args.styles:
            env = None
            record = {"task": task, "requested_style": style, "status": "error"}
            try:
                env = create_env(
                    env_name=normalize_env_name(task),
                    camera_names=[
                        "robot0_agentview_left",
                        "robot0_agentview_right",
                        "robot0_eye_in_hand",
                    ],
                    camera_widths=128,
                    camera_heights=128,
                    render_onscreen=False,
                    seed=42,
                    style_ids=[style],
                )
                env.reset()
                episode_meta = env.get_ep_meta()
                layout_id = episode_meta.get("layout_id")
                style_id = episode_meta.get("style_id")
                record["layout_id"] = int(layout_id) if layout_id is not None else None
                record["style_id"] = int(style_id) if style_id is not None else None
                if record["style_id"] != style:
                    raise RuntimeError(f"realized style {record['style_id']} differs from {style}")
                record["status"] = "ok"
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        record["status"] = "error"
                        record["error"] = f"close: {type(error).__name__}: {error}"
            records.append(record)
            print(json.dumps(record), flush=True)

    report = {"styles": args.styles, "tasks": tasks, "records": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(record["status"] != "ok" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
