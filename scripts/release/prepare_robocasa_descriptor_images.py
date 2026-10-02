#!/usr/bin/env python3
"""Freeze a small, disjoint-from-DROID RoboCasa image set for descriptor gates.

The source files are read only. Four demos and four fixed trajectory phases per
task yield one frame from each demo, for 24 tasks x 4 images = 96 images. This
is a synthetic-warp descriptor diagnostic, not robot policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PHASES = (0.2, 0.4, 0.6, 0.8)
CAMERA = "robot0_agentview_left_image"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(source_root.glob("*/*/*/*.hdf5"))
    if len(files) != 24:
        raise RuntimeError(f"Expected the declared 24 RoboCasa task files, found {len(files)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in files:
        task = source.parent.parent.name
        with h5py.File(source, "r") as handle:
            for demo_index, phase in enumerate(PHASES, start=1):
                demo = f"demo_{demo_index}"
                if demo not in handle["data"]:
                    raise KeyError(f"Missing {demo} in {source}")
                frames = handle["data"][demo]["obs"][CAMERA]
                frame_index = min(int(len(frames) * phase), len(frames) - 1)
                frame = np.asarray(frames[frame_index])
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"Unexpected RGB shape or dtype in {source}:{demo}:{CAMERA}")
                name = f"{task}_{demo}_{frame_index:04d}.png"
                destination = output_dir / name
                Image.fromarray(frame, mode="RGB").save(destination)
                rows.append({
                    "image": name,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "task": task,
                    "source": str(source),
                    "demo": demo,
                    "frame_index": frame_index,
                    "trajectory_phase": phase,
                    "camera": CAMERA,
                    "source_resolution": list(frame.shape[:2]),
                })

    manifest = {
        "purpose": "Fixed cross-domain descriptor gate; synthetic image warps only",
        "image_count": len(rows),
        "task_count": len(files),
        "selection": "demo_1..demo_4 at trajectory fractions 0.2,0.4,0.6,0.8",
        "images": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} images from {len(files)} tasks to {output_dir}")


if __name__ == "__main__":
    main()
