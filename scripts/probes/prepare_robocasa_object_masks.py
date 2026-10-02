#!/usr/bin/env python3
"""Replay RoboCasa demonstrations and label a task target by geom prefix.

Use the recorded RGB as model input and MuJoCo segmentation from the saved XML
and state as a visible-object mask. This is an exploratory feature diagnostic:
the selected prefix must be checked for each task, and replay/image alignment
must be inspected before interpreting the probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from PIL import Image
import robocasa


GEOM_SEGMENT_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)


def save_image(path: Path, array: np.ndarray) -> str:
    Image.fromarray(array).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-geom-prefix", required=True)
    parser.add_argument("--contrast-geom-prefix",
                        help="Optional second visible object in the same frame for a target-switch probe")
    parser.add_argument("--demo-count", type=int, default=32)
    parser.add_argument("--phase", type=float, default=0.2)
    parser.add_argument("--camera", default="robot0_eye_in_hand")
    parser.add_argument("--save-replay-preview", action="store_true")
    args = parser.parse_args()
    if args.demo_count < 8 or not 0 <= args.phase < 1 or not args.target_geom_prefix:
        parser.error("Need at least eight demos, a phase in [0,1), and a nonempty target prefix")
    if args.contrast_geom_prefix == args.target_geom_prefix:
        parser.error("Contrast and target geom prefixes must differ")

    source = args.dataset_file.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(source, "r") as handle:
        data = handle["data"]
        env_meta = json.loads(data.attrs["env_args"])
        kwargs = dict(env_meta["env_kwargs"])
        kwargs["env_name"] = env_meta["env_name"]
        kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True)
        env = robocasa.make(**kwargs)
        try:
            for index in range(1, args.demo_count + 1):
                demo_name = f"demo_{index}"
                if demo_name not in data:
                    raise KeyError(f"Missing {demo_name} in {source}")
                demo = data[demo_name]
                frame_index = min(int(len(demo["states"]) * args.phase), len(demo["states"]) - 1)
                state = np.asarray(demo["states"][frame_index])
                xml = str(demo.attrs["model_file"])
                ep_meta = json.loads(demo.attrs["ep_meta"])
                env.set_ep_meta(ep_meta)
                env.reset()
                env.reset_from_xml_string(env.edit_model_xml(xml))
                env.sim.reset()
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if hasattr(env, "update_state"):
                    env.update_state()

                names = [env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)]
                geom_ids = [i for i, name in enumerate(names)
                            if name and name.startswith(args.target_geom_prefix)]
                if not geom_ids:
                    raise ValueError(f"No {args.target_geom_prefix!r} geoms in {demo_name}")
                frame = np.asarray(demo["obs"][f"{args.camera}_image"][frame_index])
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"Unexpected recorded RGB shape/dtype in {demo_name}")
                height, width = frame.shape[:2]
                segmentation = env.sim.render(width=width, height=height,
                                              camera_name=args.camera, segmentation=True)[::-1]
                mask = ((segmentation[:, :, 0] == GEOM_SEGMENT_TYPE)
                        & np.isin(segmentation[:, :, 1], geom_ids))
                stem = f"{env_meta['env_name']}_{demo_name}_{frame_index:04d}_{args.camera}"
                image_name = stem + ".png"
                mask_name = stem + "_target_mask.png"
                image_sha = save_image(output / image_name, frame)
                mask_sha = save_image(output / mask_name, (mask * 255).astype(np.uint8))
                row = {
                    "demo": demo_name,
                    "frame_index": frame_index,
                    "camera": args.camera,
                    "image": image_name,
                    "sha256": image_sha,
                    "object_mask": mask_name,
                    "object_mask_sha256": mask_sha,
                    "object_mask_pixels": int(mask.sum()),
                    "in_frame": bool(mask.any()),
                    "image_size": [height, width],
                    "target_geom_prefix": args.target_geom_prefix,
                    "target_geom_count": len(geom_ids),
                    "instruction": ep_meta.get("lang", ""),
                    "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                }
                if args.contrast_geom_prefix:
                    contrast_ids = [i for i, name in enumerate(names)
                                    if name and name.startswith(args.contrast_geom_prefix)]
                    if not contrast_ids:
                        raise ValueError(f"No {args.contrast_geom_prefix!r} geoms in {demo_name}")
                    contrast_mask = ((segmentation[:, :, 0] == GEOM_SEGMENT_TYPE)
                                     & np.isin(segmentation[:, :, 1], contrast_ids))
                    if np.any(mask & contrast_mask):
                        raise ValueError(f"Target/contrast masks overlap in {demo_name}")
                    object_cfgs = {cfg["name"]: cfg for cfg in ep_meta.get("object_cfgs", [])}
                    target_name = args.target_geom_prefix.rstrip("_")
                    contrast_name = args.contrast_geom_prefix.rstrip("_")
                    try:
                        target_category = object_cfgs[target_name]["info"]["cat"].replace("_", " ")
                        contrast_category = object_cfgs[contrast_name]["info"]["cat"].replace("_", " ")
                    except KeyError as exc:
                        raise ValueError(f"Missing object category for {demo_name}: {exc}") from exc
                    if target_category == contrast_category:
                        raise ValueError(f"Identical target/contrast categories in {demo_name}")
                    instruction = str(ep_meta.get("lang", ""))
                    noun_phrase = f"the {target_category} from"
                    if instruction.count(noun_phrase) != 1:
                        raise ValueError(f"Cannot replace one target noun phrase in {demo_name}: {instruction}")
                    contrast_instruction = instruction.replace(
                        noun_phrase, f"the {contrast_category} from", 1)
                    contrast_name_file = stem + "_contrast_mask.png"
                    contrast_sha = save_image(
                        output / contrast_name_file, (contrast_mask * 255).astype(np.uint8))
                    row.update({
                        "contrast_geom_prefix": args.contrast_geom_prefix,
                        "contrast_geom_count": len(contrast_ids),
                        "contrast_object_mask": contrast_name_file,
                        "contrast_object_mask_sha256": contrast_sha,
                        "contrast_object_mask_pixels": int(contrast_mask.sum()),
                        "contrast_in_frame": bool(contrast_mask.any()),
                        "target_category": target_category,
                        "contrast_category": contrast_category,
                        "contrast_instruction": contrast_instruction,
                    })
                if args.save_replay_preview and index <= 3:
                    replay = env.sim.render(width=width, height=height,
                                            camera_name=args.camera)[::-1]
                    replay_name = stem + "_replay.png"
                    row["replay_image"] = replay_name
                    row["replay_image_sha256"] = save_image(output / replay_name, replay)
                    row["replay_mean_abs_rgb"] = float(np.abs(
                        frame.astype(np.float32) - replay.astype(np.float32)
                    ).mean())
                rows.append(row)
                print(f"{demo_name}: target geoms={len(geom_ids)} visible pixels={mask.sum()}", flush=True)
        finally:
            env.close()

    manifest = {
        "scope": "Exploratory task-target visible object mask from simulator replay",
        "source_file": str(source),
        "env_name": env_meta["env_name"],
        "target_geom_prefix": args.target_geom_prefix,
        "contrast_geom_prefix": args.contrast_geom_prefix,
        "trajectory_phase": args.phase,
        "cameras": [args.camera],
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    usable = sum(row["object_mask_pixels"] >= 32 for row in rows)
    print(f"saved {len(rows)} frames; {usable} masks with at least 32 pixels: {output}")


if __name__ == "__main__":
    main()
