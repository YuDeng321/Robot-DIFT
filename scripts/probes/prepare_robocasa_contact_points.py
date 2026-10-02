#!/usr/bin/env python3
"""Project a RoboCasa contact geom into recorded demonstration frames.

The saved XML and simulator state define the camera and target position. This
creates a small *diagnostic* keypoint set; the projected collision geom is not
itself a visible-pixel annotation. Rows include a depth check so occluded or
off-screen points can be excluded by an evaluator. Source HDF5 is read only.
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
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map


DEFAULT_CAMERAS = ("robot0_agentview_right", "robot0_eye_in_hand")
GEOM_SEGMENT_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)


def project_geom(sim, geom_id: int, camera: str, image_size: tuple[int, int]) -> tuple[float, float, float]:
    height, width = image_size
    world = np.concatenate((np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64), [1.0]))
    projected = get_camera_transform_matrix(sim, camera, height, width) @ world
    if projected[2] <= 0 or not np.isfinite(projected).all():
        return float("nan"), float("nan"), float(projected[2])
    return float(projected[0] / projected[2]), float(projected[1] / projected[2]), float(projected[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-geom-substring", default="coffee_machine,start_button",
                        help="Comma-separated name fragments that must all match the target geom")
    parser.add_argument("--demo-count", type=int, default=8)
    parser.add_argument("--phase", type=float, default=0.2)
    parser.add_argument("--camera", action="append", dest="cameras")
    args = parser.parse_args()
    if args.demo_count < 2 or not 0 <= args.phase < 1:
        parser.error("Need at least two demos and a phase in [0,1)")
    cameras = tuple(args.cameras or DEFAULT_CAMERAS)
    target_fragments = tuple(part.strip() for part in args.target_geom_substring.split(",") if part.strip())
    if not target_fragments:
        parser.error("Need at least one target geom name fragment")
    if len(set(cameras)) != len(cameras):
        parser.error("Camera names must be unique")
    source = args.dataset_file.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        data = handle["data"]
        meta = json.loads(data.attrs["env_args"])
        kwargs = dict(meta["env_kwargs"])
        kwargs["env_name"] = meta["env_name"]
        kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True)
        env = robocasa.make(**kwargs)
        rows = []
        try:
            for demo_index in range(1, args.demo_count + 1):
                demo_name = f"demo_{demo_index}"
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

                names = [env.sim.model.geom_id2name(index) for index in range(env.sim.model.ngeom)]
                matching = [index for index, name in enumerate(names)
                            if name and all(fragment in name for fragment in target_fragments)]
                if len(matching) != 1:
                    raise ValueError(f"Expected one target geom in {demo_name}; found {[(i, names[i]) for i in matching]}")
                geom_id = matching[0]
                fixture_prefix = names[geom_id].rsplit("_start_button", 1)[0]
                fixture_geom_ids = [index for index, name in enumerate(names)
                                    if name and name.startswith(fixture_prefix)]
                for camera in cameras:
                    frame = np.asarray(demo["obs"][f"{camera}_image"][frame_index])
                    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                        raise ValueError(f"Unexpected RGB frame {demo_name}:{camera}:{frame_index}")
                    height, width = frame.shape[:2]
                    x, y, point_depth = project_geom(env.sim, geom_id, camera, (height, width))
                    in_frame = bool(np.isfinite((x, y)).all() and 0 <= x < width and 0 <= y < height)
                    surface_depth = None
                    depth_gap = None
                    surface_geom = None
                    surface_same_fixture = False
                    front_facing_score = None
                    mask_path = None
                    mask_sha256 = None
                    mask_pixels = None
                    if in_frame:
                        _, raw_depth = env.sim.render(width=width, height=height,
                                                      camera_name=camera, depth=True)
                        depth = get_real_depth_map(env.sim, raw_depth)[::-1]
                        ix = min(max(int(round(x)), 0), width - 1)
                        iy = min(max(int(round(y)), 0), height - 1)
                        surface_depth = float(depth[iy, ix])
                        depth_gap = float(point_depth - surface_depth)
                        segmentation = env.sim.render(width=width, height=height,
                                                      camera_name=camera, segmentation=True)[::-1]
                        object_mask = ((segmentation[:, :, 0] == GEOM_SEGMENT_TYPE)
                                       & np.isin(segmentation[:, :, 1], fixture_geom_ids))
                        mask_pixels = int(object_mask.sum())
                        mask_path = f"{meta['env_name']}_{demo_name}_{frame_index:04d}_{camera}_target_mask.png"
                        Image.fromarray((object_mask * 255).astype(np.uint8)).save(output / mask_path)
                        mask_sha256 = hashlib.sha256((output / mask_path).read_bytes()).hexdigest()
                        visible_id = int(segmentation[iy, ix, 1])
                        if segmentation[iy, ix, 0] == GEOM_SEGMENT_TYPE and 0 <= visible_id < len(names):
                            surface_geom = names[visible_id]
                        surface_same_fixture = bool(surface_geom and surface_geom.startswith(fixture_prefix))
                        cam_id = env.sim.model.camera_name2id(camera)
                        button_normal = env.sim.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
                        view = env.sim.data.cam_xpos[cam_id] - env.sim.data.geom_xpos[geom_id]
                        front_facing_score = float(button_normal @ view / np.linalg.norm(view))
                    name = f"{meta['env_name']}_{demo_name}_{frame_index:04d}_{camera}.png"
                    image_path = output / name
                    Image.fromarray(frame).save(image_path)
                    rows.append({
                        "image": name,
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "demo": demo_name,
                        "frame_index": frame_index,
                        "camera": camera,
                        "image_size": [height, width],
                        "target_geom": names[geom_id],
                        "target_xy": [x if np.isfinite(x) else None, y if np.isfinite(y) else None],
                        "target_depth_m": point_depth if np.isfinite(point_depth) else None,
                        "surface_depth_m": surface_depth,
                        "target_minus_surface_m": depth_gap,
                        "surface_geom": surface_geom,
                        "surface_same_fixture": surface_same_fixture,
                        "button_front_facing_score": front_facing_score,
                        "object_mask": mask_path,
                        "object_mask_sha256": mask_sha256,
                        "object_mask_pixels": mask_pixels,
                        "in_frame": in_frame,
                        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    })
                print(f"{demo_name} frame={frame_index} rows={len(rows)}", flush=True)
        finally:
            env.close()

    manifest = {
        "scope": "Exploratory projected contact point diagnostic; collision geom may be occluded",
        "source_file": str(source),
        "env_name": meta["env_name"],
        "target_geom_substring": args.target_geom_substring,
        "trajectory_phase": args.phase,
        "cameras": list(cameras),
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    valid = sum(row["in_frame"] and row["target_minus_surface_m"] is not None
                and abs(row["target_minus_surface_m"]) <= 0.08 for row in rows)
    print(f"saved {len(rows)} rows; {valid} within 8 cm of rendered surface: {output}")


if __name__ == "__main__":
    main()
