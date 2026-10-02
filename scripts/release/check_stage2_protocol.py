#!/usr/bin/env python3
"""Audit Stage-II RoboCasa candidate and paper-numeric presets without a GPU.

The launcher is run only with ``--dry-run``. Its command is then passed to
Hydra's ``--cfg job --resolve`` mode, which composes configuration without
starting training. A matching configuration is not evidence of a reproduced
paper result, a trained checkpoint, or a successful simulator rollout.

Examples::

    python scripts/release/check_stage2_protocol.py --json
    python scripts/release/check_stage2_protocol.py --preset paper --json
    python scripts/release/check_stage2_protocol.py \
        --work-root /path/to/work --json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = {
    "candidate": REPO_ROOT / "scripts/release/stage2_robocasa_candidate.sh",
    "paper": REPO_ROOT / "scripts/release/stage2_robocasa_paper_protocol.sh",
}
LAUNCHER = LAUNCHERS["candidate"]
TASK_MANIFEST = REPO_ROOT / "configs/release/robocasa_24_tasks.json"
DEFAULT_WORK_ROOT = Path.cwd()
FEATURE_KEYS = ["us3", "us6", "us8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(LAUNCHERS), default="candidate")
    parser.add_argument("--task", default="CoffeePressButton", help="one task in the 24-task manifest")
    parser.add_argument(
        "--work-root", type=Path, help="also check all 24 RoboCasa files and local model assets"
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable audit")
    return parser.parse_args()


def _override(command: list[str], name: str) -> str | None:
    prefix = name + "="
    values = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    return values[-1] if values else None


def compose_candidate(
    task: str, work_root: Path | None = None, *, preset: str = "candidate"
) -> tuple[list[str], dict[str, Any]]:
    """Read the canonical launcher command and ask Hydra for resolved YAML."""
    launcher = LAUNCHERS[preset]
    env = os.environ.copy()
    if work_root is not None:
        env["WORK"] = str(work_root.expanduser().resolve())
    dry_run = subprocess.run(
        ["bash", str(launcher), "--dry-run", task],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    command = shlex.split(dry_run.stdout)
    run_indices = [index for index, part in enumerate(command) if Path(part).name == "run.py"]
    if len(run_indices) != 1 or run_indices[0] < 1 or not command[0].endswith("/python"):
        raise ValueError("Stage-II dry-run did not produce a run.py command")
    run_index = run_indices[0]

    work = Path(env.get("WORK", str(DEFAULT_WORK_ROOT)))
    clip_path = _override(command, "agents.language_encoders.model_name")
    if not clip_path:
        raise ValueError("Stage-II dry-run omitted the local CLIP model override")
    env.setdefault("ROBOT_DIFT_CLIP_MODEL", clip_path)
    env.setdefault("ROBOT_DIFT_STAGE1_ENCODER", "/path/to/stage1/encoder/checkpoint-300000-ema")
    env.setdefault("ROBOT_DIFT_MODEL_DIR", str(work / "datasets/robot_dift/pretrained/sd2-1-base"))
    env.setdefault("WORK", str(work))
    env.setdefault("WANDB_MODE", "offline")
    resolved = subprocess.run(
        [command[0], *command[run_index:], "--cfg", "job", "--resolve"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    config = yaml.safe_load(resolved.stdout)
    if not isinstance(config, dict):
        raise ValueError("Hydra did not produce a resolved mapping")
    return command, config


def _get(config: dict[str, Any], *path: str) -> Any:
    current: Any = config
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _check(name: str, actual: Any, expected: Any, *, source: str = "resolved_hydra_config") -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "status": "MATCH" if actual == expected else "DIFF",
        "source": source,
    }


def _launcher_world_size(command: list[str]) -> int:
    values = [arg.split("=", 1)[1] for arg in command if arg.startswith("--nproc_per_node=")]
    if not values:
        return 1
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) < 1:
        raise ValueError("Invalid torchrun process count in launcher")
    return int(values[0])


def paper_numeric_checks(command: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare the resolved policy/trainer against stated Table S1 numerics."""
    agent = _get(config, "agents") or {}
    policy = agent.get("model", {})
    optimizer = agent.get("optimization", {})
    trainer = _get(config, "trainers") or {}
    world_size = _launcher_world_size(command)
    batch = trainer.get("train_batch_size")
    accumulation = trainer.get("gradient_accumulation_steps")
    effective_batch = batch * accumulation * world_size if isinstance(batch, int) and isinstance(accumulation, int) else None
    return [
        _check("action_diffusion_sampler", policy.get("scheduler_type"), "ddim"),
        _check("optimizer", optimizer.get("_target_"), "torch.optim.Adam"),
        _check("effective_global_batch", effective_batch, 256, source="launcher_world_size_and_resolved_hydra_config"),
        _check("full_accumulation_groups", trainer.get("full_accumulation_batches"), True),
        _check("ema_enabled", trainer.get("if_use_ema"), True),
        _check("ema_power", trainer.get("ema_power"), 0.75),
        _check("learning_rate_scheduler", trainer.get("lr_scheduler_type"), "linear"),
    ]


def _contains_call(method: ast.AST, receiver: str, name: str, argument: bool) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != name or len(node.args) != 1:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value is not argument:
            continue
        if receiver == "self" and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            return True
        if receiver == "super" and isinstance(node.func.value, ast.Call):
            super_call = node.func.value
            if isinstance(super_call.func, ast.Name) and super_call.func.id == "super":
                return True
    return False


def _class_methods(file: Path, class_name: str) -> tuple[ast.ClassDef, dict[str, ast.FunctionDef]]:
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    klass = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if klass is None:
        raise ValueError(f"class {class_name} missing from {file}")
    methods = {node.name: node for node in klass.body if isinstance(node, ast.FunctionDef)}
    return klass, methods


def implementation_checks() -> list[dict[str, Any]]:
    """Check explicit freeze and readout construction statements in local source."""
    student_file = REPO_ROOT / "agents/encoders/robot_dift_student_feature_extractor.py"
    wrapper_file = REPO_ROOT / "agents/encoders/robot_dift_candidate_obs_encoder.py"
    readout_file = REPO_ROOT / "agents/encoders/robot_dift_paper_readout.py"
    _, student = _class_methods(student_file, "RobotDIFTStudentFeatureExtractor")
    readout_class, _ = _class_methods(readout_file, "RobotDIFTPaperReadout")
    _, wrapper = _class_methods(wrapper_file, "RobotDIFTCandidateObsEncoder")

    student_frozen = (
        "__init__" in student
        and "train" in student
        and _contains_call(student["__init__"], "self", "requires_grad_", False)
        and _contains_call(student["train"], "super", "train", False)
    )
    constructs_readout = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "readout"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "RobotDIFTPaperReadout"
        for node in ast.walk(wrapper.get("__init__", ast.Pass()))
    )
    readout_explicitly_frozen = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "requires_grad_"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value is False
        for node in ast.walk(readout_class)
    )
    wrapper_explicitly_frozen = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "requires_grad_"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value is False
        for node in ast.walk(wrapper.get("__init__", ast.Pass()))
    )
    return [
        _check("student_frozen_in_source", student_frozen, True, source="student_python_ast"),
        _check(
            "candidate_readout_trainable_in_source",
            constructs_readout and not readout_explicitly_frozen and not wrapper_explicitly_frozen,
            True,
            source="candidate_python_ast",
        ),
    ]


def protocol_checks(command: list[str], config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    agent = _get(config, "agents") or {}
    obs = agent.get("obs_encoders", {})
    student = obs.get("rgb_model", {})
    policy = agent.get("model", {})
    trainer = _get(config, "trainers") or {}
    sim = trainer.get("simulation_cfg", {})
    epochs = _get(config, "epoch")
    sim_period = _get(config, "sim_eval_every_n_epochs")
    isolated_episodes = config.get("isolated_sim_eval_episodes")
    final_sim_epoch = epochs if sim_period == 0 and isolated_episodes == 50 else None
    # The launcher reloads the final EMA checkpoint in a fresh process. This
    # avoids retaining training caches and optimizer state during simulation.
    final_rollout_sets = int(final_sim_epoch is not None) + int(config.get("skip_final_sim") is not True)
    clip_query = obs.get("clip_model_path")
    clip_goal = _get(agent, "language_encoders", "model_name")
    rgb_shapes = [
        item.get("shape")
        for item in (_get(config, "shape_meta", "obs") or {}).values()
        if isinstance(item, dict) and item.get("type") == "rgb"
    ]

    checks = [
        _check("launcher_config", "--config-name=robocasa_config" in command, True, source="dry_run_command"),
        _check("launcher_agent", "agents=droid_diffusion_agent" in command, True, source="dry_run_command"),
        _check("launcher_policy", "agents/model=droid/droid_diffusion_unet_stage2" in command, True, source="dry_run_command"),
        _check("launcher_candidate_readout", "agents/obs_encoders=robot_dift_paper_candidate" in command, True, source="dry_run_command"),
        _check("task", _get(config, "env_name"), [task]),
        _check("observation_horizon", _get(config, "obs_seq_len"), 2),
        _check("prediction_horizon", _get(config, "pred_seq_len"), 16),
        _check("action_execution_horizon", _get(config, "act_seq_len"), 8),
        _check("agent_observation_horizon", agent.get("obs_seq_len"), 2),
        _check("agent_action_horizon", agent.get("act_seq_len"), 8),
        _check("policy_prediction_horizon", policy.get("action_seq_len"), 16),
        _check("observation_tokens", policy.get("obs_tokens"), 2),
        _check("trainset_observation_horizon", _get(trainer, "trainset", "obs_seq_len"), 2),
        _check("valset_observation_horizon", _get(trainer, "valset", "obs_seq_len"), 2),
        _check("dataset_window", _get(trainer, "trainset", "window_size"), 17),
        _check("valset_window", _get(trainer, "valset", "window_size"), 17),
        _check("epochs", epochs, 100),
        _check("trainer_epochs", trainer.get("epoch"), 100),
        _check("sim_eval_period", sim_period, 0),
        _check("trainer_sim_eval_period", trainer.get("sim_eval_every_n_epochs"), 0),
        _check("isolated_sim_eval_episodes", isolated_episodes, 50),
        _check("final_sim_eval_epoch", final_sim_epoch, 100),
        _check("runner_skips_duplicate_final_sim", config.get("skip_final_sim"), True),
        _check("final_rollout_sets", final_rollout_sets, 1),
        _check("rollouts_per_task", sim.get("num_episode"), 50),
        _check("configured_rollouts_per_task", _get(config, "simulation", "num_episode"), 50),
        _check("source_camera_size", [_get(config, "img_height"), _get(config, "img_width")], [128, 128]),
        _check("source_rgb_shapes", rgb_shapes, [[3, 128, 128]] * 3),
        _check("student_preprocessing_size", obs.get("resize_shape"), [256, 256]),
        _check("policy_type", policy.get("_target_"), "agents.models.droid.diffusion_policy.DroidDiffusionPolicy"),
        _check("readout_type", obs.get("_target_"), "agents.encoders.robot_dift_candidate_obs_encoder.RobotDIFTCandidateObsEncoder"),
        _check("student_type", student.get("_target_"), "agents.encoders.robot_dift_student_feature_extractor.RobotDIFTStudentFeatureExtractor"),
        _check("readout_feature_taps", obs.get("feature_keys"), FEATURE_KEYS),
        _check("student_feature_taps", student.get("feature_keys"), FEATURE_KEYS),
        _check("stage1_fusion_initialized", obs.get("pretrained_fusion_checkpoint"), student.get("checkpoint_dir")),
        _check("clip_query_matches_policy_goal", clip_query == clip_goal, True),
        _check("clip_query_is_local_path", isinstance(clip_query, str) and Path(clip_query).is_absolute(), True),
    ]
    if float(obs.get("feature_cache_max_gib", 0) or 0) > 0:
        checks.append(_check("cache_has_frame_ids", _get(trainer, "trainset", "return_frame_ids"), True))
    checks.extend(implementation_checks())
    return checks


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if manifest.get("schema_version") != 1 or not isinstance(tasks, list) or len(tasks) != 24:
        raise ValueError("RoboCasa manifest must contain exactly 24 tasks with schema_version=1")
    names = [item["name"] for item in tasks]
    files = [item["relative_hdf5"] for item in tasks]
    if len(set(names)) != 24 or len(set(files)) != 24:
        raise ValueError("RoboCasa manifest has duplicate names or paths")
    for name, relative in zip(names, files):
        path = Path(relative)
        if (
            not name.isidentifier()
            or path.is_absolute()
            or ".." in path.parts
            or name not in path.parts
            or path.name != "demo_gentex_im128_randcams.hdf5"
        ):
            raise ValueError(f"invalid RoboCasa manifest entry: {name!r} {relative!r}")
    return manifest


def prerequisite_checks(config: dict[str, Any], manifest: dict[str, Any], work_root: Path) -> list[dict[str, Any]]:
    dataset_root = (work_root / manifest["dataset_relative_root"]).resolve()
    configured_dataset = Path(config["dataset_path"]).resolve()
    clip = Path(config["agents"]["obs_encoders"]["clip_model_path"])
    student = config["agents"]["obs_encoders"]["rgb_model"]
    stage1 = Path(student["checkpoint_dir"])
    sd21 = Path(student["model_repo"])
    checks = [
        {"name": "canonical_dataset_root", "path": str(configured_dataset), "status": "FOUND" if configured_dataset == dataset_root else "MISSING"},
        {"name": "local_clip_vit_b32", "path": str(clip), "status": "FOUND" if clip.is_file() else "MISSING"},
        {"name": "local_sd21", "path": str(sd21), "status": "FOUND" if (sd21 / "model_index.json").is_file() else "MISSING"},
    ]
    for item in manifest["tasks"]:
        path = dataset_root / item["relative_hdf5"]
        checks.append({
            "name": f"robocasa_task:{item['name']}",
            "path": str(path),
            "status": "FOUND" if path.is_file() else "MISSING",
        })
    components = (stage1 / "metadata.json", stage1 / "timestep.bin")
    unet_dir = stage1 / "unet"
    weights = (unet_dir / "diffusion_pytorch_model.bin", unet_dir / "model.safetensors", unet_dir / "diffusion_pytorch_model.safetensors")
    checks.append({
        "name": "stage1_student_checkpoint",
        "path": str(stage1),
        "status": "FOUND" if all(path.is_file() for path in components) and any(path.is_file() for path in weights) else "MISSING",
        "detail": "requires metadata.json, timestep.bin, and one Student U-Net weight file",
    })
    return checks


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest()
        if args.task not in {item["name"] for item in manifest["tasks"]}:
            raise ValueError(f"task {args.task!r} is absent from the 24-task manifest")
        command, config = compose_candidate(args.task, args.work_root, preset=args.preset)
        checks = protocol_checks(command, config, args.task)
        numeric_checks = paper_numeric_checks(command, config)
        prerequisites = prerequisite_checks(config, manifest, args.work_root) if args.work_root else []
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired, yaml.YAMLError) as exc:
        print(f"Stage-II protocol audit failed to read inputs: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 2

    protocol_match = all(check["status"] == "MATCH" for check in checks)
    paper_numeric_match = all(check["status"] == "MATCH" for check in numeric_checks)
    preflight_pass = None if args.work_root is None else all(item["status"] == "FOUND" for item in prerequisites)
    result = {
        "architecture_status": "explicit_candidate_not_paper_reproduction",
        "paper_result_reproduced": False,
        "preset": args.preset,
        "task": args.task,
        "task_manifest": str(TASK_MANIFEST),
        "manifest_task_count": len(manifest["tasks"]),
        "launcher": str(LAUNCHERS[args.preset]),
        "command": command,
        "settings_source": "launcher_dry_run_plus_hydra_resolved_config_and_source_inspection",
        "checks": checks,
        "protocol_match": protocol_match,
        "paper_numeric_checks": numeric_checks,
        "paper_numeric_match": paper_numeric_match,
        "prerequisites": prerequisites,
        "preflight_pass": preflight_pass,
        "limitations": [
            "No GPU training, checkpoint provenance, or RoboCasa rollout is checked.",
            "Student freezing and readout trainability are statically inspected; runtime behavior still needs training and inference validation.",
            "RoboCasa HDF5 source images are 128x128; the candidate resizes to 256x256 before the Student. This does not mean native observations are 256x256.",
            "CLIP variant, cross-modal RoPE convention, and readout details are explicit candidate choices because the manuscript does not fully specify them.",
            "The paper says linear LR and EMA power 0.75 but omits the LR endpoint, EMA inverse-gamma/cap, and DDIM inference-step count. The paper preset explicitly uses 0.1, 1/no cap, and four steps respectively.",
            "The reported effective batch is the configured full-group size; this static audit does not execute a distributed optimizer update.",
        ],
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("RoboCasa Stage-II architecture: explicit candidate, paper reproduction unverified")
        print(f"Protocol checks: {sum(item['status'] == 'MATCH' for item in checks)}/{len(checks)} match")
        for item in checks:
            if item["status"] != "MATCH":
                print(f"{item['status']} {item['name']}: actual={item['actual']!r}, expected={item['expected']!r}")
        print(f"Paper numeric checks: {sum(item['status'] == 'MATCH' for item in numeric_checks)}/{len(numeric_checks)} match")
        for item in numeric_checks:
            if item["status"] != "MATCH":
                print(f"{item['status']} {item['name']}: actual={item['actual']!r}, expected={item['expected']!r}")
        if args.work_root:
            print(f"Prerequisites: {sum(item['status'] == 'FOUND' for item in prerequisites)}/{len(prerequisites)} found")
            for item in prerequisites:
                if item["status"] != "FOUND":
                    print(f"{item['status']} {item['name']}: {item['path']}")
    return 0 if protocol_match and (args.preset != "paper" or paper_numeric_match) and preflight_pass is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
