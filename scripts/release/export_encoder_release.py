#!/usr/bin/env python3
"""Export only the deployable Robot-DIFT Student and fusion head.

The SD2.1 VAE, tokenizer, and text encoder remain external dependencies. The
original Stage-I metadata is identified by a SHA-256 digest. The packaged copy
preserves training settings while replacing machine-local paths with portable
asset names. The deploy metadata omits training-only Teacher adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


UNET_WEIGHT_NAMES = (
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    return path


def current_git_revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return head, dirty


PUBLIC_PATHS = {
    "sd_model_repo": "${ROBOT_DIFT_MODEL_DIR}",
    "data_path": "${ROBOT_DIFT_DATA_ROOT}",
    "clip_model_path": "${ROBOT_DIFT_CLIP_MODEL}",
    "save_cleandift_dir": "${ENCODER_OUTPUT_DIR}",
    "checkpoint_dir": "${POLICY_CHECKPOINT_DIR}",
    "output_dir": "${RUN_LOG_DIR}",
}


def public_training_metadata(metadata: dict) -> dict:
    """Preserve hyperparameters while removing local filesystem locations."""
    def clean(value, key=""):
        if isinstance(value, dict):
            return {name: clean(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [clean(item, key) for item in value]
        if isinstance(value, str):
            if key == "training_config":
                try:
                    config = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError("Stage-I training_config must be valid JSON") from exc
                return json.dumps(clean(config), indent=2)
            if Path(value).is_absolute() or value.startswith("~/"):
                return PUBLIC_PATHS.get(key, "${LOCAL_ASSET_PATH}")
        return value

    return clean(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve(strict=True)
    model_repo = args.model_repo.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Release output must be a new path: {output}")
    if source == output or source in output.parents:
        raise ValueError("Release output cannot be inside its source checkpoint")
    model_index = require_regular(model_repo / "model_index.json")
    source_metadata = require_regular(source / "metadata.json")
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata.get("model_type") != "cleandift" or metadata.get("sd_version") != "sd21":
        raise ValueError("Expected an SD2.1 Robot-DIFT Student checkpoint")
    if metadata.get("components", {}).get("student_unet", "").strip("/") != "unet":
        raise ValueError("Checkpoint metadata does not identify its Student U-Net")
    if metadata.get("feature_key") != ["us3", "us6", "us8"]:
        raise ValueError("Release checkpoint must expose us3/us6/us8")

    weights = [source / "unet" / name for name in UNET_WEIGHT_NAMES if (source / "unet" / name).is_file()]
    if len(weights) != 1:
        raise ValueError(f"Expected exactly one Student UNet weight file; found {weights}")
    copied = [
        require_regular(weights[0]),
        require_regular(source / "timestep.bin"),
        require_regular(source / "deploy_head.safetensors"),
    ]
    plan = {
        "source_checkpoint": str(source),
        "output": str(output),
        "files": [{"path": str(path.relative_to(source)), "bytes": path.stat().st_size} for path in copied],
        "external_model_repo": str(model_repo),
        "estimated_bytes": sum(path.stat().st_size for path in copied),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for path in copied:
            destination = temporary / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if destination.stat().st_size != path.stat().st_size:
                raise OSError(f"Incomplete Student weight copy: {destination}")

        public_metadata = public_training_metadata(metadata)
        (temporary / "training_metadata.json").write_text(
            json.dumps(public_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        deploy_metadata = dict(public_metadata)
        deploy_metadata["components"] = {
            "student_unet": "unet/",
            "timestep": "timestep.bin",
            "deploy_head": "deploy_head.safetensors",
            "adapters": None,
            "mapping_network": None,
            "token_mapper": None,
            "full_state_dict": None,
            "robot_dift_encoder_state": None,
        }
        deploy_metadata["release_package"] = {
            "training_metadata": "training_metadata.json",
            "training_only_alignment_adapters_included": False,
            "requires_external_sd21_vae_tokenizer_text_encoder": True,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(deploy_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        root = Path(__file__).resolve().parents[2]
        head, dirty = current_git_revision(root)
        manifest = {
            "schema_version": 1,
            "kind": "robot_dift_student_encoder",
            "source_checkpoint": source.name,
            "source_training_metadata_sha256": sha256(source_metadata),
            "packaged_training_metadata_sha256": sha256(temporary / "training_metadata.json"),
            "source_manifest_sha256": sha256(require_regular(root / "release/source_manifest.json")),
            "source_git_revision": head,
            "source_worktree_dirty": dirty,
            "external_sd21_model_repo": "${ROBOT_DIFT_MODEL_DIR}",
            "external_sd21_model_index_sha256": sha256(model_index),
            "files": {},
        }
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(temporary))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        (temporary / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "files": sorted(manifest["files"]),
                      "bytes": sum(item["bytes"] for item in manifest["files"].values())}, indent=2))


if __name__ == "__main__":
    main()
