#!/usr/bin/env python3
"""Check every file in a portable Robot-DIFT Student package against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_FILES = {
    "metadata.json",
    "training_metadata.json",
    "timestep.bin",
    "deploy_head.safetensors",
}
UNET_WEIGHTS = {
    "unet/diffusion_pytorch_model.bin",
    "unet/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_paths(value, where="") -> list[str]:
    """Find machine-local paths in metadata, including the embedded config."""
    if isinstance(value, dict):
        return [path for key, item in value.items() for path in local_paths(item, f"{where}.{key}")]
    if isinstance(value, list):
        return [path for index, item in enumerate(value) for path in local_paths(item, f"{where}[{index}]")]
    if isinstance(value, str):
        if where.endswith(".training_config"):
            return local_paths(json.loads(value), where)
        if Path(value).is_absolute() or value.startswith("~/"):
            return [where]
    return []


def verify(package: Path, model_repo: Path) -> dict:
    package = package.expanduser().resolve(strict=True)
    manifest_path = package / "release_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing regular release_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "robot_dift_student_encoder":
        raise ValueError("Unsupported Robot-DIFT release manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Release manifest lacks its file table")
    names = set(files)
    if names - UNET_WEIGHTS != EXPECTED_FILES or len(names & UNET_WEIGHTS) != 1:
        raise ValueError("Student package is missing a required file or includes an unexpected component")
    entries = list(package.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Release package must not contain symlinks")
    actual = {str(path.relative_to(package)) for path in entries if path.is_file()}
    if actual != names | {"release_manifest.json"}:
        raise ValueError(f"Package file set differs from manifest: {sorted(actual ^ (names | {'release_manifest.json'}))}")
    for name, recorded in files.items():
        path = package / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent not in {package, package / "unet"}:
            raise ValueError(f"Invalid package file path: {name}")
        if path.stat().st_size != recorded.get("bytes") or sha256(path) != recorded.get("sha256"):
            raise ValueError(f"Package checksum mismatch: {name}")
    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("feature_key") != ["us3", "us6", "us8"]:
        raise ValueError("Package does not expose the release feature taps")
    if metadata.get("components", {}).get("adapters") is not None:
        raise ValueError("Training-only alignment adapters remain in deployment metadata")
    if sha256(package / "training_metadata.json") != manifest.get("packaged_training_metadata_sha256"):
        raise ValueError("Packaged training metadata hash mismatch")
    private_fields = local_paths(manifest, "manifest") + local_paths(metadata, "metadata")
    private_fields += local_paths(
        json.loads((package / "training_metadata.json").read_text(encoding="utf-8")),
        "training_metadata",
    )
    if private_fields:
        raise ValueError(f"Release package contains machine-local paths in {private_fields}")
    model_index = model_repo.expanduser().resolve(strict=True) / "model_index.json"
    if model_index.is_symlink() or not model_index.is_file():
        raise ValueError("Missing external SD2.1 model_index.json")
    if sha256(model_index) != manifest.get("external_sd21_model_index_sha256"):
        raise ValueError("External SD2.1 model index differs from the export")
    return {"package": str(package), "files_verified": len(files), "external_model_index_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.model_repo), indent=2))


if __name__ == "__main__":
    main()
