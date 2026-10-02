#!/usr/bin/env python3
"""Validate, enumerate, or locally export the Robot-DIFT source candidate.

Export copies only manifest-selected files into a new local directory. It does
not modify the research tree or publish anything.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("release/source_manifest.json")

# These checks are intentionally independent of the manifest's exclusion list.
# A modified manifest must not be able to select local datasets or weights.
BLOCKED_DIRECTORY_NAMES = {
    ".local", "checkpoints", "data", "datasets", "logs", "outputs",
    "runs", "wandb", "weights",
}
BLOCKED_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".hdf5", ".npy", ".npz", ".parquet",
    ".pkl", ".pt", ".pth", ".safetensors", ".tfrecord", ".tfrecords",
}
SOURCE_SUFFIXES = {".cff", ".json", ".md", ".png", ".py", ".sh", ".svg", ".txt", ".yaml"}
SOURCE_NAMES_WITHOUT_SUFFIX = {".gitignore", "LICENSE"}
TOKENIZER_RESOURCE = "agents/models/beso/utils/bpe_simple_vocab_16e6.txt.gz"


def _safe_relative(pattern: str) -> bool:
    path = PurePosixPath(pattern)
    return (
        bool(pattern)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in pattern
    )


def _matching(path: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if fnmatch.fnmatchcase(path, pattern):
            return pattern
    return None


def validate(manifest_path: Path, repo_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    files: set[str] = set()
    repo_root = repo_root.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Cannot read manifest: {exc}"]

    if manifest.get("schema_version") != 1:
        errors.append("Unsupported schema_version; expected 1")

    entrypoints = manifest.get("entrypoints")
    source_groups = manifest.get("source_groups")
    artifact_excludes = manifest.get("artifact_exclude_globs")
    research_excludes = manifest.get("optional_research_exclude_globs")
    if not isinstance(entrypoints, dict) or not entrypoints:
        errors.append("entrypoints must be a nonempty object")
        entrypoints = {}
    if not isinstance(source_groups, dict) or not source_groups:
        errors.append("source_groups must be a nonempty object")
        source_groups = {}
    for name, patterns in (("artifact_exclude_globs", artifact_excludes),
                           ("optional_research_exclude_globs", research_excludes)):
        if not isinstance(patterns, list) or not all(isinstance(p, str) and _safe_relative(p) for p in patterns):
            errors.append(f"{name} must contain safe relative patterns")
    artifact_excludes = [p for p in artifact_excludes if isinstance(p, str)] if isinstance(artifact_excludes, list) else []
    research_excludes = [p for p in research_excludes if isinstance(p, str)] if isinstance(research_excludes, list) else []

    for section_name, groups, allow_globs in (
        ("entrypoints", entrypoints, False),
        ("source_groups", source_groups, True),
    ):
        for group_name, patterns in groups.items():
            if not isinstance(patterns, list) or not patterns:
                errors.append(f"{section_name}.{group_name} must be a nonempty list")
                continue
            for pattern in patterns:
                label = f"{section_name}.{group_name}: {pattern!r}"
                if not isinstance(pattern, str) or not _safe_relative(pattern):
                    errors.append(f"Unsafe path in {label}")
                    continue
                if not allow_globs and any(char in pattern for char in "*?["):
                    errors.append(f"Entrypoint must be an exact file: {label}")
                    continue
                matches = list(repo_root.glob(pattern)) if allow_globs else [repo_root / pattern]
                if not matches:
                    errors.append(f"No source file matches {label}")
                for file_path in matches:
                    if not file_path.is_file():
                        errors.append(f"Missing or non-file source in {label}: {file_path}")
                        continue
                    resolved = file_path.resolve()
                    if not resolved.is_relative_to(repo_root):
                        errors.append(f"Source escapes repository in {label}: {file_path}")
                        continue
                    relative = file_path.relative_to(repo_root).as_posix()
                    if any(part.lower() in BLOCKED_DIRECTORY_NAMES for part in file_path.relative_to(repo_root).parts[:-1]):
                        errors.append(f"Data/output directory cannot be released: {relative}")
                    if file_path.suffix.lower() in BLOCKED_SUFFIXES or ".tfrecord" in file_path.name.lower():
                        errors.append(f"Data/checkpoint file cannot be released: {relative}")
                    if not (
                        file_path.suffix.lower() in SOURCE_SUFFIXES
                        or file_path.name in SOURCE_NAMES_WITHOUT_SUFFIX
                        or relative == TOKENIZER_RESOURCE
                    ):
                        errors.append(f"Unsupported source file type: {relative}")
                    blocked_pattern = _matching(relative, artifact_excludes)
                    if blocked_pattern:
                        errors.append(f"Artifact exclusion {blocked_pattern!r} matches {relative}")
                    research_pattern = _matching(relative, research_excludes)
                    if research_pattern:
                        errors.append(f"Research exclusion {research_pattern!r} matches {relative}")
                    files.add(relative)

    external_assets = manifest.get("external_assets")
    if not isinstance(external_assets, list) or not external_assets:
        errors.append("external_assets must list required out-of-repository resources")
    else:
        for index, asset in enumerate(external_assets):
            if not isinstance(asset, dict) or not asset.get("role") or not asset.get("kind"):
                errors.append(f"external_assets[{index}] must have role and kind")
                continue
            path = asset.get("relative_to_work_root")
            env = asset.get("environment_variable")
            if (path is None) == (env is None):
                errors.append(f"external_assets[{index}] must define exactly one location")
            if path is not None and (not isinstance(path, str) or not _safe_relative(path)):
                errors.append(f"external_assets[{index}] has unsafe relative path")

    runtime_modules = manifest.get("external_runtime_modules")
    if not isinstance(runtime_modules, dict) or not runtime_modules:
        errors.append("external_runtime_modules must identify external packages")
    else:
        for stage, modules in runtime_modules.items():
            if not isinstance(modules, list) or not modules or not all(isinstance(module, str) and module for module in modules):
                errors.append(f"external_runtime_modules.{stage} must be a nonempty module list")

    return sorted(files), errors


def export_source(files: list[str], repo_root: Path, destination: Path) -> None:
    """Copy validated source files to a new or empty directory atomically."""
    repo_root = repo_root.resolve()
    destination = destination.expanduser().absolute()
    if ".." in destination.parts:
        raise ValueError(f"Export path must not contain '..': {destination}")
    if destination.resolve().is_relative_to(repo_root):
        raise ValueError(f"Export destination must be outside the source repository: {destination}")
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError(f"Export destination traverses a symlink: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"Export parent directory does not exist: {destination.parent}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Export destination must be absent or empty: {destination}")

    for relative in files:
        current = repo_root
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Source release path traverses a symlink: {relative}")
        if not current.is_file() or not current.resolve().is_relative_to(repo_root):
            raise ValueError(f"Source release file is missing or escapes the repository: {relative}")

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        for relative in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / relative, target)
        if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
            raise ValueError(f"Export destination changed or became nonempty: {destination}")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list-files", action="store_true", help="print release candidate paths")
    action.add_argument(
        "--export-dir", type=Path, metavar="DEST",
        help="copy only selected sources into a new or empty directory outside the repository",
    )
    args = parser.parse_args()

    files, errors = validate(args.manifest, args.repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Source manifest invalid: {len(errors)} error(s)", file=sys.stderr)
        return 1
    if args.list_files:
        print("\n".join(files))
    elif args.export_dir:
        try:
            export_source(files, args.repo_root, args.export_dir)
        except (OSError, ValueError) as exc:
            print(f"Source export failed: {exc}", file=sys.stderr)
            return 1
        print(f"Source export ready: {len(files)} files at {args.export_dir.absolute()}")
    else:
        print(f"Source manifest valid: {len(files)} files; datasets and weights remain external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
