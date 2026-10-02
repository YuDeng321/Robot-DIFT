#!/usr/bin/env python3
"""Compare a trained Robot-DIFT Student with the SD2.1 DIFT it started from.

Does DROID adaptation make the diffusion features better for robots? Every
Student goes through the same probes and is compared with its initialization,
written by scripts/release/interpolate_student.py at alpha 0, so the baseline
uses the same loader, text conditioning, VAE mode, and feature maps:

* fine detail: dense correspondence under known warps (us3/us6/us8) and
  cross-demo contact points on RoboCasa frames;
* objects: leave-one-demo-out target masks on RoboCasa frames;
* robot state and cameras: the held-out DROID multi-camera probe
  (scripts/probes/droid_multiview_probe.py).

Probes whose inputs are not given are skipped, and finished steps are not
rerun. Paired differences (candidate minus initialization) are collected in
summary.md under --output-dir. Stage-II success remains the deciding test.

Example:
    python scripts/probes/compare_students.py \\
        --checkpoint /path/to/encoder/checkpoint-300000-ema \\
        --descriptor-images /path/to/robocasa_descriptor_images \\
        --contact-images /path/to/contact_points --mask-images /path/to/object_masks \\
        --droid-root "$ROBOT_DIFT_DATA_ROOT" --output-dir /path/to/student_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
FEATURE_KEYS = ("us3", "us6", "us8")


@dataclass
class Step:
    name: str
    command: list[str]
    done: Path
    env: dict[str, str] = field(default_factory=dict)


def _python(script: str, *arguments) -> list[str]:
    return [sys.executable, str(ROOT / script), *map(str, arguments)]


def models(args) -> dict[str, Path]:
    """Label -> Student directory; the first entry is the reference initialization."""
    checkpoint = args.checkpoint.expanduser().resolve()
    students = args.output_dir.expanduser().resolve() / "students"
    reference = "init" if args.init_checkpoint else "sd21_dift"
    result = {reference: students / f"{checkpoint.name}-alpha0.00"}
    for alpha in args.alpha:
        result[f"alpha{alpha:.2f}"] = students / f"{checkpoint.name}-alpha{alpha:.2f}"
    result[args.label] = checkpoint
    for value in args.extra:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"--extra expects LABEL=PATH, got {value!r}")
        if label in result:
            raise ValueError(f"Duplicate model label: {label}")
        result[label] = Path(path).expanduser().resolve()
    paths = [str(path) for path in result.values()]
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path in other or other in path:
                raise ValueError(f"Model paths must not contain one another (probe reports match by substring): "
                                 f"{path} / {other}")
    return result


def plan(args) -> list[Step]:
    out = args.output_dir.expanduser().resolve()
    table = models(args)
    labels = list(table)
    reference = labels[0]
    model_env = {"ROBOT_DIFT_MODEL_DIR": str(args.model_repo)}
    interpolated = {0.0: table[reference], **{alpha: table[f"alpha{alpha:.2f}"] for alpha in args.alpha}}
    missing = [alpha for alpha, path in interpolated.items() if not path.exists()] or [0.0]
    interpolate = ["--checkpoint", table[args.label], "--model-repo", args.model_repo,
                   "--alpha", *[f"{alpha:g}" for alpha in missing], "--output-root", out / "students"]
    if args.init_checkpoint:
        interpolate += ["--init-checkpoint", args.init_checkpoint]
    steps = [Step("interpolate", _python("scripts/release/interpolate_student.py", *interpolate),
                  interpolated[missing[0]])]

    if args.descriptor_images:
        for key in FEATURE_KEYS:
            directory = out / "correspondence" / key
            for label, path in table.items():
                steps.append(Step(
                    f"correspondence {key} {label}",
                    _python("scripts/probes/correspondence_repeatability.py", "--image-dir", args.descriptor_images,
                            "--model-spec", ROOT / "configs/correspondence" / f"robot_dift_student_{key}.json",
                            "--checkpoint", path, "--device", args.device, "--max-images", args.max_images,
                            "--output-dir", directory / label),
                    directory / label / "per_pair.csv", model_env,
                ))
            for label in labels[1:]:
                steps.append(Step(
                    f"correspondence {key} {label} vs {reference}",
                    _python("scripts/probes/compare_correspondence_results.py",
                            "--reference", directory / reference / "per_pair.csv",
                            "--candidate", directory / label / "per_pair.csv",
                            "--output", directory / f"{label}_vs_{reference}.json"),
                    directory / f"{label}_vs_{reference}.json",
                ))

    for name, images, probe, compare in (
        ("contact", args.contact_images, "robocasa_contact_point_probe.py", "compare_robocasa_contact_point_probe.py"),
        ("mask", args.mask_images, "robocasa_object_mask_probe.py", "compare_robocasa_object_mask_probe.py"),
    ):
        if not images:
            continue
        report = out / name / "report.json"
        checkpoints = [item for path in table.values() for item in ("--checkpoint", path)]
        steps.append(Step(name, _python(f"scripts/probes/{probe}", "--image-dir", images, "--model-repo",
                                        args.model_repo, *checkpoints, "--device", args.device, "--output", report),
                          report, model_env))
        for label in labels[1:]:
            output = out / name / f"{label}_vs_{reference}.json"
            steps.append(Step(f"{name} {label} vs {reference}",
                              _python(f"scripts/probes/{compare}", "--report", report,
                                      "--reference", table[reference], "--candidate", table[label],
                                      "--output", output),
                              output))

    samples = args.droid_samples or out / "droid" / "frames.npz"
    if args.droid_root or Path(samples).is_file():
        command = [item for label, path in table.items() for item in ("--checkpoint", f"{label}={path}")]
        command += ["--samples", samples, "--model-repo", args.model_repo, "--device", args.device,
                    "--output-dir", out / "droid"]
        if args.droid_root:
            command += ["--data-root", args.droid_root]
        if args.clip_model:
            command += ["--clip-model", args.clip_model]
        steps.append(Step("droid multi-camera", _python("scripts/probes/droid_multiview_probe.py", *command),
                          out / "droid" / "report.json", model_env))
    return steps


def _interval(value: dict) -> tuple[float, float, float]:
    if "mean" in value:
        return value["mean"], value["ci95_low"], value["ci95_high"]
    low, high = value["confidence_interval_95"]
    return value["estimate"], low, high


def _rows(metrics: dict) -> list[str]:
    rows = []
    for metric, value in sorted(metrics.items()):
        if metric == "num_points":
            continue
        estimate, low, high = _interval(value)
        better = "lower" if "error" in metric else "higher"
        evidence = "yes" if (low > 0 or high < 0) else "no"
        rows.append(f"| {metric} | {estimate:+.4f} | [{low:+.4f}, {high:+.4f}] | {better} | {evidence} |")
    return rows


def summarize(out: Path, table: dict[str, Path]) -> str:
    labels = list(table)
    reference = labels[0]
    lines = ["# Robot-DIFT Student comparison", "",
             f"Reference: `{reference}` (the Student's initialization). Differences are candidate minus reference; "
             "'CI excludes 0' marks a 95% interval that does not contain zero.", "",
             "| label | Student |", "|---|---|"]
    lines += [f"| {label} | `{path}` |" for label, path in table.items()]
    header = ["| metric | difference | 95% CI | better if | CI excludes 0 |", "|---|---|---|---|---|"]
    for label in labels[1:]:
        lines += ["", f"## {label} vs {reference}"]
        for key in FEATURE_KEYS:
            path = out / "correspondence" / key / f"{label}_vs_{reference}.json"
            if path.is_file():
                result = json.loads(path.read_text())
                lines += ["", f"### Correspondence under warps, {key} ({result['num_images']} images)", "", *header,
                          *_rows(result["metrics"])]
        for name, title in (("contact", "Cross-demo contact points"), ("mask", "Target-object masks")):
            path = out / name / f"{label}_vs_{reference}.json"
            if path.is_file():
                result = json.loads(path.read_text())
                for key, layer in sorted(result["maps"].items()):
                    lines += ["", f"### {title}, {key}", "", *header, *_rows(layer["metrics"])]
    droid = out / "droid" / "summary.md"
    if droid.is_file():
        lines += ["", droid.read_text().replace("# DROID multi-camera probe", "## DROID multi-camera probe", 1)]
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained Stage-I Student directory")
    parser.add_argument("--label", default="robot_dift")
    parser.add_argument("--extra", action="append", default=[], help="LABEL=PATH of another Student to compare")
    parser.add_argument("--alpha", type=float, action="append", default=[],
                        help="also compare this weight-interpolated Student (0 < alpha < 1); repeatable")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="initialization when the Student was not copied from SD2.1")
    parser.add_argument("--model-repo", type=Path, default=os.environ.get("ROBOT_DIFT_MODEL_DIR"))
    parser.add_argument("--clip-model", default=os.environ.get("ROBOT_DIFT_CLIP_MODEL"))
    parser.add_argument("--descriptor-images", type=Path, help="RoboCasa descriptor images (correspondence)")
    parser.add_argument("--contact-images", type=Path, help="prepared RoboCasa contact-point frames")
    parser.add_argument("--mask-images", type=Path, help="prepared RoboCasa object-mask frames")
    parser.add_argument("--droid-root", default=os.environ.get("ROBOT_DIFT_DATA_ROOT"),
                        help="TFDS data dir with droid/ (default: ROBOT_DIFT_DATA_ROOT)")
    parser.add_argument("--droid-samples", type=Path, help="DROID probe frame cache (default: <output>/droid/frames.npz)")
    parser.add_argument("--max-images", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    args = parser.parse_args(argv)
    if any(not 0.0 < alpha < 1.0 for alpha in args.alpha):
        parser.error("--alpha values must lie strictly between 0 and 1")
    if not args.model_repo:
        parser.error("--model-repo (or ROBOT_DIFT_MODEL_DIR) is required")
    out = args.output_dir.expanduser().resolve()
    if out / "students" == args.checkpoint.expanduser().resolve().parent:
        parser.error("--output-dir/students must not be the directory that holds --checkpoint")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.expanduser().resolve()
    steps = plan(args)
    for step in steps:
        prefix = "".join(f"{key}={shlex.quote(value)} " for key, value in step.env.items())
        status = "done" if step.done.exists() else "todo"
        print(f"[{status}] {step.name}\n    {prefix}{shlex.join(step.command)}")
    if args.dry_run:
        return 0
    for step in steps:
        if step.done.exists():
            continue
        # Child scripts run by absolute path, so Python otherwise starts with
        # scripts/probes on sys.path rather than the repository package root.
        child_env = {**os.environ, **step.env}
        child_env["PYTHONPATH"] = str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")
        subprocess.run(step.command, cwd=ROOT, env=child_env, check=True)
    summary = summarize(out, models(args))
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
