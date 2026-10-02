"""Student comparison driver: probe plan, model path checks, and the paired summary."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("compare_students", ROOT / "scripts/probes/compare_students.py")
compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare
_SPEC.loader.exec_module(compare)


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "encoder" / "checkpoint-300000-ema"
    checkpoint.mkdir(parents=True)
    return checkpoint


def _args(tmp_path, *extra):
    return compare.parse_args(["--checkpoint", str(_checkpoint(tmp_path)), "--model-repo", str(tmp_path / "sd21"),
                               "--output-dir", str(tmp_path / "out"), *extra])


def test_plan_runs_every_probe_for_every_student(tmp_path):
    old = tmp_path / "old_run" / "checkpoint-120000-ema"
    old.mkdir(parents=True)
    args = _args(tmp_path, "--alpha", "0.5", "--extra", f"old={old}", "--descriptor-images", str(tmp_path / "d"),
                 "--contact-images", str(tmp_path / "c"), "--mask-images", str(tmp_path / "m"),
                 "--droid-root", str(tmp_path / "droid"), "--clip-model", str(tmp_path / "clip.pt"))
    steps = compare.plan(args)
    names = [step.name for step in steps]
    assert names[0] == "interpolate"
    assert steps[0].command[steps[0].command.index("--alpha") + 1:][:2] == ["0", "0.5"]
    assert sum(name.startswith("correspondence") and " vs " not in name for name in names) == 3 * 4
    assert sum(name.startswith("correspondence") and " vs " in name for name in names) == 3 * 3
    assert names.count("contact") == names.count("mask") == 1
    assert sum(name.startswith(("contact ", "mask ")) for name in names) == 6
    contact = steps[names.index("contact")]
    assert contact.command.count("--checkpoint") == 4 and contact.env == {"ROBOT_DIFT_MODEL_DIR": str(tmp_path / "sd21")}
    droid = steps[names.index("droid multi-camera")].command
    labels = [droid[i + 1].split("=")[0] for i, item in enumerate(droid) if item == "--checkpoint"]
    assert labels == ["sd21_dift", "alpha0.50", "robot_dift", "old"]
    assert "--clip-model" in droid and "--data-root" in droid


def test_probes_without_inputs_are_skipped_and_finished_steps_reused(tmp_path):
    samples = tmp_path / "frames.npz"
    samples.write_bytes(b"cached")
    args = _args(tmp_path, "--droid-samples", str(samples))
    args.droid_root = None
    steps = compare.plan(args)
    assert [step.name for step in steps] == ["interpolate", "droid multi-camera"]
    assert not steps[0].done.exists()
    steps[0].done.mkdir(parents=True)
    assert compare.plan(args)[0].done.exists()


def test_nested_student_paths_are_rejected(tmp_path):
    nested = tmp_path / "encoder" / "checkpoint-300000-ema-old"
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not contain one another"):
        compare.plan(_args(tmp_path, "--extra", f"old={nested}"))
    with pytest.raises(SystemExit):
        _args(tmp_path / "second", "--alpha", "1.0")


def test_summary_collects_paired_differences(tmp_path):
    args = _args(tmp_path)
    out = args.output_dir.resolve()
    (out / "correspondence" / "us6").mkdir(parents=True)
    (out / "correspondence" / "us6" / "robot_dift_vs_sd21_dift.json").write_text(json.dumps({
        "num_images": 96,
        "metrics": {
            "pck@4px": {"mean": 0.05, "ci95_low": 0.02, "ci95_high": 0.08},
            "mean_error_px": {"mean": -1.2, "ci95_low": -2.0, "ci95_high": -0.3},
            "num_points": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
        },
    }))
    (out / "contact").mkdir()
    (out / "contact" / "robot_dift_vs_sd21_dift.json").write_text(json.dumps({"maps": {"us8": {"pairs": 12, "metrics": {
        "pck8_difference": {"estimate": 0.1, "confidence_interval_95": [-0.01, 0.2]},
    }}}}))
    (out / "droid").mkdir()
    (out / "droid" / "summary.md").write_text("# DROID multi-camera probe\n\nbody\n")
    summary = compare.summarize(out, compare.models(args))
    assert "| pck@4px | +0.0500 | [+0.0200, +0.0800] | higher | yes |" in summary
    assert "| mean_error_px | -1.2000 | [-2.0000, -0.3000] | lower | yes |" in summary
    assert "| pck8_difference | +0.1000 | [-0.0100, +0.2000] | higher | no |" in summary
    assert "num_points" not in summary and "## DROID multi-camera probe" in summary


def test_dry_run_lists_the_steps(tmp_path, capsys):
    assert compare.main(["--checkpoint", str(_checkpoint(tmp_path)), "--model-repo", str(tmp_path / "sd21"),
                         "--droid-root", str(tmp_path / "droid"), "--output-dir", str(tmp_path / "out"),
                         "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "[todo] interpolate" in printed and "[todo] droid multi-camera" in printed
    assert "ROBOT_DIFT_MODEL_DIR=" in printed
