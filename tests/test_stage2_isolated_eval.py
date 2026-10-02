"""An interrupted Stage-II evaluation must combine only matching policy episodes."""

import json

import pytest

from scripts.release.eval_stage2_isolated import merge_episodes


def _episode(directory, index, success, artifact_hash="same-policy"):
    row = {"episode": index, "success": success, "seed": 42, "training_epoch": 100}
    summary = {
        "episodes": 1, "start_episode": index, "successes": int(success),
        "config_sha256": "same-config", "artifact_kind": "trusted_full_checkpoint",
        "artifact_sha256": artifact_hash, "student": "same-student", "task": ["CoffeePressButton"],
        "seed": 42, "style_ids": [9], "training_epoch": 100,
        "inference_scheduler": "ddim", "inference_steps": 4,
    }
    path = directory / f"episode_{index:04d}.jsonl"
    path.write_text(json.dumps(row) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(summary))


def test_merge_keeps_episode_order_and_success_count(tmp_path):
    _episode(tmp_path, 0, False)
    _episode(tmp_path, 1, True)
    output = tmp_path / "combined.jsonl"
    summary = merge_episodes(tmp_path, output, 0, 2)
    assert summary["successes"] == 1
    assert [json.loads(line)["episode"] for line in output.read_text().splitlines()] == [0, 1]
    assert json.loads(output.with_suffix(".summary.json").read_text())["artifact_sha256"] == "same-policy"


def test_merge_rejects_a_different_policy_without_combined_output(tmp_path):
    _episode(tmp_path, 0, False)
    _episode(tmp_path, 1, True, artifact_hash="different-policy")
    output = tmp_path / "combined.jsonl"
    with pytest.raises(ValueError, match="Checkpoint or protocol changed"):
        merge_episodes(tmp_path, output, 0, 2)
    assert not output.exists()
