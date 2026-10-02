import json

import pytest

from scripts.release.compare_stage2_transfer_probe import compare, read_episode_parts, read_episodes


def _episode(index, success, *, state="same", style=9):
    return {
        "task": "CoffeePressButton",
        "episode": index,
        "layout_id": 1,
        "style_id": style,
        "obj_instance_split": None,
        "seed": 42,
        "reset_state_sha256": f"{state}{index}",
        "policy_seed": 42 + index,
        "steps": 100,
        "success": success,
    }


def test_exact_episode_pairing_and_unverified_state_are_distinguished(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text("".join(json.dumps(_episode(i, i == 0)) + "\n" for i in range(4)))
    right.write_text("".join(json.dumps(_episode(i, i < 3)) + "\n" for i in range(4)))
    report = compare(read_episodes(left, 4), read_episodes(right, 4), 1000, 0)
    assert report["difference_interval_method"] == "paired_episode_bootstrap"
    assert report["candidate_minus_reference_success_rate"] == 0.5
    assert report["reference"]["successes"] == 1
    assert report["candidate"]["successes"] == 3
    assert report["paired_outcomes"]["candidate_only_success"] == 2
    assert report["paired_outcomes"]["reference_only_success"] == 0
    assert report["paired_outcomes"]["exact_mcnemar_two_sided_p"] == 0.5

    right.write_text("".join(json.dumps(_episode(i, i < 3, state="different")) + "\n" for i in range(4)))
    report = compare(read_episodes(left, 4), read_episodes(right, 4), 1000, 0)
    assert report["difference_interval_method"] == "independent_episode_bootstrap"
    assert not report["exact_reset_state_matches"]


def test_mismatched_scene_or_incomplete_log_cannot_be_compared(tmp_path):
    left = {0: _episode(0, False)}
    right = {0: _episode(0, True, style=10)}
    with pytest.raises(ValueError, match="style_id"):
        compare(left, right, 100, 0)

    log = tmp_path / "run.out"
    log.write_text("[SimScene] env=CoffeePressButton episode=0 layout_id=1 style_id=9 obj_instance_split=None\n")
    with pytest.raises(ValueError, match="Incomplete"):
        read_episodes(log, 1)


def test_resumed_rollout_parts_require_disjoint_complete_episode_ids(tmp_path):
    first = tmp_path / "first.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    first.write_text(json.dumps(_episode(0, False)) + "\n")
    resumed.write_text("".join(json.dumps(_episode(i, True)) + "\n" for i in (1, 2)))
    assert set(read_episode_parts([first, resumed], 3)) == {0, 1, 2}

    with pytest.raises(ValueError, match="Repeated episode IDs"):
        read_episode_parts([first, first], 3)
    with pytest.raises(ValueError, match="incomplete"):
        read_episode_parts([first, resumed], 4)
