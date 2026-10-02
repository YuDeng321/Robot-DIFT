"""Stage-II Student input resolution: launcher switch, resolved Hydra config, resolution-agnostic readout."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from agents.encoders.robot_dift_candidate_obs_encoder import RobotDIFTCandidateObsEncoder

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "paper": ROOT / "scripts/release/stage2_robocasa_paper_protocol.sh",
    "robocasa": ROOT / "scripts/release/stage2_robocasa_candidate.sh",
    "libero": ROOT / "scripts/release/stage2_libero_candidate.sh",
}


def _environment(**overrides):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ROBOT_DIFT_")}
    environment.update(overrides)
    return environment


def _dry_run(launcher, **overrides):
    return subprocess.run(
        ["bash", str(LAUNCHERS[launcher]), "--dry-run"],
        cwd=ROOT, env=_environment(**overrides), capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("launcher", sorted(LAUNCHERS))
def test_default_launch_keeps_the_paper_resolution(launcher):
    result = _dry_run(launcher)
    assert result.returncode == 0, result.stderr
    assert "resize_shape" not in result.stdout


@pytest.mark.parametrize("launcher", sorted(LAUNCHERS))
def test_image_size_switch_adds_one_hydra_override(launcher):
    result = _dry_run(launcher, ROBOT_DIFT_STAGE2_IMAGE_SIZE="384", ROBOT_DIFT_ABLATION="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("resize_shape") == 1
    assert r"agents.obs_encoders.resize_shape=\[384\,384\]" in result.stdout


def test_paper_preset_needs_the_ablation_flag_and_sizes_are_validated():
    result = _dry_run("paper", ROBOT_DIFT_STAGE2_IMAGE_SIZE="384")
    assert result.returncode == 2 and "ROBOT_DIFT_ABLATION=1" in result.stderr
    for size in ("300", "0", "abc", "2048"):
        result = _dry_run("libero", ROBOT_DIFT_STAGE2_IMAGE_SIZE=size)
        assert result.returncode == 2 and "multiple of 64" in result.stderr, size


def test_protocol_checker_resolves_the_override_and_flags_only_the_resolution():
    command = [sys.executable, str(ROOT / "scripts/release/check_stage2_protocol.py"), "--preset", "paper", "--json"]
    default = subprocess.run(command, cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=300)
    assert default.returncode == 0, default.stderr[-2000:]
    ablation = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=300,
        env=_environment(ROBOT_DIFT_STAGE2_IMAGE_SIZE="384", ROBOT_DIFT_ABLATION="1"),
    )
    report = json.loads(ablation.stdout)
    differences = [(check["name"], check["actual"]) for check in report["checks"] if check["status"] != "MATCH"]
    assert differences == [("student_preprocessing_size", [384, 384])]


class _ScaledStudent(nn.Module):
    """Maps shrink with the input like SD2.1: us3 at 1/32, us6 and us8 at 1/16."""

    feature_dims = {"us3": 4, "us6": 4, "us8": 4}

    def _encode_backbone(self, images, captions):
        self.input_size = tuple(images.shape[-2:])
        base = images.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return {"us3": F.avg_pool2d(base, 32), "us6": F.avg_pool2d(base, 16), "us8": F.avg_pool2d(base, 16)}


class _Text(nn.Module):
    def forward(self, captions):
        mask = torch.zeros(len(captions), 77, dtype=torch.bool)
        mask[:, :4] = True
        return torch.ones(len(captions), 77, 512), mask


def _encoder(size, student):
    meta = {"obs": {
        "agentview_image": {"shape": [3, 128, 128], "type": "rgb"},
        "eye_in_hand_image": {"shape": [3, 128, 128], "type": "rgb"},
    }}
    return RobotDIFTCandidateObsEncoder(
        meta, student, clip_model_path="unused-in-injected-test", text_encoder=_Text(), output_dim=12,
        resize_shape=(size, size), fpn_dim=8, model_dim=16, num_heads=2, mlp_hidden_dims=(16,),
    )


def test_readout_weights_transfer_across_student_resolutions():
    torch.manual_seed(0)
    low_student, high_student = _ScaledStudent(), _ScaledStudent()
    low, high = _encoder(256, low_student), _encoder(384, high_student)
    high.readout.load_state_dict(low.readout.state_dict())
    observations = {key: torch.rand(2, 3, 128, 128) for key in ("agentview_image", "eye_in_hand_image")}
    low_tokens, _ = low(observations, ["open the drawer", "press the button"])
    high_tokens, _ = high(observations, ["open the drawer", "press the button"])
    assert (low_student.input_size, high_student.input_size) == ((256, 256), (384, 384))
    assert low_tokens.shape == high_tokens.shape == (2, 1, 1, 12)
    assert torch.isfinite(high_tokens).all() and not torch.equal(low_tokens, high_tokens)
