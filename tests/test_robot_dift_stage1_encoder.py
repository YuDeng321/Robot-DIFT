"""Paper Stage-I encoder: view/frame ordering, one Student pass per image, readout export."""

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from agents.encoders.robot_dift_deploy_head import (
    DEPLOY_HEAD_FILE,
    collect_deploy_head_state,
    load_stage1_global_to_fine_fusion,
    load_stage1_paper_readout,
)
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout
from agents.encoders.robot_dift_stage1_encoder import RobotDIFTStage1Encoder

CHANNELS = {"us3": 4, "us6": 4, "us8": 4}


class _FakeStudent(nn.Module):
    """Maps encode each image's constant value so ordering can be checked."""

    feature_dims = dict(CHANNELS)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def encode_student(self, x, captions, feature_keys, alignment=False, return_raw_cosine=False):
        assert len(captions) == x.shape[0]
        self.calls.append({"images": x.shape[0], "alignment": alignment, "captions": list(captions)})
        base = x.mean(dim=(1, 2, 3))[:, None, None, None] * self.weight
        maps = {
            "us3": base.expand(-1, 4, 2, 2),
            "us6": base.expand(-1, 4, 4, 4),
            "us8": base.expand(-1, 4, 4, 4),
        }
        loss = x.mean() * 0 + 1.5 if alignment else None
        metrics = {"raw_cosine_us3": torch.tensor(0.5)} if return_raw_cosine else {}
        return {key: maps[key] for key in feature_keys}, loss, metrics

    def get_parameter_groups(self, base_lr, **_):
        return [{"params": [self.weight], "lr": base_lr, "name": "backbone"}]


class _FakeText(nn.Module):
    goal_dim = 6

    def forward(self, captions):
        tokens = torch.zeros(len(captions), 77, 512)
        tokens[:, 0, 0] = torch.tensor([float(len(c)) for c in captions])
        mask = torch.zeros(len(captions), 77, dtype=torch.bool)
        mask[:, :3] = True
        return tokens, mask

    def encode_goal(self, captions):
        return torch.tensor([[float(len(c))] * self.goal_dim for c in captions])[:, None, :]


class _CaptureReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, views, tokens, mask):
        self.inputs = (views, tokens, mask)
        return torch.zeros(tokens.shape[0], 12)


def _encoder(**kwargs):
    return RobotDIFTStage1Encoder(
        rgb_keys=["cam_b", "cam_a"],
        student=_FakeStudent(),
        text_encoder=_FakeText(),
        output_dim=12,
        fpn_dim=8,
        model_dim=16,
        num_heads=2,
        mlp_hidden_dims=(16,),
        **kwargs,
    )


def _observations(batch=2, frames=2):
    # value = 100*b + 10*view + t identifies every image
    obs = {}
    for view, key in enumerate(["cam_a", "cam_b"]):
        values = torch.tensor([[100 * b + 10 * view + t for t in range(frames)] for b in range(batch)], dtype=torch.float32)
        obs[key] = values[:, :, None, None, None].expand(batch, frames, 3, 8, 8).clone()
    return obs


@pytest.mark.parametrize("alignment", [False, True])
def test_readout_receives_every_view_in_batch_then_frame_order(alignment):
    encoder = _encoder()
    capture = _CaptureReadout()
    encoder.readout = capture
    features, loss, _ = encoder.encode_sequence(_observations(), ["ab", "abcd"], alignment=alignment)
    assert features.shape == (2, 2, 12)
    views, tokens, mask = capture.inputs
    assert len(views) == 2
    for view in range(2):
        values = views[view]["us3"][:, 0, 0, 0].tolist()
        assert values == [100 * b + 10 * view + t for b in range(2) for t in range(2)]
    assert tokens[:, 0, 0].tolist() == [2.0, 2.0, 4.0, 4.0]
    assert mask.shape == (4, 77)
    assert (loss is not None) == alignment


def test_alignment_uses_only_the_current_frame_and_one_pass_per_image():
    encoder = _encoder()
    encoder.encode_sequence(_observations(), ["ab", "abcd"], alignment=True)
    calls = encoder.student.calls
    assert [(call["images"], call["alignment"]) for call in calls] == [(4, True), (4, False)]
    assert calls[0]["captions"] == ["ab", "ab", "abcd", "abcd"]
    encoder.student.calls.clear()
    encoder.encode_sequence(_observations(), ["ab", "abcd"], alignment=False)
    assert [(call["images"], call["alignment"]) for call in encoder.student.calls] == [(8, False)]


def test_alignment_does_not_change_policy_tokens():
    torch.manual_seed(0)
    encoder = _encoder().eval()
    obs = _observations()
    with torch.no_grad():
        plain, _, _ = encoder.encode_sequence(obs, ["ab", "abcd"], alignment=False)
        merged, loss, metrics = encoder.encode_sequence(obs, ["ab", "abcd"], alignment=True, return_raw_cosine=True)
    torch.testing.assert_close(plain, merged)
    assert loss.item() == pytest.approx(1.5)
    assert metrics["raw_cosine_us3"].item() == pytest.approx(0.5)


def test_single_frame_forward_goal_and_parameter_groups():
    encoder = _encoder()
    obs = {key: value[:, -1] for key, value in _observations().items()}
    assert encoder(obs=obs, lang_cond=["ab", "abcd"]).shape == (2, 12)
    assert encoder.output_shape() == [12]
    goal = encoder.encode_language_goal(["ab", "abcd"])
    assert goal.shape == (2, 6) and goal[1, 0].item() == 4.0
    groups = {group["name"]: group for group in encoder.get_parameter_groups(base_lr=1e-4)}
    assert set(groups) == {"backbone", "head"}
    readout_ids = {id(p) for p in encoder.readout.parameters()}
    assert {id(p) for p in groups["head"]["params"]} == readout_ids
    with pytest.raises(ValueError, match="language instructions"):
        encoder.encode_sequence(_observations(), ["only one"])


def _write_checkpoint(tmp_path, encoder):
    state = collect_deploy_head_state(encoder)
    assert state and all(key.startswith("readout.") for key in state)
    save_file(state, str(tmp_path / DEPLOY_HEAD_FILE))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "readout": "paper",
        "feature_key": ["us3", "us6", "us8"],
        "components": {"deploy_head": DEPLOY_HEAD_FILE},
        "readout_config": encoder.readout_config(),
    }))


def test_stage1_readout_exports_and_loads_into_stage2(tmp_path):
    torch.manual_seed(0)
    encoder = _encoder()
    _write_checkpoint(tmp_path, encoder)

    config = encoder.readout_config()
    full = RobotDIFTPaperReadout(
        CHANNELS, fpn_dim=config["fpn_dim"], model_dim=config["model_dim"],
        output_dim=config["output_dim"], num_heads=config["num_heads"],
        mlp_hidden_dims=tuple(config["mlp_hidden_dims"]),
    )
    assert load_stage1_paper_readout(full, tmp_path) == len(encoder.readout.state_dict())
    for key, value in encoder.readout.state_dict().items():
        torch.testing.assert_close(full.state_dict()[key], value)

    fusion_only = RobotDIFTPaperReadout(CHANNELS, fpn_dim=config["fpn_dim"])
    transferred = load_stage1_global_to_fine_fusion(fusion_only, tmp_path)
    assert transferred == sum(
        1 for key in encoder.readout.state_dict() if key.startswith(("lateral.", "fusions.", "output_fusion."))
    )
    maps = {"us3": torch.randn(1, 4, 2, 2), "us6": torch.randn(1, 4, 4, 4), "us8": torch.randn(1, 4, 4, 4)}
    encoder.readout.eval()
    fusion_only.eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder.readout._fuse_view(maps), fusion_only._fuse_view(maps))


def test_non_finite_readout_is_rejected(tmp_path):
    encoder = _encoder()
    _write_checkpoint(tmp_path, encoder)
    state = collect_deploy_head_state(encoder)
    first = next(iter(state))
    state[first] = torch.full_like(state[first], float("nan"))
    save_file(state, str(tmp_path / DEPLOY_HEAD_FILE))
    target = RobotDIFTPaperReadout(CHANNELS, fpn_dim=8)
    with pytest.raises(ValueError, match="Nonfinite"):
        load_stage1_global_to_fine_fusion(target, tmp_path)
