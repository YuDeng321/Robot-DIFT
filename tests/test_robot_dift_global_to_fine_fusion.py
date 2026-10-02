"""A trained Stage-I fusion must transfer exactly to the Stage-II readout."""

import json

import torch
from safetensors.torch import save_file

from agents.encoders.cleandift_img_encoder import S2FPNGlobalToFineFusion
from agents.encoders.robot_dift_deploy_head import (
    DEPLOY_HEAD_FILE,
    collect_deploy_head_state,
    load_stage1_global_to_fine_fusion,
    validate_deploy_head,
)
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout


def test_stage1_fusion_weights_transfer_to_stage2_with_output_parity():
    torch.manual_seed(4)
    keys = ["us3", "us6", "us8"]
    channels = {"us3": 8, "us6": 16, "us8": 8}
    stage1 = S2FPNGlobalToFineFusion(channels, keys, fpn_dim=16, device="cpu")
    stage2 = RobotDIFTPaperReadout(
        channels, feature_keys=keys, fpn_dim=16, model_dim=16,
        output_dim=16, num_heads=4, mlp_hidden_dims=(32,),
    )
    state = stage1.state_dict()
    target = stage2.state_dict()
    assert state
    assert set(state).issubset(target)
    target.update(state)
    stage2.load_state_dict(target, strict=True)

    maps = {
        "us3": torch.randn(2, 8, 2, 2),
        "us6": torch.randn(2, 16, 4, 4),
        "us8": torch.randn(2, 8, 8, 8),
    }
    assert torch.allclose(stage1(maps), stage2._fuse_view(maps), atol=1e-6)

    stage1(maps).sum().backward()
    assert all(parameter.grad is not None for parameter in stage1.parameters())


def test_compact_head_file_loads_only_matching_fusion(tmp_path):
    class StageI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_fusion = S2FPNGlobalToFineFusion(
                {"us3": 8, "us6": 16, "us8": 8}, ["us3", "us6", "us8"], 16, "cpu"
            )
            self.final_proj = torch.nn.Linear(16, 16)

    source = StageI()
    state = collect_deploy_head_state(source)
    assert all(not key.startswith(("model.", "teacher.", "ae.")) for key in state)
    save_file(state, str(tmp_path / DEPLOY_HEAD_FILE))
    metadata = {
        "fusion_mode": "global_to_fine",
        "feature_key": ["us3", "us6", "us8"],
        "components": {"deploy_head": DEPLOY_HEAD_FILE},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert len(validate_deploy_head(tmp_path)) == len(state)
    target = RobotDIFTPaperReadout(
        {"us3": 8, "us6": 16, "us8": 8}, feature_keys=["us3", "us6", "us8"],
        fpn_dim=16, model_dim=16, output_dim=16, num_heads=4,
        mlp_hidden_dims=(32,),
    )
    count = load_stage1_global_to_fine_fusion(target, tmp_path)
    assert count == len(source.feature_fusion.state_dict())
    for key, value in source.feature_fusion.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)

    metadata["fusion_mode"] = "s2fpn"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    try:
        load_stage1_global_to_fine_fusion(target, tmp_path)
    except ValueError as error:
        assert "global_to_fine" in str(error)
    else:
        raise AssertionError("Legacy fusion must not be loaded as global_to_fine")
