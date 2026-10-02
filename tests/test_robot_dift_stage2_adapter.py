"""Adapter export/load contract with a shared frozen encoder."""

import json
import pickle

import pytest
import torch
from torch import nn

from agents.utils.scaler import ActionScaler
from release.stage2_adapter import export_stage2_adapter, load_stage2_adapter


class _MockAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = nn.Module()
        self.img_encoder.rgb_model = nn.Linear(2, 2)
        self.img_encoder.rgb_model.register_buffer("scalar", torch.tensor(1.0, dtype=torch.bfloat16))
        self.img_encoder.text_encoder = nn.Linear(2, 2)
        self.img_encoder.readout = nn.Linear(2, 2)
        self.model = nn.Linear(2, 2)
        self.register_buffer("robot_states_min", None)
        self.register_buffer("robot_states_max", None)
        self.scaler = None


def test_stage2_adapter_restores_policy_and_keeps_shared_encoder(tmp_path):
    torch.manual_seed(41)
    trained = _MockAgent()
    trained.robot_states_min = torch.tensor([-1.0, 0.0])
    trained.robot_states_max = torch.tensor([1.0, 2.0])
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_path = full_dir / "last_model.pth"
    torch.save(trained.state_dict(), full_path)
    scaler = ActionScaler(torch.randn(16, 7), True, "cpu")
    with (full_dir / "model_scaler.pkl").open("wb") as stream:
        pickle.dump(scaler, stream)

    stage1 = tmp_path / "encoder"
    (stage1 / "unet").mkdir(parents=True)
    (stage1 / "metadata.json").write_text(
        json.dumps({"fusion_mode": "global_to_fine", "vae_latent_mode": "mode", "ema": True}),
        encoding="utf-8",
    )
    (stage1 / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"student")
    clip = tmp_path / "clip.pt"
    clip.write_bytes(b"clip")

    artifact = tmp_path / "adapter"
    manifest = export_stage2_adapter(full_path, artifact, stage1_checkpoint=stage1, clip_model=clip)
    assert manifest["excluded_frozen_tensor_count"] == 5
    assert manifest["adapter_tensor_bytes"] < full_path.stat().st_size
    assert len(manifest["frozen_state_sha256"]) == 64

    fresh = _MockAgent()
    with pytest.raises(ValueError, match="frozen Student/CLIP"):
        load_stage2_adapter(fresh, artifact, stage1_checkpoint=stage1, clip_model=clip)
    frozen_before = {key: value.clone() for key, value in trained.state_dict().items() if key.startswith(("img_encoder.rgb_model.", "img_encoder.text_encoder."))}
    fresh.load_state_dict(frozen_before, strict=False)
    load_stage2_adapter(fresh, artifact, stage1_checkpoint=stage1, clip_model=clip)
    for key, value in trained.state_dict().items():
        if key in frozen_before:
            torch.testing.assert_close(fresh.state_dict()[key], frozen_before[key])
        else:
            torch.testing.assert_close(fresh.state_dict()[key], value)
    torch.testing.assert_close(
        fresh.scaler.inverse_scale_output(torch.ones(2, 7)),
        scaler.inverse_scale_output(torch.ones(2, 7)),
    )
    clip.write_bytes(b"different clip")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_stage2_adapter(_MockAgent(), artifact, stage1_checkpoint=stage1, clip_model=clip)
