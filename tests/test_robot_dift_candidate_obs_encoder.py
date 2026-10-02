"""CPU contract check for the Stage-II multi-camera candidate interface."""

import torch
import torch.nn.functional as F
from torch import nn

from agents.encoders.robot_dift_candidate_obs_encoder import RobotDIFTCandidateObsEncoder


class _FakeStudent(nn.Module):
    feature_dims = {"us3": 4, "us6": 4, "us8": 4}

    def _encode_backbone(self, images, captions):
        self.calls = getattr(self, "calls", 0) + 1
        assert len(captions) == images.shape[0]
        self.last_captions = list(captions)
        base = images.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return {
            "us3": F.adaptive_avg_pool2d(base, 2),
            "us6": F.adaptive_avg_pool2d(base, 4),
            "us8": F.adaptive_avg_pool2d(base, 8),
        }


class _FakeFrozenText(nn.Module):
    def forward(self, captions):
        self.last_captions = list(captions)
        tokens = torch.ones(len(captions), 77, 512)
        mask = torch.zeros(len(captions), 77, dtype=torch.bool)
        mask[:, :4] = True
        return tokens, mask


def test_multiview_candidate_returns_one_policy_token_per_frame():
    meta = {
        "obs": {
            "left_image": {"shape": [3, 16, 16], "type": "rgb"},
            "right_image": {"shape": [3, 16, 16], "type": "rgb"},
        }
    }
    encoder = RobotDIFTCandidateObsEncoder(
        meta,
        _FakeStudent(),
        clip_model_path="unused-in-injected-test",
        text_encoder=_FakeFrozenText(),
        output_dim=12,
        resize_shape=(16, 16),
        fpn_dim=8,
        model_dim=16,
        num_heads=2,
        mlp_hidden_dims=(16,),
    )
    observations = {
        "left_image": torch.rand(2, 3, 16, 16),
        "right_image": torch.rand(2, 3, 16, 16),
    }
    output, alignment = encoder(observations, ["press button", "open drawer"])
    assert output.shape == (2, 1, 1, 12)
    assert alignment is None
    assert encoder.output_shape() == (1, 1, 12)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in encoder.readout.parameters())


def test_multiview_candidate_repeats_trajectory_captions_for_flattened_frames():
    meta = {
        "obs": {
            "left_image": {"shape": [3, 16, 16], "type": "rgb"},
            "right_image": {"shape": [3, 16, 16], "type": "rgb"},
        }
    }
    student = _FakeStudent()
    text = _FakeFrozenText()
    encoder = RobotDIFTCandidateObsEncoder(
        meta,
        student,
        clip_model_path="unused-in-injected-test",
        text_encoder=text,
        output_dim=12,
        resize_shape=(16, 16),
        fpn_dim=8,
        model_dim=16,
        num_heads=2,
        mlp_hidden_dims=(16,),
    )
    observations = {
        "left_image": torch.rand(4, 3, 16, 16),
        "right_image": torch.rand(4, 3, 16, 16),
    }
    output, alignment = encoder(observations, ["press button", "open drawer"])
    repeated = ["press button", "press button", "open drawer", "open drawer"]
    assert text.last_captions == repeated
    assert student.last_captions == repeated
    assert output.shape == (4, 1, 1, 12)
    assert alignment is None


def test_frozen_feature_cache_reuses_frames_without_changing_readout_output():
    meta = {
        "obs": {
            "left_image": {"shape": [3, 16, 16], "type": "rgb"},
            "right_image": {"shape": [3, 16, 16], "type": "rgb"},
        }
    }
    student = _FakeStudent()
    encoder = RobotDIFTCandidateObsEncoder(
        meta, student, "unused-in-injected-test", text_encoder=_FakeFrozenText(),
        output_dim=12, resize_shape=(16, 16), fpn_dim=8, model_dim=16,
        num_heads=2, mlp_hidden_dims=(16,), feature_cache_max_gib=0.001,
    ).eval()
    images = {
        "left_image": torch.rand(2, 3, 16, 16),
        "right_image": torch.rand(2, 3, 16, 16),
        "_robot_dift_frame_ids": torch.tensor([[[123, 0, 5]], [[123, 0, 6]]]),
    }
    first, _ = encoder(images, ["press", "press"])
    assert student.calls == 2
    second, _ = encoder(images, ["press", "press"])
    assert student.calls == 2
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    second.sum().backward()
    assert encoder.readout.lateral["us3"][0].weight.grad is not None
    encoder(images, ["open", "open"])
    assert student.calls == 4  # Language-conditioned Student features use a distinct key.


def test_bfloat16_student_cache_is_lossless_and_uses_half_the_bytes():
    class BfloatStudent(_FakeStudent):
        model_dtype = torch.bfloat16

        def _encode_backbone(self, images, captions):
            return {
                key: value.to(torch.bfloat16).float()
                for key, value in super()._encode_backbone(images, captions).items()
            }

    meta = {"obs": {"left_image": {"shape": [3, 16, 16], "type": "rgb"}}}
    encoder = RobotDIFTCandidateObsEncoder(
        meta, BfloatStudent(), "unused-in-injected-test", text_encoder=_FakeFrozenText(),
        output_dim=12, resize_shape=(16, 16), fpn_dim=8, model_dim=16,
        num_heads=2, mlp_hidden_dims=(16,), feature_cache_max_gib=0.001,
    ).eval()
    image = torch.rand(2, 3, 16, 16)
    obs = {"left_image": image, "_robot_dift_frame_ids": torch.tensor([[[7, 0, 1]], [[7, 0, 2]]])}
    cached, _ = encoder(obs, ["press", "press"])
    direct, _ = encoder({"left_image": image}, ["press", "press"])
    torch.testing.assert_close(cached, direct, rtol=0, atol=0)
    assert all(value.dtype == torch.bfloat16 for item in encoder._feature_cache.values() for value in item.values())
    assert encoder._feature_cache_bytes == sum(
        value.numel() * 2 for item in encoder._feature_cache.values() for value in item.values()
    )


def test_feature_cache_eviction_keeps_current_batch_outputs():
    meta = {"obs": {"left_image": {"shape": [3, 16, 16], "type": "rgb"}}}
    encoder = RobotDIFTCandidateObsEncoder(
        meta, _FakeStudent(), "unused-in-injected-test", text_encoder=_FakeFrozenText(),
        output_dim=12, resize_shape=(16, 16), fpn_dim=8, model_dim=16,
        num_heads=2, mlp_hidden_dims=(16,), feature_cache_max_gib=0.0000015,
    ).eval()
    images = {
        "left_image": torch.rand(2, 3, 16, 16),
        "_robot_dift_frame_ids": torch.tensor([[[456, 0, 1]], [[456, 0, 2]]]),
    }
    cached, _ = encoder(images, ["press", "press"])
    uncached, _ = encoder({"left_image": images["left_image"]}, ["press", "press"])
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    assert encoder._feature_cache_bytes <= encoder.feature_cache_max_bytes
