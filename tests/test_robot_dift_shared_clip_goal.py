"""The Stage-II policy shares one CLIP text tower with its visual readout."""

import hydra
import torch
from torch import nn

from agents.base_agent import BaseAgent


class _Vision(nn.Module):
    def encode_language_goal(self, captions):
        self.goal_captions = list(captions)
        return torch.ones(len(captions), 1, 512)

    def forward(self, observations, captions, alignment_context=None):
        self.readout_captions = list(captions)
        batch_times_frames = observations["cam_image"].shape[0]
        return torch.ones(batch_times_frames, 1, 1, 512), None


class _Agent(BaseAgent):
    def forward(self, obs_dict, actions=None, alignment_context=None):
        return self.compute_input_embeddings(obs_dict, alignment_context)


def test_candidate_reuses_image_encoder_clip_for_goal(monkeypatch):
    vision = _Vision()

    def instantiate(config):
        if config == "vision":
            return vision
        if config == "model":
            return nn.Identity()
        if config == "language":
            raise AssertionError("A second CLIP model must not be constructed")
        raise AssertionError(config)

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    agent = _Agent(
        model="model",
        obs_encoders="vision",
        language_encoders="language",
        device="cpu",
        state_dim=7,
        latent_dim=512,
        obs_seq_len=2,
        act_seq_len=8,
        cam_names=["cam"],
        if_dift_language=True,
    )
    obs = {
        "lang": ["press button", "open drawer"],
        "cam_image": torch.rand(2, 2, 3, 4, 4),
    }
    visual_tokens, goal = agent(obs)
    assert agent.language_encoder is None
    assert vision.goal_captions == obs["lang"]
    assert vision.readout_captions == obs["lang"]
    assert visual_tokens.shape == (2, 2, 512)
    assert goal.shape == (2, 1, 512)
