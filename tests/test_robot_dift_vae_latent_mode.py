"""VAE posterior sampling is explicit and remains checkpoint-compatible."""

import torch
from torch import nn
import pytest

from agents.encoders.cleandift.src import ae as vae_module


class _Posterior:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean + torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))

    def encode(self, image, return_dict=False):
        assert return_dict is False
        return (_Posterior(image * self.gain),)


def test_vae_mode_is_deterministic_and_sample_default_remains_stochastic(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setattr(
        vae_module.diffusers.AutoencoderKL,
        "from_pretrained",
        lambda *args, **kwargs: _FakeVAE(),
    )
    image = torch.ones(2, 3, 4, 4)
    sampled = vae_module.AutoencoderKL(scale=0.5, shift=1.0, repo="local/fake")
    deterministic = vae_module.AutoencoderKL(
        scale=0.5, shift=1.0, repo="local/fake", latent_mode="mode"
    )
    deterministic.load_state_dict(sampled.state_dict(), strict=True)

    assert sampled.latent_mode == "sample"
    assert torch.equal(deterministic.encode(image), torch.full_like(image, 0.5))
    assert torch.equal(deterministic.encode(image), deterministic.encode(image))
    torch.manual_seed(1)
    first_sample = sampled.encode(image)
    torch.manual_seed(2)
    second_sample = sampled.encode(image)
    assert not torch.equal(first_sample, second_sample)
    assert not first_sample.requires_grad


def test_vae_rejects_unknown_latent_mode_before_loading_weights():
    with pytest.raises(ValueError, match="latent_mode"):
        vae_module.AutoencoderKL(latent_mode="mean")
