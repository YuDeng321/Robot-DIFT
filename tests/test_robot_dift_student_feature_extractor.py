"""Small checkpoint contract for the Stage-II Student-only feature path."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor


FEATURE_DIMS = {"us3": 4, "us6": 4, "us8": 4}


class _FakeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))

    def forward(self, latents, timesteps, encoder_hidden_states, added_cond_kwargs):
        self.last_timesteps = timesteps.detach().clone()
        base = latents.mean(dim=1, keepdim=True) * self.gain
        base = base + encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
        return {
            "us3": F.adaptive_avg_pool2d(base, 2).repeat(1, 4, 1, 1),
            "us6": F.adaptive_avg_pool2d(base, 4).repeat(1, 4, 1, 1),
            "us8": F.adaptive_avg_pool2d(base, 8).repeat(1, 4, 1, 1),
        }


class _Posterior:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean + 0.25 * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class _InnerVAE(nn.Module):
    def encode(self, image, return_dict=False):
        return (_Posterior(image.mean(dim=1, keepdim=True)),)


class _FakeAE(nn.Module):
    scale = 0.18215
    shift = 0.0

    def __init__(self):
        super().__init__()
        self.ae = _InnerVAE()

    def encode(self, image):
        return (self.ae.encode(image, return_dict=False)[0].sample() - self.shift) * self.scale


class _FakeTokenizer:
    model_max_length = 77

    def __call__(self, captions, **kwargs):
        self.last_captions = list(captions)
        ids = torch.tensor([len(caption) for caption in captions]).clamp(max=127)
        return {"input_ids": ids[:, None].expand(-1, 77), "attention_mask": torch.ones(len(ids), 77)}


class _FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_attention_mask=False)
        self.embedding = nn.Embedding(128, 4)

    def forward(self, input_ids):
        return (self.embedding(input_ids),)


def _checkpoint(tmp_path, state=None, latent_mode=None):
    unet_dir = tmp_path / "unet"
    unet_dir.mkdir(parents=True)
    if state is None:
        state = _FakeStudent().state_dict()
    torch.save(state, unet_dir / "diffusion_pytorch_model.bin")
    torch.save({"timestep": torch.tensor(17.5)}, tmp_path / "timestep.bin")
    metadata = {
            "model_type": "cleandift",
            "sd_version": "sd21",
            "use_text_condition": True,
            "feature_dims": FEATURE_DIMS,
            "components": {"student_unet": "unet/"},
    }
    if latent_mode is not None:
        metadata["vae_latent_mode"] = latent_mode
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def _extractor(path, latent_mode="auto"):
    return RobotDIFTStudentFeatureExtractor(
        str(path),
        device="cpu",
        feature_dims=FEATURE_DIMS,
        vae_latent_mode=latent_mode,
        student_unet=_FakeStudent(),
        ae=_FakeAE(),
        tokenizer=_FakeTokenizer(),
        text_encoder=_FakeText(),
    )


def test_mode_is_repeatable_and_sample_keeps_stage1_sampling(tmp_path):
    mode_checkpoint = _checkpoint(tmp_path / "mode", latent_mode="mode")
    image = torch.rand(2, 3, 16, 16)
    deterministic = _extractor(mode_checkpoint)
    assert deterministic.vae_latent_mode == "mode"
    deterministic.train()
    first = deterministic._encode_backbone(image, ["press", "drawer"])
    second = deterministic._encode_backbone(image, ["press", "drawer"])
    assert all(torch.equal(first[key], second[key]) for key in FEATURE_DIMS)
    assert torch.equal(deterministic.student_unet.last_timesteps, torch.full((2,), 17.5))
    assert deterministic.tokenizer.last_captions == ["press", "drawer"]
    assert not deterministic.training
    assert not any(param.requires_grad for param in deterministic.parameters())
    assert not hasattr(deterministic, "teacher")
    assert not hasattr(deterministic, "readout")

    sampled = _extractor(_checkpoint(tmp_path / "legacy_sample"))
    assert sampled.vae_latent_mode == "sample"
    torch.manual_seed(1)
    first_sample = sampled._encode_backbone(image, ["press", "drawer"])
    torch.manual_seed(2)
    second_sample = sampled._encode_backbone(image, ["press", "drawer"])
    assert not torch.equal(first_sample["us6"], second_sample["us6"])


def test_explicit_latent_mode_mismatch_is_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path, latent_mode="mode")
    with pytest.raises(ValueError, match="differs from Stage-I checkpoint metadata"):
        _extractor(checkpoint, latent_mode="sample")


def test_invalid_metadata_latent_mode_is_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path, latent_mode="mean")
    with pytest.raises(ValueError, match="Invalid Stage-I VAE latent mode"):
        _extractor(checkpoint)


@pytest.mark.parametrize("bad_state", [{"other": torch.tensor(1.0)}, {"gain": torch.tensor(2.0), "other": torch.tensor(1.0)}])
def test_student_checkpoint_rejects_missing_or_unexpected_keys(tmp_path, bad_state):
    checkpoint = _checkpoint(tmp_path, state=bad_state)
    with pytest.raises(RuntimeError, match="strict loading"):
        _extractor(checkpoint)


def test_student_checkpoint_requires_learned_timestep(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "timestep.bin").unlink()
    with pytest.raises(FileNotFoundError, match="timestep"):
        _extractor(checkpoint)


def test_student_checkpoint_requires_stage1_text_conditioning(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["use_text_condition"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="text-conditioned"):
        _extractor(checkpoint)


def test_public_forward_normalizes_explicit_ranges_and_keeps_batch_order(tmp_path):
    encoder = _extractor(_checkpoint(tmp_path, latent_mode="mode"))
    byte_images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    prompts = ["press the button", "open the drawer"]
    expected = encoder._encode_backbone(byte_images.float() / 127.5 - 1.0, prompts)
    actual = encoder(byte_images, prompts, input_range="uint8")
    unit = encoder.extract(byte_images.float() / 255.0, prompts, input_range="zero_one")
    assert encoder.tokenizer.last_captions == prompts
    assert set(actual) == set(FEATURE_DIMS)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
        torch.testing.assert_close(unit[key], expected[key])
        assert actual[key].dtype == torch.float32
        assert not actual[key].requires_grad


@pytest.mark.parametrize("images,input_range,error", [
    (torch.zeros(0, 3, 64, 64), "minus_one_one", "nonempty"),
    (torch.zeros(1, 1, 64, 64), "minus_one_one", "nonempty"),
    (torch.zeros(1, 3, 65, 64), "minus_one_one", "multiples of 64"),
    (torch.full((1, 3, 64, 64), float("nan")), "minus_one_one", "finite"),
    (torch.full((1, 3, 64, 64), 2.0), "zero_one", "within"),
    (torch.zeros(1, 3, 64, 64), "auto", "input_range"),
])
def test_public_forward_rejects_invalid_images(tmp_path, images, input_range, error):
    encoder = _extractor(_checkpoint(tmp_path, latent_mode="mode"))
    with pytest.raises(ValueError, match=error):
        encoder(images, input_range=input_range)


def test_public_forward_rejects_dtype_and_prompt_mismatch(tmp_path):
    encoder = _extractor(_checkpoint(tmp_path, latent_mode="mode"))
    images = torch.zeros(1, 3, 64, 64)
    with pytest.raises(TypeError, match="uint8"):
        encoder(images, input_range="uint8")
    with pytest.raises(TypeError, match="Floating-point"):
        encoder(images.to(torch.uint8))
    with pytest.raises(ValueError, match="text prompts"):
        encoder(images, ["one", "two"])
