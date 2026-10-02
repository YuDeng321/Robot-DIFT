"""Teacher/Student alignment core: one Student pass, exact per-map terms, strict options."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner


class _IdentityAE(nn.Module):
    def encode(self, x):
        return x


class _ConstantMaps(nn.Module):
    """Returns constant maps and records every call's batch size and kwargs."""

    supports_requested_feature_keys = True

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, x, timesteps, **kwargs):
        self.calls.append((x.shape[0], kwargs.get("requested_feature_keys")))
        return {
            "us3": torch.full_like(x, self.value),
            "us6": torch.full_like(x, self.value + 1),
        }


class _OffsetAdapter(nn.Module):
    def forward(self, x, cond):
        return x + 10


def _mock_aligner(num_bins=1):
    model = object.__new__(StableFeatureAligner)
    nn.Module.__init__(model)
    model.t_min = 1
    model.t_max = 1000
    model.t_max_model = 999
    model.num_t_stratification_bins = num_bins
    model.alignment_loss = "mse"
    model.ae = _IdentityAE()
    model.pipe = SimpleNamespace(scheduler=SimpleNamespace(add_noise=lambda x, noise, t: x))
    model.unet_feature_extractor_base = _ConstantMaps(0)
    model.unet_feature_extractor_cleandift = _ConstantMaps(2)
    model.feature_keys = ("us3", "us6")
    model.use_adapters = True
    model.use_text_condition = False
    model._empty_prompt_embeds = torch.zeros(1, 3, 4)
    model.adapters = nn.ModuleDict({"us3": _OffsetAdapter(), "us6": _OffsetAdapter()})
    model.time_emb = nn.Identity()
    model.time_in_proj = nn.Identity()
    model.mapping = nn.Identity()
    model.timestep = nn.Parameter(torch.tensor(261.0))
    return model


def test_forward_aligns_every_map_through_adapters():
    model = _mock_aligner()
    losses = model(torch.zeros(2, 1, 2, 2), ["left", "right"], apply_adapter=True)
    # Student 2 (+10 adapter) vs Teacher 0 for us3; 3 (+10) vs 1 for us6.
    assert losses["mse_us3"].item() == pytest.approx(144.0)
    assert losses["mse_us6"].item() == pytest.approx(144.0)
    raw = model(torch.zeros(2, 1, 2, 2), ["left", "right"], apply_adapter=False)
    assert raw["mse_us3"].item() == pytest.approx(4.0)


def test_forward_rejects_unknown_options():
    model = _mock_aligner()
    with pytest.raises(TypeError, match="unknown_option"):
        model(torch.zeros(1, 1, 2, 2), ["task"], apply_adapter=True, unknown_option=1)


def test_student_runs_once_for_all_teacher_timesteps():
    model = _mock_aligner(num_bins=3)
    losses = model(torch.zeros(4, 1, 2, 2), ["task"] * 4, apply_adapter=True)
    student_calls = model.unet_feature_extractor_cleandift.calls
    teacher_calls = model.unet_feature_extractor_base.calls
    assert [batch for batch, _ in student_calls] == [4]
    assert [batch for batch, _ in teacher_calls] == [12]
    assert teacher_calls[0][1] == ["us3", "us6"]
    assert losses["mse_us3"].item() == pytest.approx(144.0)


def test_alignment_terms_report_raw_cosine_and_reuse_a_given_student_pass():
    model = _mock_aligner()
    x_0 = torch.ones(2, 1, 2, 2)
    student = {"us3": torch.ones(2, 1, 2, 2), "us6": -torch.ones(2, 1, 2, 2)}
    teacher = model.unet_feature_extractor_base
    teacher.value = 1.0
    conds = model._get_unet_conds(["a", "b"], "cpu", torch.float32, 1)
    losses, metrics = model.alignment_terms(x_0, conds, student, apply_adapter=False, return_raw_cosine=True)
    assert model.unet_feature_extractor_cleandift.calls == []
    assert metrics["raw_cosine_us3"].item() == pytest.approx(1.0)
    assert metrics["raw_cosine_us6"].item() == pytest.approx(-1.0)
    assert losses["mse_us6"].item() == pytest.approx(9.0)
    with pytest.raises(KeyError, match="us6"):
        model.alignment_terms(x_0, conds, {"us3": student["us3"]})


def test_teacher_timesteps_cover_one_to_999():
    model = _mock_aligner(num_bins=1)
    torch.manual_seed(0)
    samples = model.sample_timesteps(200_000, "cpu")
    assert samples.shape == (200_000, 1)
    assert int(samples.min()) == 1 and int(samples.max()) == 999


def test_student_maps_skip_unused_final_block_when_possible():
    model = _mock_aligner()
    model.student_maps(torch.zeros(1, 1, 2, 2), {}, ["us3"])
    model.student_maps(torch.zeros(1, 1, 2, 2), {})
    assert model.unet_feature_extractor_cleandift.calls == [(1, ["us3"]), (1, None)]


def test_each_distinct_prompt_is_encoded_once_and_order_is_kept():
    model = _mock_aligner()
    model.use_text_condition = True
    encoded = []

    def fake_prompt_embeds(prompts):
        encoded.append(list(prompts))
        return {"prompt_embeds": torch.tensor([[float(len(p))] for p in prompts])[:, :, None]}

    model.get_prompt_embeds = fake_prompt_embeds
    conds = model._get_unet_conds(["ab", "abcd", "ab", "abcd", "ab"], "cpu", torch.float32, 1)
    assert encoded == [["ab", "abcd"]]
    assert conds["encoder_hidden_states"].flatten().tolist() == [2.0, 4.0, 2.0, 4.0, 2.0]


class _RecordingTimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, t):
        self.seen.append((t.dtype, torch.is_autocast_enabled("cpu")))
        return t


class _Bf16AE(nn.Module):
    def encode(self, x):
        return x.to(torch.bfloat16)


def test_latents_and_adapter_time_stay_float32_under_bf16_autocast():
    model = _mock_aligner()
    model.ae = _Bf16AE()
    model.time_emb = _RecordingTimeEmbedding()
    image = torch.zeros(2, 1, 2, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model.encode_latents(image).dtype == torch.float32
        model(image, ["a", "b"], apply_adapter=True)
    assert model.time_emb.seen == [(torch.float32, False)]
