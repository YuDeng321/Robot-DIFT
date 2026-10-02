"""End-to-end CPU check of the paper Stage-I path with tiny stand-ins for SD2.1 and CLIP.

The robomimic diffusion policy, Stage-I encoder, S2-FPN/CLIP readout, alignment
logic, EMA, inference, checkpointing, and Student export are the real code; only
the SD2.1 U-Nets, VAE, and CLIP text tower are replaced by small modules.
"""

import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_policy_learning"))
sys.path.insert(0, str(ROOT))

import agents.encoders.cleandift_img_encoder as encoder_module  # noqa: E402
import agents.encoders.robot_dift_stage1_encoder as stage1_module  # noqa: E402
from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner  # noqa: E402
from agents.encoders.robot_dift_deploy_head import load_stage1_paper_readout, validate_deploy_head  # noqa: E402
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from robomimic.algo import algo_factory  # noqa: E402
from robomimic.utils.robot_dift_export import export_robot_dift_encoder  # noqa: E402
import train_droid_auto  # noqa: E402

CAMERAS = [
    "camera/image/hand_camera_left_image",
    "camera/image/varied_camera_1_left_image",
    "camera/image/varied_camera_2_left_image",
]
# Relative resolution of each tapped map for a 4x4 latent (us3 coarsest).
MAP_SIZES = {"mid": 1, "us1": 1, "us2": 1, "us3": 1, "us4": 1, "us5": 1, "us6": 2, "us7": 2, "us8": 2, "us9": 4, "us10": 4}


class _TinyAE(nn.Module):
    def __init__(self, repo=None, latent_mode="sample"):
        super().__init__()
        self.latent_mode = latent_mode
        self.proj = nn.Conv2d(3, 4, 1)
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        return self.proj(F.avg_pool2d(img, 8))


class _TinyUNet(nn.Module):
    supports_requested_feature_keys = True

    def __init__(self, feature_dims):
        super().__init__()
        self.heads = nn.ModuleDict({key: nn.Conv2d(4, dim, 1) for key, dim in feature_dims.items()})

    def forward(self, sample, timesteps, encoder_hidden_states=None, added_cond_kwargs=None, **kwargs):
        scale = 1.0 + 1e-3 * timesteps.float().reshape(-1, 1, 1, 1)
        context = encoder_hidden_states.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        maps = {}
        for key, head in self.heads.items():
            size = MAP_SIZES[key]
            maps[key] = F.interpolate(head(sample) * scale + context, size=(size, size), mode="nearest")
        return maps


class _ScaleAdapter(nn.Module):
    def __init__(self, dim, **_):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, cond):
        return x * self.scale


class _TinyAligner(StableFeatureAligner):
    """Real StableFeatureAligner methods over tiny networks."""

    def __init__(self, ae, feature_dims, t_min=1, t_max=999, num_t_stratification_bins=3,
                 use_text_condition=False, **_unused):
        nn.Module.__init__(self)
        self.ae = ae
        self.repo = "tiny/sd21"
        self.device = "cpu"
        self.feature_keys = tuple(feature_dims)
        self.use_adapters = True
        self.use_text_condition = use_text_condition
        self.t_min, self.t_max, self.t_max_model = t_min, t_max, 999
        self.num_t_stratification_bins = num_t_stratification_bins
        self.alignment_loss = "cossim"
        all_dims = dict(encoder_module.CleanDIFTImgEncoder.__init__.__globals__["OmegaConf"].load(
            ROOT / "agents/encoders/cleandift/configs/sd21_feature_extractor.yaml")["model"]["feature_dims"])
        # Real channel counts (the readout is built from them); maps stay tiny spatially.
        small = {key: int(dim) for key, dim in all_dims.items()}
        self.unet_feature_extractor_base = _TinyUNet(small)
        self.unet_feature_extractor_base.requires_grad_(False)
        self.unet_feature_extractor_cleandift = copy.deepcopy(self.unet_feature_extractor_base)
        self.unet_feature_extractor_cleandift.requires_grad_(True)
        self.adapters = nn.ModuleDict({key: _ScaleAdapter(small[key]) for key in self.feature_keys})
        self.time_emb = nn.Identity()
        self.time_in_proj = nn.Identity()
        self.mapping = nn.Identity()
        self.timestep = nn.Parameter(torch.tensor(261.0))
        self.pipe = SimpleNamespace(scheduler=SimpleNamespace(add_noise=lambda x, noise, t: x + 0.1 * noise))
        self._small_dims = small

    def get_prompt_embeds(self, prompt):
        return {"prompt_embeds": torch.tensor([[float(len(p))] * 8 for p in prompt])[:, None, :].repeat(1, 3, 1)}


class _TinyText(nn.Module):
    goal_dim = 6

    def __init__(self, *_):
        super().__init__()
        self.embed = nn.Linear(1, 512)
        self.requires_grad_(False)

    def forward(self, captions):
        lengths = torch.tensor([[float(len(c))] for c in captions])
        tokens = self.embed(lengths)[:, None, :].repeat(1, 77, 1)
        mask = torch.zeros(len(captions), 77, dtype=torch.bool)
        mask[:, :4] = True
        return tokens, mask

    def encode_goal(self, captions):
        return torch.tensor([[float(len(c))] * self.goal_dim for c in captions])[:, None, :]


def _build_stage1_algo(tmp_path, monkeypatch, *extra_args):
    monkeypatch.setattr(encoder_module, "StableFeatureAligner", _TinyAligner)
    monkeypatch.setattr(encoder_module, "AutoencoderKL", _TinyAE)
    monkeypatch.setattr(stage1_module, "FrozenCLIPTextTokens", _TinyText)
    monkeypatch.setenv("WORK", str(tmp_path))
    args = train_droid_auto.build_parser().parse_args(
        ["--name", "tiny", "--config_only", "--save_cleandift_dir", str(tmp_path / "encoder"), *extra_args]
    )
    train_droid_auto.validate_args(args)
    config = train_droid_auto.create_droid_config(args)
    with config.unlocked():
        readout = config.algo.robot_dift_paper_readout
        readout.output_dim, readout.fpn_dim, readout.model_dim = 8, 8, 8
        readout.num_heads, readout.mlp_hidden_dims = 2, [16]
        config.algo.unet.down_dims = [8, 16]
        config.algo.unet.diffusion_step_embed_dim = 8
        config.algo.noise_samples = 2
        config.algo.optim_params.policy.learning_rate.initial = 1e-2
    ObsUtils.initialize_obs_utils_with_config(config)
    ObsUtils.ImageModality.set_obs_processor(
        lambda obs: ObsUtils.batch_image_hwc_to_chw(ObsUtils.TU.to_float(obs))
    )
    shapes = OrderedDict((key, [3, 32, 32]) for key in CAMERAS)
    model = algo_factory(
        algo_name=config.algo_name, config=config, obs_key_shapes=shapes, ac_dim=10, device=torch.device("cpu"),
    )
    return model, config


@pytest.fixture
def stage1_algo_factory(tmp_path, monkeypatch):
    yield lambda *extra_args: _build_stage1_algo(tmp_path, monkeypatch, *extra_args)
    ObsUtils.ImageModality.set_obs_processor(None)


@pytest.fixture
def stage1_algo(stage1_algo_factory):
    return stage1_algo_factory()


def _batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    obs = {key: torch.rand(2, 2, 32, 32, 3, generator=generator) * 2 - 1 for key in CAMERAS}
    obs["raw_language"] = np.array([b"put the cup in the sink", b"close the drawer"])
    obs["pad_mask"] = torch.ones(2, 2, 1)
    return {"obs": obs, "actions": torch.rand(2, 16, 10, generator=generator) * 2 - 1}


def _train_step(model, epoch=1, seed=0):
    batch = model.process_batch_for_training(_batch(seed))
    batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
    model.set_train()
    return model.train_on_batch(batch, epoch)


def test_paper_stage1_step_trains_student_readout_and_policy(stage1_algo):
    model, _ = stage1_algo
    encoder = model.nets["policy"]["obs_encoder"]
    assert type(encoder).__name__ == "RobotDIFTStage1Encoder"
    assert encoder.student.readout == "none" and encoder.student.freeze_backbone is False
    student = encoder.student.model.unet_feature_extractor_cleandift
    teacher = encoder.student.model.unet_feature_extractor_base
    assert not any(p.requires_grad for p in teacher.parameters())
    groups = {group["name"] for group in model.optimizers["policy"].param_groups}
    assert groups == {"backbone", "head", "policy"}
    assert len(model.ema.shadow_params) == len(model.ema_parameters())

    snapshot = {
        "student": copy.deepcopy(student.state_dict()),
        "teacher": copy.deepcopy(teacher.state_dict()),
        "readout": copy.deepcopy(encoder.readout.state_dict()),
        "timestep": encoder.student.model.timestep.detach().clone(),
    }
    info = _train_step(model)
    losses = info["losses"]
    assert torch.isfinite(losses["total_loss"]) and losses["alignment_loss"].item() > 0
    assert info["alignment_weight"] == pytest.approx(0.1)
    assert info["optimizer_step"] == 1.0 and model.ema.optimization_step == 1
    assert {"raw_cosine_us3", "neg_cossim_us10"} <= set(info["alignment_metrics"])
    changed = lambda old, new: any(not torch.equal(old[k], new[k]) for k in old)  # noqa: E731
    assert changed(snapshot["student"], student.state_dict())
    assert changed(snapshot["readout"], encoder.readout.state_dict())
    assert not changed(snapshot["teacher"], teacher.state_dict())
    assert not torch.equal(snapshot["timestep"], encoder.student.model.timestep.detach())
    log = model.log_info(info)
    assert "Alignment/raw_cosine_us3" in log and "Optimizer_Group_Grad_Norms/backbone" in log


def test_student_freeze_then_warmup_in_the_training_step(stage1_algo_factory):
    model, _ = stage1_algo_factory("--student_freeze_steps", "1", "--student_lr_warmup_steps", "2")
    encoder = model.nets["policy"]["obs_encoder"]
    aligner = encoder.student.model
    student = aligner.unet_feature_extractor_cleandift
    optimizer = model.optimizers["policy"]
    assert [group["name"] for group in optimizer.param_groups] == ["student", "alignment", "head", "policy"]
    configured_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert len(set(configured_lrs.values())) == 1

    def snapshot():
        return {
            "student": copy.deepcopy(student.state_dict()),
            "timestep": {"t": aligner.timestep.detach().clone()},
            "readout": copy.deepcopy(encoder.readout.state_dict()),
            "adapters": copy.deepcopy(aligner.adapters.state_dict()),
        }

    changed = lambda old, new: any(not torch.equal(old[k], new[k]) for k in old)  # noqa: E731
    before = snapshot()
    info = _train_step(model, epoch=1)
    frozen = snapshot()
    assert info["student_lr_scale"] == 0.0 and info["optimizer_step"] == 1.0
    assert not changed(before["student"], frozen["student"])
    assert not changed(before["timestep"], frozen["timestep"])
    assert changed(before["readout"], frozen["readout"]) and changed(before["adapters"], frozen["adapters"])

    info = _train_step(model, epoch=2, seed=1)
    assert info["student_lr_scale"] == 0.5
    assert changed(frozen["student"], student.state_dict())
    assert model.log_info(info)["Student_LR_Scale"] == 0.5
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == configured_lrs


def test_inference_restores_live_weights_and_checkpoint_roundtrip(stage1_algo):
    model, config = stage1_algo
    _train_step(model)
    live = [p.detach().clone() for p in model.ema_parameters()]
    batch = model.postprocess_batch_for_training(model.process_batch_for_training(_batch(1)), None)
    model.set_eval()
    actions = model._get_action_trajectory(batch["obs"], batch["lang_prompts"])
    assert actions.shape == (2, 8, 10) and torch.isfinite(actions).all()
    for before, after in zip(live, model.ema_parameters()):
        torch.testing.assert_close(before, after)

    state = model.serialize()
    shapes = OrderedDict((key, [3, 32, 32]) for key in CAMERAS)
    clone = algo_factory(algo_name=config.algo_name, config=config, obs_key_shapes=shapes, ac_dim=10,
                         device=torch.device("cpu"))
    clone.deserialize(copy.deepcopy(state))
    for key, value in model.nets.state_dict().items():
        torch.testing.assert_close(clone.nets.state_dict()[key], value)
    assert clone.ema.optimization_step == model.ema.optimization_step


def test_export_writes_a_deployable_student_and_readout(stage1_algo, tmp_path):
    model, config = stage1_algo
    _train_step(model)
    saved = export_robot_dift_encoder(model, config, 7, str(tmp_path / "encoder"), save_ema=True)
    assert [Path(path).name for path in saved] == ["checkpoint-7", "checkpoint-7-ema"]
    ema_dir = Path(saved[1])
    metadata = json.loads((ema_dir / "metadata.json").read_text())
    assert metadata["readout"] == "paper" and metadata["ema"] is True
    assert metadata["feature_key"] == ["us3", "us6", "us8"]
    assert metadata["vae_latent_mode"] == "mode" and metadata["student_init"] == "sd_teacher"
    assert json.loads(metadata["training_config"])["algo"]["robot_dift_readout"] == "paper"
    assert validate_deploy_head(ema_dir)

    encoder = model.nets["policy"]["obs_encoder"]
    student_state = torch.load(ema_dir / "unet" / "diffusion_pytorch_model.bin", weights_only=True)
    reloaded = _TinyUNet(encoder.student.model._small_dims)
    reloaded.load_state_dict(student_state, strict=True)
    # EMA export differs from the live Student after one warmup step (decay 0 copies, so check keys).
    assert set(student_state) == set(encoder.student.model.unet_feature_extractor_cleandift.state_dict())

    config_readout = metadata["readout_config"]
    channels = {key: int(encoder.student.feature_dims[key]) for key in metadata["feature_key"]}
    readout = RobotDIFTPaperReadout(
        channels, fpn_dim=config_readout["fpn_dim"], model_dim=config_readout["model_dim"],
        output_dim=config_readout["output_dim"], num_heads=config_readout["num_heads"],
        mlp_hidden_dims=tuple(config_readout["mlp_hidden_dims"]),
    )
    assert load_stage1_paper_readout(readout, ema_dir) == len(readout.state_dict())


def test_interpolated_student_spans_initialization_and_trained_weights(stage1_algo, tmp_path):
    import importlib.util

    from safetensors.torch import save_file

    model, config = stage1_algo
    for epoch in (1, 2):
        _train_step(model, epoch=epoch, seed=epoch)
    trained_dir = Path(export_robot_dift_encoder(model, config, 2, str(tmp_path / "encoder"), save_ema=False)[0])
    aligner = model.nets["policy"]["obs_encoder"].student.model
    # The Student starts as a copy of the Teacher, so the Teacher weights stand in for SD2.1.
    repo = tmp_path / "sd21"
    (repo / "unet").mkdir(parents=True)
    initial = {key: value.contiguous() for key, value in aligner.unet_feature_extractor_base.state_dict().items()}
    save_file(initial, str(repo / "unet" / "diffusion_pytorch_model.safetensors"))

    spec = importlib.util.spec_from_file_location("interpolate_student", ROOT / "scripts/release/interpolate_student.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--checkpoint", str(trained_dir), "--model-repo", str(repo), "--alpha", "0", "0.5", "1",
                        "--output-root", str(tmp_path / "interpolated")]) == 0

    trained = torch.load(trained_dir / "unet" / "diffusion_pytorch_model.bin", weights_only=True)
    trained_t = torch.load(trained_dir / "timestep.bin", weights_only=True)["timestep"].item()
    assert trained_t != 261.0 and any(not torch.equal(trained[key], initial[key]) for key in initial)
    for alpha, weights, timestep in ((0.0, initial, 261.0), (1.0, trained, trained_t)):
        path = tmp_path / "interpolated" / f"checkpoint-2-alpha{alpha:.2f}"
        state = torch.load(path / "unet" / "diffusion_pytorch_model.bin", weights_only=True)
        assert all(torch.equal(state[key], weights[key]) for key in weights)
        _TinyUNet(aligner._small_dims).load_state_dict(state, strict=True)
        assert torch.load(path / "timestep.bin", weights_only=True)["timestep"].item() == pytest.approx(timestep)
        assert validate_deploy_head(path)
