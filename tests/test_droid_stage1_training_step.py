"""Stage-I optimizer step: EMA power, non-finite guard, and policy conditioning."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from diffusers.training_utils import EMAModel
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "droid_policy_learning"))
from robomimic.algo.algo import Algo  # noqa: E402
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def _policy(max_grad_norm=1.0):
    policy = object.__new__(DiffusionPolicyUNet)
    policy.nets = nn.ModuleDict({
        "policy": nn.ModuleDict({
            "obs_encoder": nn.Linear(3, 3),
            "noise_pred_net": nn.Linear(3, 3),
        })
    })
    frozen = nn.Linear(3, 3)
    frozen.requires_grad_(False)
    policy.nets["policy"]["obs_encoder"].frozen_teacher = frozen
    policy.optimizers = {"policy": torch.optim.Adam(
        [
            {"params": list(policy.nets["policy"]["obs_encoder"].parameters())[:2], "name": "backbone"},
            {"params": list(policy.nets["policy"]["noise_pred_net"].parameters()), "name": "policy"},
        ],
        lr=1e-2,
    )}
    policy.global_config = SimpleNamespace(train=SimpleNamespace(max_grad_norm=max_grad_norm))
    policy.algo_config = SimpleNamespace(optim_params={"policy": {}})
    policy.grad_scaler = None
    policy._nonfinite_gradient_skips = 0
    policy._optimizer_stepped_since_scheduler = False
    policy.ema = EMAModel(
        parameters=DiffusionPolicyUNet._trainable_parameters(policy.nets),
        decay=1.0, min_decay=0.0, use_ema_warmup=True, inv_gamma=1.0, power=0.75,
    )
    return policy


def _loss(policy, value=1.0):
    x = torch.ones(2, 3)
    out = policy.nets["policy"]["noise_pred_net"](policy.nets["policy"]["obs_encoder"](x))
    return out.pow(2).mean() * value


def test_ema_excludes_frozen_parameters_and_uses_power_schedule():
    policy = _policy()
    assert len(policy.ema.shadow_params) == 4  # two trainable Linear layers
    decays = []
    for _ in range(3):
        info = {}
        policy._backward_and_step(_loss(policy), info, epoch=1, accumulation_steps=1,
                                  finish_accumulation=True, audit_active=False)
        decays.append(policy.ema.cur_decay_value)
    # diffusers warmup: decay_s = 1 - (1 + s)^-0.75 with s = optimization_step - 1.
    assert decays == pytest.approx([0.0, 1 - 2 ** -0.75, 1 - 3 ** -0.75])


def test_non_finite_gradients_skip_the_update_and_the_ema():
    policy = _policy()
    before = [p.detach().clone() for p in policy.ema_parameters()]
    info = {}
    policy._backward_and_step(_loss(policy, float("nan")), info, epoch=1, accumulation_steps=1,
                              finish_accumulation=True, audit_active=False)
    assert info["optimizer_step"] == 0.0
    assert info["nonfinite_gradient_skips"] == 1.0
    assert policy.ema.optimization_step == 0
    for old, new in zip(before, policy.ema_parameters()):
        torch.testing.assert_close(old, new)
    assert all(p.grad is None for p in policy.ema_parameters())

    info = {}
    policy._backward_and_step(_loss(policy), info, epoch=2, accumulation_steps=1,
                              finish_accumulation=True, audit_active=False)
    assert info["optimizer_step"] == 1.0 and policy.ema.optimization_step == 1
    assert set(info["optimizer_group_grad_norms"]) == {"backbone", "policy"}
    assert info["grad_norm"] > 0


def test_accumulation_steps_only_on_the_last_microbatch():
    policy = _policy()
    info = {}
    policy._backward_and_step(_loss(policy), info, epoch=1, accumulation_steps=2,
                              finish_accumulation=False, audit_active=False)
    assert "optimizer_step" not in info and policy.ema.optimization_step == 0
    policy._backward_and_step(_loss(policy), info, epoch=1, accumulation_steps=2,
                              finish_accumulation=True, audit_active=False)
    assert info["optimizer_step"] == 1.0


def test_ema_parameter_set_must_not_change():
    policy = _policy()
    policy.nets["policy"]["noise_pred_net"].weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="Trainable parameters changed"):
        policy.ema_parameters()


class _PaperEncoder(nn.Module):
    output_dim = 4
    goal_dim = 3

    def encode_sequence(self, obs, prompts, alignment=False, return_raw_cosine=False):
        batch = obs["cam"].shape[0]
        features = torch.arange(batch * 2 * 4, dtype=torch.float32).reshape(batch, 2, 4)
        loss = torch.tensor(2.0) if alignment else None
        return features, loss, {"raw_cosine_us3": torch.tensor(0.25)} if alignment else {}

    def encode_language_goal(self, prompts, batch=None):
        return torch.full((batch, self.goal_dim), -1.0)


def test_paper_condition_concatenates_frame_tokens_and_clip_goal():
    policy = object.__new__(DiffusionPolicyUNet)
    policy.algo_config = SimpleNamespace(robot_dift_readout="paper")
    policy.nets = nn.ModuleDict({"policy": nn.ModuleDict({"obs_encoder": _PaperEncoder()})})
    obs = {"cam": torch.zeros(2, 2, 3, 4, 4)}
    cond, loss, metrics = policy._observation_condition(obs, ["a", "b"], compute_alignment=True)
    assert cond.shape == (2, 2 * 4 + 3)
    assert cond[0, :8].tolist() == list(range(8)) and cond[0, 8:].tolist() == [-1.0] * 3
    assert loss.item() == 2.0 and "raw_cosine_us3" in metrics
    cond, loss, metrics = policy._observation_condition(obs, ["a", "b"], compute_alignment=False)
    assert loss.item() == 0.0 and metrics == {}


def test_all_parameter_ema_state_maps_onto_trainable_tensors():
    policy = _policy()
    all_parameters = list(policy.nets.parameters())
    full = EMAModel(parameters=[p.detach().clone() + 1 for p in all_parameters], decay=0.9999)
    policy.deserialize({"nets": policy.nets.state_dict(), "ema": full.state_dict()})
    trainable = policy.ema_parameters()
    assert len(policy.ema.shadow_params) == len(trainable)
    for shadow, parameter in zip(policy.ema.shadow_params, trainable):
        torch.testing.assert_close(shadow, parameter.detach() + 1)

    wrong = EMAModel(parameters=[torch.zeros(7) for _ in trainable], decay=0.9999)
    with pytest.raises(RuntimeError, match="different parameter set"):
        policy.deserialize({"nets": policy.nets.state_dict(), "ema": wrong.state_dict()})


def test_locked_configs_without_optional_keys_use_defaults():
    from robomimic.config import config_factory

    config = config_factory("diffusion_policy")
    with config.unlocked():
        del config.algo["robot_dift_readout"]
        del config.algo.ema["max_decay"]
    config.lock()
    policy = object.__new__(DiffusionPolicyUNet)
    policy.algo_config = config.algo
    assert policy.uses_paper_readout is False
    with pytest.raises(RuntimeError):
        getattr(config.algo, "robot_dift_readout", "legacy")


def test_paper_encoder_uses_configured_device_and_feature_keys(monkeypatch):
    import agents.encoders.robot_dift_stage1_encoder as stage1_module

    captured = {}

    class Capture:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(stage1_module, "RobotDIFTStage1Encoder", Capture)
    policy = object.__new__(DiffusionPolicyUNet)
    policy.device = torch.device("cpu")
    policy.obs_shapes = {"camera/image/a": [3, 8, 8]}
    policy.obs_config = SimpleNamespace(encoder=SimpleNamespace(rgb=SimpleNamespace(
        core_kwargs=SimpleNamespace(backbone_kwargs={"feature_key": ["us3", "us6"], "device": "cuda"}))))
    policy.algo_config = SimpleNamespace(robot_dift_paper_readout={"clip_model_path": "/x.pt", "output_dim": 8})
    import robomimic.utils.obs_utils as ObsUtils
    monkeypatch.setattr(ObsUtils, "OBS_KEYS_TO_MODALITIES", {"camera/image/a": "rgb"})
    policy._create_paper_encoder()
    assert captured["feature_keys"] == ("us3", "us6")
    assert captured["student"]["device"] == "cpu"
    assert captured["clip_model_path"] == "/x.pt" and captured["output_dim"] == 8


def test_student_lr_scale_freezes_then_ramps():
    assert [TorchUtils.student_lr_scale(step, 3, 4) for step in range(9)] == [
        0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0
    ]
    assert [TorchUtils.student_lr_scale(step, 2, 0) for step in range(3)] == [0.0, 0.0, 1.0]
    assert TorchUtils.student_lr_schedule({}) == (0, 0)
    with pytest.raises(ValueError, match="student_freeze_steps"):
        TorchUtils.student_lr_schedule({"student_freeze_steps": -1})


def _student_policy(freeze_steps, warmup_steps):
    policy = _policy()
    encoder = policy.nets["policy"]["obs_encoder"]
    policy.optimizers = {"policy": torch.optim.Adam(
        [
            {"params": [encoder.weight, encoder.bias], "name": "student"},
            {"params": list(policy.nets["policy"]["noise_pred_net"].parameters()), "name": "policy"},
        ],
        lr=1e-2,
    )}
    policy.algo_config = SimpleNamespace(optim_params={"policy": {
        "student_freeze_steps": freeze_steps, "student_lr_warmup_steps": warmup_steps,
    }})
    return policy


def test_student_schedule_scales_only_the_student_group_and_restores_its_lr():
    policy = _student_policy(freeze_steps=2, warmup_steps=2)
    optimizer = policy.optimizers["policy"]
    lrs_at_step = []
    original_step = optimizer.step

    def recording_step(*args, **kwargs):
        lrs_at_step.append({group["name"]: group["lr"] for group in optimizer.param_groups})
        return original_step(*args, **kwargs)

    optimizer.step = recording_step
    student = policy.nets["policy"]["obs_encoder"]
    head = policy.nets["policy"]["noise_pred_net"]
    scales, student_moved, head_moved = [], [], []
    for epoch in range(1, 6):
        student_before, head_before = student.weight.detach().clone(), head.weight.detach().clone()
        info = {}
        policy._backward_and_step(_loss(policy), info, epoch=epoch, accumulation_steps=1,
                                  finish_accumulation=True, audit_active=False)
        scales.append(info["student_lr_scale"])
        student_moved.append(not torch.equal(student_before, student.weight))
        head_moved.append(not torch.equal(head_before, head.weight))
        assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {"student": 1e-2, "policy": 1e-2}
    assert scales == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert [lrs["student"] for lrs in lrs_at_step] == pytest.approx([0.0, 0.0, 5e-3, 1e-2, 1e-2])
    assert all(lrs["policy"] == 1e-2 for lrs in lrs_at_step)
    assert student_moved == [False, False, True, True, True] and all(head_moved)
    assert policy.log_info({"losses": {"l2_loss": 0.0, "total_loss": 0.0, "alignment_loss": 0.0},
                            **info})["Student_LR_Scale"] == 1.0


def test_without_a_student_schedule_nothing_is_scaled_or_logged():
    policy = _policy()
    info = {}
    policy._backward_and_step(_loss(policy), info, epoch=1, accumulation_steps=1,
                              finish_accumulation=True, audit_active=False)
    assert "student_lr_scale" not in info


def test_student_schedule_requires_a_student_group(monkeypatch):
    policy = _policy()
    policy.optim_params = {"policy": {"student_lr_warmup_steps": 10}}
    monkeypatch.setattr(Algo, "_create_optimizers", lambda self: None)
    with pytest.raises(RuntimeError, match="'student' optimizer group"):
        policy._create_optimizers()


class _GroupedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = nn.Linear(2, 2)
        self.adapter = nn.Linear(2, 2)

    def get_parameter_groups(self, base_lr, backbone_lr_multiplier=1.0, head_lr_multiplier=1.0,
                             student_lr_multiplier=None, **_unused):
        if student_lr_multiplier is None:
            return [{"params": list(self.parameters()), "lr": base_lr * backbone_lr_multiplier, "name": "backbone"}]
        return [
            {"params": list(self.unet.parameters()), "lr": base_lr * student_lr_multiplier, "name": "student"},
            {"params": list(self.adapter.parameters()), "lr": base_lr * backbone_lr_multiplier, "name": "alignment"},
        ]


def test_student_schedule_splits_the_student_group_at_the_same_lr():
    net = nn.ModuleDict({"encoder": _GroupedEncoder(), "head": nn.Linear(2, 2)})
    groups = TorchUtils._parameter_groups_from_modules({}, net, 1e-3)
    assert [group["name"] for group in groups] == ["backbone", "policy"]
    groups = TorchUtils._parameter_groups_from_modules({"student_freeze_steps": 5}, net, 1e-3)
    assert [(group["name"], group["lr"]) for group in groups] == [
        ("student", 1e-3), ("alignment", 1e-3), ("policy", 1e-3)
    ]
