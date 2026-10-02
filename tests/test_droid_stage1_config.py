"""Stage-I configuration: paper defaults, every camera reaches the policy, strict inputs."""

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_policy_learning"))
sys.path.insert(0, str(ROOT))

import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from robomimic.models.obs_nets import _accepts_lang_cond  # noqa: E402

import train_droid_auto  # noqa: E402

HAND = "camera/image/hand_camera_left_image"
EXT1 = "camera/image/varied_camera_1_left_image"
EXT2 = "camera/image/varied_camera_2_left_image"


def _config(tmp_path, monkeypatch, *extra):
    monkeypatch.setenv("WORK", str(tmp_path))
    args = train_droid_auto.build_parser().parse_args(["--name", "unit", "--config_only", *extra])
    train_droid_auto.validate_args(args)
    return args, train_droid_auto.create_droid_config(args)


def test_cli_defaults_are_the_paper_protocol(tmp_path, monkeypatch):
    args, config = _config(tmp_path, monkeypatch)
    algo = config.algo
    assert list(config.observation.modalities.obs.rgb) == [HAND, EXT1, EXT2]
    assert list(config.observation.modalities.obs.low_dim) == []
    assert algo.robot_dift_readout == "paper"
    assert algo.optim_params.policy.optimizer_type == "adam"
    assert algo.optim_params.policy.regularization.L2 == 0.0
    assert algo.optim_params.policy.backbone_lr_multiplier == 1.0
    assert algo.optim_params.policy.learning_rate.scheduler_type == "linear"
    assert algo.optim_params.policy.learning_rate.epoch_schedule == [300000]
    assert algo.cleandift_alignment_weight == pytest.approx(0.1)
    assert algo.cleandift_alignment_weight * algo.cleandift_alignment_min_decay_factor == pytest.approx(0.001)
    assert algo.cleandift_alignment_warmdown_steps == 150000
    assert algo.cleandift_alignment_schedule_origin == "absolute"
    assert algo.ema.power == 0.75
    assert algo.optim_params.policy.student_freeze_steps == 0
    assert algo.optim_params.policy.student_lr_warmup_steps == 0
    student = config.observation.encoder.rgb.core_kwargs.backbone_kwargs
    assert student["student_init"] == "sd_teacher"
    assert list(student["alignment_feature_keys"]) == [
        "mid", "us1", "us2", "us3", "us4", "us5", "us6", "us7", "us8", "us9", "us10"
    ]
    assert student["num_t_stratification_bins"] == 1 and (student["t_min"], student["t_max"]) == (1, 1000)
    assert student["alignment_layer_reduction"] == "sum" and student["alignment_apply_feature_adapter"] is True
    assert student["vae_latent_mode"] == "mode"
    assert config.train.amp_dtype == "bfloat16"
    assert config.experiment.epoch_every_n_steps == 1


def test_checkpoint_init_is_recorded_and_clip_is_required(tmp_path, monkeypatch):
    args, config = _config(tmp_path, monkeypatch, "--cleandift_checkpoint", "/tmp/robot_dift_init")
    assert config.observation.encoder.rgb.core_kwargs.backbone_kwargs["student_init"] == "cleandift"
    args = train_droid_auto.build_parser().parse_args(["--name", "unit", "--clip_model", str(tmp_path / "none.pt")])
    with pytest.raises(FileNotFoundError, match="CLIP"):
        train_droid_auto.validate_args(args)
    args = train_droid_auto.build_parser().parse_args(["--name", "unit", "--config_only", "--cameras", "hand_left", "hand_camera_left"])
    with pytest.raises(ValueError, match="Duplicate"):
        train_droid_auto.validate_args(args)


def test_student_schedule_options_reach_the_config(tmp_path, monkeypatch):
    _, config = _config(tmp_path, monkeypatch, "--student_freeze_steps", "2000",
                        "--student_lr_warmup_steps", "3000")
    policy = config.algo.optim_params.policy
    assert (policy.student_freeze_steps, policy.student_lr_warmup_steps) == (2000, 3000)


def test_droid_shape_metadata_covers_every_configured_camera():
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"rgb": [HAND, EXT1, EXT2], "low_dim": []}})
    batch = {
        "obs": {key: torch.zeros(2, 2, 256, 256, 3) for key in (HAND, EXT1, EXT2)},
        "actions": torch.zeros(2, 16, 10),
    }
    batch["obs"]["raw_language"] = ["a", "b"]
    meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=None, batch=batch, action_keys=[], all_obs_keys=[EXT1, EXT2, HAND], ds_format="droid_rlds",
    )
    assert set(meta["all_shapes"]) == {HAND, EXT1, EXT2}
    del batch["obs"][HAND]
    with pytest.raises(KeyError, match="hand_camera"):
        FileUtils.get_shape_metadata_from_dataset(
            dataset_path=None, batch=batch, action_keys=[], all_obs_keys=[EXT1, EXT2, HAND], ds_format="droid_rlds",
        )


def test_linear_schedule_honours_warmup():
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    params = {"learning_rate": {
        "scheduler_type": "linear", "epoch_schedule": [10], "decay_factor": 0.1,
        "warmup_epochs": 2, "warmup_start_factor": 0.5,
    }}
    scheduler = TorchUtils.lr_scheduler_from_optim_params(params, None, optimizer)
    lrs = []
    for _ in range(12):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[2] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(0.1) and lrs[11] == pytest.approx(0.1)
    no_warmup = TorchUtils.lr_scheduler_from_optim_params(
        {"learning_rate": {**params["learning_rate"], "warmup_epochs": 0}}, None,
        torch.optim.Adam([nn.Parameter(torch.zeros(()))], lr=1.0),
    )
    assert isinstance(no_warmup, torch.optim.lr_scheduler.LinearLR)


def test_language_is_passed_only_to_encoders_that_declare_it():
    class WithLanguage(nn.Module):
        def forward(self, inputs, lang_cond=None):
            return inputs

    class WithoutLanguage(nn.Module):
        def forward(self, inputs):
            return inputs

    assert _accepts_lang_cond(WithLanguage)
    assert not _accepts_lang_cond(WithoutLanguage)


def test_release_launcher_matches_the_paper_protocol():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/release/check_paper_protocol.py"), "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
    report = json.loads(result.stdout)
    assert report["protocol_match"] is True
    assert report["processes"] == 8


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--cleandift_alignment_feature_keys", "us3", "us99"], "alignment_feature_keys"),
        (["--max_grad_norm", "0"], "max_grad_norm"),
        (["--ema_power", "-1"], "ema_power"),
        (["--align_decay_power", "0.5"], "align_decay_power"),
        (["--save_robot_dift_full_encoder"], "legacy readout"),
        (["--student_freeze_steps", "-1"], "student_freeze_steps"),
        (["--student_lr_warmup_steps", "-5"], "student_lr_warmup_steps"),
    ],
)
def test_invalid_stage1_arguments_fail_before_data_loading(extra, message):
    args = train_droid_auto.build_parser().parse_args(["--name", "unit", "--config_only", *extra])
    with pytest.raises(ValueError, match=message):
        train_droid_auto.validate_args(args)


def test_default_save_steps_beyond_a_short_run_are_dropped():
    args = train_droid_auto.build_parser().parse_args(
        ["--name", "unit", "--config_only", "--num_epochs", "2000", "--save_steps", "1000", "5000"]
    )
    train_droid_auto.validate_args(args)
    assert args.save_steps == [1000]


def test_full_encoder_state_without_student_weights_is_rejected():
    from agents.encoders.cleandift_img_encoder import CleanDIFTImgEncoder

    encoder = object.__new__(CleanDIFTImgEncoder)
    nn.Module.__init__(encoder)
    encoder.model = nn.Module()
    encoder.model.unet_feature_extractor_cleandift = nn.Linear(2, 2)
    encoder.strict_full_encoder_checkpoint = False
    encoder._pending_full_encoder_state = {"student.model.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="lacks Student U-Net"):
        encoder._load_pending_full_encoder_state()
