#!/usr/bin/env python
"""Robot-DIFT Stage I: adapt the diffusion Student on DROID (paper protocol by default).

One optimizer step minimizes ``L = L_policy + lambda(s) L_align`` (paper Eq. 4):

* The clean-input Student (a copy of the SD2.1 U-Net) encodes every camera
  view. The S2-FPN + frozen-CLIP readout turns the raw ``us3/us6/us8`` maps of
  all views into one token per frame for a 1D U-Net diffusion policy.
* ``L_align`` = sum over 11 decoder maps of (1 - cos) between timestep-conditioned
  adapters of the Student maps and the frozen SD2.1 Teacher at one uniformly
  sampled timestep t in [1, 999] per view. lambda decays linearly 0.1 -> 0.001
  over 150k steps, then stays constant.

Every default below reproduces the paper protocol; see docs/STAGE1_PROTOCOL.md.
Each robomimic "epoch" is exactly one optimizer step (``--steps_per_epoch 1``).
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import torch

droid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "droid_policy_learning")
sys.path.insert(0, droid_path)

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
torch.set_float32_matmul_precision("high")


PAPER_ALIGNMENT_KEYS = ["mid", "us1", "us2", "us3", "us4", "us5", "us6", "us7", "us8", "us9", "us10"]
PAPER_FEATURE_KEYS = ["us3", "us6", "us8"]
PAPER_CAMERAS = ["hand_camera_left", "varied_camera_1_left", "varied_camera_2_left"]
CAMERA_KEYS = {
    "hand_camera_left": "camera/image/hand_camera_left_image",
    "hand_left": "camera/image/hand_camera_left_image",
    "wrist_left": "camera/image/wrist_image_left",
    "wrist_image_left": "camera/image/wrist_image_left",
    "varied_camera_1_left": "camera/image/varied_camera_1_left_image",
    "exterior_1": "camera/image/varied_camera_1_left_image",
    "exterior_image_1_left": "camera/image/varied_camera_1_left_image",
    "varied_camera_2_left": "camera/image/varied_camera_2_left_image",
    "exterior_2": "camera/image/varied_camera_2_left_image",
    "exterior_image_2_left": "camera/image/varied_camera_2_left_image",
}


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    run = parser.add_argument_group("run")
    run.add_argument("--name", type=str, required=True, help="experiment name")
    run.add_argument("--seed", type=int, default=0, help="training seed; rank r uses seed + r")
    run.add_argument("--debug", action="store_true", help="debug mode (no wandb)")
    run.add_argument("--no_wandb", action="store_true", help="disable wandb")
    run.add_argument("--wandb_proj_name", type=str, default=None)
    run.add_argument("--verbose_config", action="store_true", help="print the full resolved config")
    run.add_argument("--config_only", action="store_true",
                     help="validate and save the resolved config without loading data or a model")
    run.add_argument("--use_ddp", action="store_true", help="multi-GPU training (set by the launcher under torchrun)")

    data = parser.add_argument_group("data")
    data.add_argument("--data_path", type=str, default=None,
                      help="TFDS root containing droid/ (default: ROBOT_DIFT_DATA_ROOT or $WORK/datasets/robot_dift)")
    data.add_argument("--dataset_names", type=str, nargs="+", default=["droid"])
    data.add_argument("--cameras", type=str, nargs="+", default=list(PAPER_CAMERAS),
                      help=f"camera views (2-3); names: {sorted(CAMERA_KEYS)}")
    data.add_argument("--shuffle_buffer_size", type=_positive_int, default=50000,
                      help="per-rank frame shuffle buffer (holds encoded JPEG frames before decoding)")
    data.add_argument("--num_parallel_calls", type=_positive_int, default=64)
    data.add_argument("--traj_transform_threads", type=_positive_int, default=16)
    data.add_argument("--traj_read_threads", type=_positive_int, default=16)
    data.add_argument("--rlds_deterministic", action="store_true",
                      help="enable TensorFlow op determinism (slower; RLDS order still depends on threads)")
    data.add_argument("--lang_dropout_prob", type=float, default=0.0)

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument("--num_epochs", type=_positive_int, default=300000,
                          help="final optimizer step to reach (paper: 300000)")
    schedule.add_argument("--steps_per_epoch", type=_positive_int, default=1,
                          help="optimizer steps per robomimic epoch; keep 1 so every schedule counts steps")
    schedule.add_argument("--batch_size", type=_positive_int, default=32, help="per-GPU batch (paper run: 8 GPUs x 32, no accumulation)")
    schedule.add_argument("--gradient_accumulation_steps", type=_positive_int, default=1)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--optimizer_type", choices=["adam", "adamw", "muon"], default="adam")
    optim.add_argument("--policy_lr", type=float, default=1e-4)
    optim.add_argument("--lr_scheduler_type", choices=["linear", "cosine_warmup"], default="linear")
    optim.add_argument("--lr_total_epochs", type=int, default=None, help="LR schedule horizon (default: num_epochs)")
    optim.add_argument("--lr_linear_end_factor", type=float, default=0.1,
                       help="final/initial LR of the linear schedule (not stated by the paper)")
    optim.add_argument("--lr_warmup_epochs", type=int, default=0)
    optim.add_argument("--lr_warmup_start_factor", type=float, default=0.01)
    optim.add_argument("--lr_min_ratio", type=float, default=1e-4, help="cosine_warmup floor")
    optim.add_argument("--weight_decay", type=float, default=0.0, help="weight decay of all groups")
    optim.add_argument("--backbone_weight_decay", type=float, default=None)
    optim.add_argument("--head_weight_decay", type=float, default=None)
    optim.add_argument("--backbone_lr_multiplier", type=float, default=1.0)
    optim.add_argument("--student_freeze_steps", type=int, default=0,
                       help="optimizer steps that keep the Student fixed while the readout, policy, "
                            "and alignment adapters train (0 = paper protocol)")
    optim.add_argument("--student_lr_warmup_steps", type=int, default=0,
                       help="linear Student LR warmup after the freeze (0 = paper protocol)")
    optim.add_argument("--head_lr_multiplier", type=float, default=1.0)
    optim.add_argument("--student_lr_multiplier", type=float, default=None,
                       help="optional separate LR multiplier for the Student U-Net and timestep")
    optim.add_argument("--max_grad_norm", type=float, default=1.0)
    optim.add_argument("--amp_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    optim.add_argument("--ema_power", type=float, default=0.75)
    optim.add_argument("--muon_momentum", type=float, default=0.95)
    optim.add_argument("--muon_nesterov", action=argparse.BooleanOptionalAction, default=True)
    optim.add_argument("--muon_ns_steps", type=int, default=5)
    optim.add_argument("--muon_adjust_lr_fn", type=str, default="match_rms_adamw",
                       choices=["original", "match_rms_adamw"])
    optim.add_argument("--muon_scope", type=str, default="all", choices=["all", "backbone", "head", "policy"])

    dift = parser.add_argument_group("Robot-DIFT Student and Teacher alignment")
    dift.add_argument("--stage1_readout", choices=["paper", "legacy"], default="paper",
                      help="paper: S2-FPN + frozen CLIP readout shared with Stage II; legacy: CleanDIFT S2-FPN/query head")
    dift.add_argument("--clip_model", type=str, default=os.environ.get("ROBOT_DIFT_CLIP_MODEL"),
                      help="local CLIP ViT-B/32 weight file for the paper readout (ROBOT_DIFT_CLIP_MODEL)")
    dift.add_argument("--cleandift_student_init", choices=["sd_teacher", "cleandift"], default="sd_teacher",
                      help="sd_teacher copies the SD2.1 U-Net (paper); cleandift loads CompVis/cleandift")
    dift.add_argument("--cleandift_checkpoint", type=str, default=None,
                      help="initialize the Student from a Robot-DIFT checkpoint instead")
    dift.add_argument("--cleandift_vae_latent_mode", choices=["mode", "sample"], default="mode",
                      help="mode keeps the deployed Student deterministic")
    dift.add_argument("--alignment_weight", type=float, default=0.1, help="lambda_0")
    dift.add_argument("--align_warmdown_steps", type=int, default=150000, help="T_decay")
    dift.add_argument("--align_min_decay_factor", type=float, default=0.01, help="lambda_min / lambda_0")
    dift.add_argument("--align_decay_power", type=float, default=1.0, help="1 = linear decay")
    dift.add_argument("--align_schedule_origin", choices=["absolute", "first_active"], default="absolute")
    dift.add_argument("--cleandift_alignment_feature_keys", nargs="+", default=list(PAPER_ALIGNMENT_KEYS))
    dift.add_argument("--cleandift_alignment_layer_reduction", choices=["sum", "mean"], default="sum")
    dift.add_argument("--cleandift_alignment_apply_feature_adapter", action=argparse.BooleanOptionalAction,
                      default=True, help="align through training-only timestep adapters")
    dift.add_argument("--cleandift_num_t_bins", type=_positive_int, default=1,
                      help="Teacher timesteps per view and step")
    dift.add_argument("--cleandift_t_min", type=int, default=1)
    dift.add_argument("--cleandift_t_max", type=int, default=1000, help="exclusive upper bound")
    dift.add_argument("--cleandift_use_fp32", action="store_true",
                      help="disable the encoder's own autocast; --amp_dtype still applies")

    legacy = parser.add_argument_group("legacy readout (only with --stage1_readout legacy)")
    legacy.add_argument("--legacy_output_mode", choices=["pooled", "queries"], default="queries")
    legacy.add_argument("--legacy_fpn_num_queries", type=_positive_int, default=8)
    legacy.add_argument("--legacy_fusion_mode", choices=["s2fpn", "global_to_fine", "concat"], default="s2fpn")

    save = parser.add_argument_group("checkpoints and export")
    save.add_argument("--checkpoint_dir", type=str, default=None, help="full policy checkpoints")
    save.add_argument("--save_cleandift_dir", type=str, default=None, help="Student/readout export directory")
    save.add_argument("--save_freq", type=_positive_int, default=10000)
    save.add_argument("--save_steps", nargs="+", type=int, default=None, help="additional exact save steps")
    save.add_argument("--save_cleandift_ema", action=argparse.BooleanOptionalAction, default=True,
                      help="also export the EMA Student (the release candidate)")
    save.add_argument("--save_cleandift_full_state", action="store_true")
    save.add_argument("--save_robot_dift_full_encoder", action="store_true")
    save.add_argument("--require_encoder_export", action="store_true",
                      help="fail training if an encoder export fails (release runs)")
    save.add_argument("--resume_from", type=str, default=None, help="full policy checkpoint (.pth) to resume")

    diag = parser.add_argument_group("diagnostics")
    diag.add_argument("--representation_audit_steps", type=int, nargs="*", default=[])
    diag.add_argument("--feature_vis_dir", type=str, default=None)
    diag.add_argument("--feature_vis_max_images", type=int, default=8)
    diag.add_argument("--feature_vis_grid_nrow", type=int, default=4)
    diag.add_argument("--feature_vis_prefix", type=str, default="RobotDIFT")
    diag.add_argument("--feature_vis_every", type=int, default=10000)
    diag.add_argument("--feature_vis_num_clusters", type=int, default=6)
    return parser


def validate_args(args):
    if not 2 <= len(args.cameras) <= 3:
        raise ValueError(f"2-3 cameras required, got {args.cameras}")
    unknown = [camera for camera in args.cameras if camera not in CAMERA_KEYS]
    if unknown:
        raise ValueError(f"Unknown cameras {unknown}; use {sorted(CAMERA_KEYS)}")
    if len({CAMERA_KEYS[camera] for camera in args.cameras}) != len(args.cameras):
        raise ValueError(f"Duplicate camera views: {args.cameras}")
    if args.stage1_readout == "paper" and not args.config_only:
        if not args.clip_model or not os.path.isfile(os.path.expanduser(args.clip_model)):
            raise FileNotFoundError(
                "The paper readout needs a local CLIP ViT-B/32 file (--clip_model or ROBOT_DIFT_CLIP_MODEL); "
                f"got {args.clip_model!r}"
            )
    if args.student_freeze_steps < 0 or args.student_lr_warmup_steps < 0:
        raise ValueError("--student_freeze_steps and --student_lr_warmup_steps must be >= 0")
    if args.cleandift_checkpoint:
        if not args.config_only and not os.path.exists(os.path.expanduser(args.cleandift_checkpoint)):
            raise FileNotFoundError(f"--cleandift_checkpoint does not exist: {args.cleandift_checkpoint}")
        if args.cleandift_student_init == "sd_teacher":
            # A checkpoint replaces the SD2.1 copy; record the actual initialization.
            args.cleandift_student_init = "cleandift"
    alignment_keys = list(args.cleandift_alignment_feature_keys)
    unknown_keys = [key for key in alignment_keys if key not in PAPER_ALIGNMENT_KEYS]
    if not alignment_keys or unknown_keys or len(set(alignment_keys)) != len(alignment_keys):
        raise ValueError(
            f"--cleandift_alignment_feature_keys must be unique keys from {PAPER_ALIGNMENT_KEYS}; got {alignment_keys}"
        )
    if not 1 <= args.cleandift_t_min < args.cleandift_t_max <= 1000:
        raise ValueError("Teacher timesteps must satisfy 1 <= t_min < t_max <= 1000")
    for name in ("alignment_weight", "weight_decay", "lang_dropout_prob", "lr_linear_end_factor"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    for name in ("policy_lr", "max_grad_norm", "ema_power", "backbone_lr_multiplier", "head_lr_multiplier"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    if not 0.0 <= args.align_min_decay_factor <= 1.0:
        raise ValueError("--align_min_decay_factor must be in [0, 1]")
    if not math.isfinite(args.align_decay_power) or args.align_decay_power < 1.0:
        raise ValueError("--align_decay_power must be >= 1 (1 is the paper's linear decay)")
    if args.align_warmdown_steps < 1:
        raise ValueError("--align_warmdown_steps must be positive")
    if args.student_lr_multiplier is not None and (
        not math.isfinite(args.student_lr_multiplier) or args.student_lr_multiplier <= 0
    ):
        raise ValueError("--student_lr_multiplier must be finite and positive")
    if args.save_robot_dift_full_encoder and args.stage1_readout == "paper":
        raise ValueError("--save_robot_dift_full_encoder applies to the legacy readout; "
                         "the paper readout is exported in the deploy head")
    if any(step < 1 for step in args.representation_audit_steps):
        raise ValueError("--representation_audit_steps must be positive optimizer steps")
    if args.require_encoder_export and not args.save_cleandift_dir:
        raise ValueError("--require_encoder_export requires --save_cleandift_dir")
    save_steps = sorted(set(args.save_steps or []))
    if any(step < 1 for step in save_steps):
        raise ValueError("--save_steps must be positive optimizer steps")
    # Launcher defaults (1k, 5k) may exceed a short smoke or ablation run.
    args.save_steps = [step for step in save_steps if step <= args.num_epochs]


def student_kwargs(args):
    """Keyword arguments of the Stage-I CleanDIFTImgEncoder Student."""
    return {
        "sd_version": "sd21",
        "feature_key": list(PAPER_FEATURE_KEYS),
        "freeze_backbone": False,
        "use_text_condition": True,
        "use_fp32": bool(args.cleandift_use_fp32),
        "custom_checkpoint": args.cleandift_checkpoint,
        "num_t_stratification_bins": int(args.cleandift_num_t_bins),
        "t_min": int(args.cleandift_t_min),
        "t_max": int(args.cleandift_t_max),
        "vae_latent_mode": args.cleandift_vae_latent_mode,
        "alignment_apply_feature_adapter": bool(args.cleandift_alignment_apply_feature_adapter),
        "alignment_feature_keys": list(args.cleandift_alignment_feature_keys),
        "alignment_layer_reduction": args.cleandift_alignment_layer_reduction,
        "student_init": args.cleandift_student_init,
    }


def create_droid_config(args):
    name = "debug" if args.debug else datetime.datetime.now().strftime("%m-%d-") + str(args.name)
    config = config_factory("diffusion_policy")

    if args.data_path:
        data_path = os.path.expanduser(args.data_path)
    else:
        data_root = os.environ.get("ROBOT_DIFT_DATA_ROOT")
        data_path = os.path.expanduser(os.path.expandvars(data_root)) if data_root else os.path.join(
            os.environ.get("WORK", os.getcwd()), "datasets", "robot_dift"
        )
    log_root = os.path.join(os.environ.get("WORK", os.getcwd()), "logs")
    camera_keys = [CAMERA_KEYS[camera] for camera in args.cameras]
    # Record the requested precision; the policy enables autocast only on CUDA.
    use_amp = args.amp_dtype != "float32"

    backbone_kwargs = student_kwargs(args)
    if args.stage1_readout == "legacy":
        backbone_kwargs.update(
            pretrained=True,
            map_out_dim=512,
            fpn_dim=256,
            fpn_num_queries=int(args.legacy_fpn_num_queries),
            fpn_dropout=0.0,
            layer_scale_init=0.1,
            fusion_mode=args.legacy_fusion_mode,
            output_mode=args.legacy_output_mode,
            alignment_weight=float(args.alignment_weight),
        )

    with config.values_unlocked():
        config.experiment.name = name
        config.experiment.validate = False
        config.experiment.verbose_config = args.verbose_config
        config.experiment.logging.terminal_output_to_txt = True
        config.experiment.logging.log_tb = True
        config.experiment.logging.log_wandb = not (args.no_wandb or args.debug)
        config.experiment.logging.wandb_proj_name = args.wandb_proj_name or ("debug" if args.debug else None)
        config.experiment.reset_optimizer_on_resume = False
        config.experiment.auto_remove_exp_dir = True

        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = args.save_freq
        config.experiment.save.epochs = args.save_steps
        config.experiment.save.on_best_rollout_success_rate = False
        config.experiment.ckpt_path = args.resume_from

        if args.save_cleandift_dir:
            os.makedirs(args.save_cleandift_dir, exist_ok=True)
        config.experiment.save_cleandift_dir = args.save_cleandift_dir
        config.experiment.save_cleandift_ema = bool(args.save_cleandift_dir and args.save_cleandift_ema)
        config.experiment.save_robot_dift_full_encoder = bool(args.save_robot_dift_full_encoder)
        config.experiment.save_cleandift_full_state = bool(args.save_cleandift_full_state)
        config.experiment.require_encoder_export = bool(args.require_encoder_export)

        config.experiment.epoch_every_n_steps = args.steps_per_epoch
        config.experiment.validation_epoch_every_n_steps = 100
        config.experiment.rollout.enabled = False
        config.experiment.env = None
        config.experiment.render = False
        config.experiment.render_video = False
        config.experiment.mse.enabled = True
        config.experiment.mse.every_n_epochs = 50000
        config.experiment.mse.on_save_ckpt = False
        config.experiment.mse.num_samples = 6
        config.experiment.mse.visualize = False
        config.experiment.feature_vis = None if not args.feature_vis_dir else {
            "image_dir": args.feature_vis_dir,
            "max_images_per_key": args.feature_vis_max_images,
            "apply_token_mapper": False,
            "grid_nrow": args.feature_vis_grid_nrow,
            "log_prefix": args.feature_vis_prefix,
            "image_keys": None,
            "frequency": args.feature_vis_every,
            "num_clusters": args.feature_vis_num_clusters,
        }

        # Data: DROID RLDS, 256x256 images in [-1, 1], no augmentation.
        config.train.data_format = "droid_rlds"
        config.train.data_path = data_path
        config.train.dataset_names = args.dataset_names
        config.train.sample_weights = [1] * len(args.dataset_names)
        config.train.num_epochs = args.num_epochs
        config.train.shuffle_buffer_size = args.shuffle_buffer_size
        config.train.view_dropout_prob = 0.0
        config.train.batch_size = args.batch_size
        config.train.subsample_length = 100
        config.train.num_parallel_calls = args.num_parallel_calls
        config.train.traj_transform_threads = args.traj_transform_threads
        config.train.traj_read_threads = args.traj_read_threads
        config.train.cuda = True
        config.train.seed = args.seed
        config.train.use_ddp = args.use_ddp
        config.train.use_amp = use_amp
        config.train.amp_dtype = args.amp_dtype
        config.train.max_grad_norm = args.max_grad_norm
        config.algo.optim_params.policy.student_freeze_steps = args.student_freeze_steps
        config.algo.optim_params.policy.student_lr_warmup_steps = args.student_lr_warmup_steps
        config.train.output_dir = os.path.join(log_root, "droid", "im", "diffusion_policy")
        config.train.checkpoint_dir = args.checkpoint_dir or os.path.join(config.train.output_dir, "models")
        os.makedirs(config.train.checkpoint_dir, exist_ok=True)
        config.train.action_keys = ["action/abs_pos", "action/abs_rot_6d", "action/gripper_position"]
        config.train.action_shapes = [(1, 3), (1, 6), (1, 1)]
        config.train.action_config = {
            "action/abs_pos": {"normalization": "min_max"},
            "action/abs_rot_6d": {
                "normalization": "min_max",
                "format": "rot_6d",
                "convert_at_runtime": "rot_euler",
            },
            "action/gripper_position": {"normalization": "min_max"},
        }
        config.train.goal_mode = None
        config.train.truncated_geom_factor = 0.3
        config.experiment.rollout.rate = args.num_epochs + 1
        config.experiment.rollout.warmstart = args.num_epochs

        # Optimization (paper Table S1).
        policy_optim = config.algo.optim_params.policy
        policy_optim.optimizer_type = args.optimizer_type
        policy_optim.betas = (0.9, 0.999)
        policy_optim.eps = 1e-8
        policy_optim.learning_rate.initial = float(args.policy_lr)
        policy_optim.learning_rate.decay_factor = float(args.lr_linear_end_factor)
        policy_optim.learning_rate.scheduler_type = args.lr_scheduler_type
        policy_optim.learning_rate.warmup_epochs = int(args.lr_warmup_epochs)
        policy_optim.learning_rate.warmup_type = "linear"
        lr_total_epochs = max(int(args.lr_total_epochs or args.num_epochs), int(args.num_epochs))
        policy_optim.learning_rate.total_epochs = lr_total_epochs
        policy_optim.learning_rate.min_lr_ratio = float(args.lr_min_ratio)
        policy_optim.learning_rate.epoch_schedule = [lr_total_epochs] if args.lr_scheduler_type == "linear" else []
        policy_optim.regularization.L2 = float(args.weight_decay)

        # Policy: 1D U-Net diffusion policy, horizons 2/16/8, DDIM.
        config.algo.horizon.observation_horizon = 2
        config.algo.horizon.action_horizon = 8
        config.algo.horizon.prediction_horizon = 16
        config.algo.unet.enabled = True
        config.algo.unet.diffusion_step_embed_dim = 256
        config.algo.unet.down_dims = [256, 512, 1024]
        config.algo.unet.kernel_size = 5
        config.algo.unet.n_groups = 8
        config.algo.ema.enabled = True
        config.algo.ema.power = float(args.ema_power)
        config.algo.ddpm.enabled = False
        config.algo.ddim.enabled = True
        config.algo.ddim.num_train_timesteps = 100
        config.algo.ddim.num_inference_timesteps = 10
        config.algo.ddim.beta_schedule = "squaredcos_cap_v2"
        config.algo.ddim.clip_sample = True
        config.algo.ddim.set_alpha_to_one = True
        config.algo.ddim.steps_offset = 0
        config.algo.ddim.prediction_type = "epsilon"
        config.algo.noise_samples = 8
        config.algo.language_dropout_prob = float(args.lang_dropout_prob)
        config.algo.rgb_only = True
        config.algo.representation_audit_steps = sorted(set(args.representation_audit_steps))

        # Teacher alignment (paper Eq. 4).
        config.algo.cleandift_alignment_weight = float(args.alignment_weight)
        config.algo.cleandift_alignment_warmdown_steps = args.align_warmdown_steps
        config.algo.cleandift_alignment_min_decay_factor = float(args.align_min_decay_factor)
        config.algo.cleandift_alignment_decay_power = float(args.align_decay_power)
        config.algo.cleandift_alignment_schedule_origin = args.align_schedule_origin

        # Visual interface.
        config.algo.robot_dift_readout = args.stage1_readout
        config.algo.robot_dift_paper_readout.clip_model_path = (
            os.path.expanduser(args.clip_model) if args.clip_model else None
        )

        config.observation.image_dim = [256, 256]
        config.observation.modalities.obs.low_dim = []
        config.observation.modalities.obs.rgb = camera_keys
        rgb = config.observation.encoder.rgb
        rgb.core_class = "VisualCore"
        rgb.core_kwargs.backbone_class = "CleanDIFTConv"
        rgb.core_kwargs.backbone_kwargs = backbone_kwargs
        rgb.core_kwargs.pool_class = None
        rgb.core_kwargs.pool_kwargs = None
        rgb.core_kwargs.feature_dimension = None
        rgb.core_kwargs.flatten = False
        rgb.share_rgb_encoder_across_cameras = True
        # Images are already in [-1, 1]; the paper uses no augmentation.
        rgb.obs_randomizer_class = None
        rgb.obs_randomizer_kwargs = {}
        rgb.fuser = None

    with config.train.unlocked():
        config.train.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        config.train.rlds_deterministic = bool(args.rlds_deterministic)

    policy_optim = config.algo.optim_params.policy
    with policy_optim.unlocked():
        policy_optim.learning_rate.warmup_start_factor = float(args.lr_warmup_start_factor)
        policy_optim.backbone_lr_multiplier = float(args.backbone_lr_multiplier)
        policy_optim.head_lr_multiplier = float(args.head_lr_multiplier)
        policy_optim.backbone_weight_decay = float(
            args.weight_decay if args.backbone_weight_decay is None else args.backbone_weight_decay
        )
        policy_optim.head_weight_decay = float(
            args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
        )
        if args.student_lr_multiplier is not None:
            policy_optim.student_lr_multiplier = float(args.student_lr_multiplier)
        if args.optimizer_type == "muon":
            policy_optim.muon_momentum = args.muon_momentum
            policy_optim.muon_nesterov = args.muon_nesterov
            policy_optim.muon_ns_steps = args.muon_ns_steps
            policy_optim.muon_adjust_lr_fn = args.muon_adjust_lr_fn
            policy_optim.muon_scope = args.muon_scope

    config.lock()
    return config


def print_summary(args, config):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.batch_size * args.gradient_accumulation_steps * world
    lines = [
        "=" * 80,
        "Robot-DIFT Stage I (DROID)",
        "=" * 80,
        f"  Run: {config.experiment.name}  seed={args.seed}  ranks={world}",
        f"  Data: {config.train.data_path}  cameras={list(config.observation.modalities.obs.rgb)}",
        f"  Batch: {args.batch_size}/GPU x accumulation {args.gradient_accumulation_steps} x {world} = {global_batch}",
        f"  Steps: {args.num_epochs}  optimizer={args.optimizer_type} lr={args.policy_lr} "
        f"schedule={args.lr_scheduler_type}(end x{args.lr_linear_end_factor}) weight_decay={args.weight_decay}",
        f"  AMP: {config.train.amp_dtype}  EMA power: {args.ema_power}  shuffle buffer: {args.shuffle_buffer_size}",
        f"  Readout: {args.stage1_readout}  CLIP: {args.clip_model}",
        f"  Student init: {args.cleandift_student_init}  checkpoint: {args.cleandift_checkpoint}  "
        f"VAE latent: {args.cleandift_vae_latent_mode}",
        f"  Alignment: lambda {args.alignment_weight} -> {args.alignment_weight * args.align_min_decay_factor:g} "
        f"over {args.align_warmdown_steps} steps (power {args.align_decay_power}, {args.align_schedule_origin}); "
        f"{len(args.cleandift_alignment_feature_keys)} maps, {args.cleandift_alignment_layer_reduction}, "
        f"adapters={args.cleandift_alignment_apply_feature_adapter}, t in [{args.cleandift_t_min}, "
        f"{args.cleandift_t_max}) x{args.cleandift_num_t_bins}",
        f"  Export: {args.save_cleandift_dir} (EMA={config.experiment.save_cleandift_ema})",
        "=" * 80,
    ]
    print("\n".join(lines), flush=True)


def main():
    args = build_parser().parse_args()
    validate_args(args)

    # One precision setting drives the policy autocast and the encoder's own autocast.
    os.environ["ROBOT_DIFT_DROID_AMP_DTYPE"] = args.amp_dtype
    if os.environ.get("ROBOT_DIFT_DISABLE_CUDNN", "0").lower() in {"1", "true", "yes"}:
        torch.backends.cudnn.enabled = False
        print("[Robot-DIFT] cuDNN disabled via ROBOT_DIFT_DISABLE_CUDNN")

    config = create_droid_config(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print_summary(args, config)
    # Keep the resolved configuration beside this run's policy checkpoints.
    # A shared output_dir is used by multiple concurrent experiments and DDP
    # ranks; a single auto_generated_config.json there can be overwritten.
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(config.train.checkpoint_dir, exist_ok=True)
        config_path = os.path.join(config.train.checkpoint_dir, "auto_generated_config.json")
        with open(config_path, "w") as f:
            json.dump(json.loads(config.dump()), f, indent=4)
        print(f"Configuration saved to: {config_path}")
    if args.config_only:
        print("Configuration-only preflight completed.")
        return

    # Imported here so --config_only works without TensorFlow/Octo installed.
    from robomimic.scripts.train import train

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    start = time.time()
    train(config, device=device)
    print(f"Training completed in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
