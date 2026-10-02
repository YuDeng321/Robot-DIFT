"""Robot-DIFT Stage-I diffusion policy (robomimic ``diffusion_policy`` algo).

One optimizer step minimizes ``L = L_policy + lambda(s) * L_align``:

* ``L_policy``: epsilon-prediction loss of a 1D U-Net action diffusion model
  (observation/prediction/action horizons 2/16/8), conditioned on the visual
  tokens of every observation frame and, for the paper readout, on the frozen
  CLIP language goal.
* ``L_align``: Teacher alignment of the clean-input Student (see
  :class:`StableFeatureAligner`), annealed by ``alignment_weight_at_step``.

``algo.robot_dift_readout`` selects the visual interface. ``"paper"`` uses
:class:`RobotDIFTStage1Encoder` (S2-FPN + CLIP readout shared with Stage II).
``"legacy"`` uses the per-camera robomimic encoder with the CleanDIFT
S2-FPN/query head.
"""

from typing import Callable, Union
import math
import os
import re
from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
from packaging.version import parse as parse_version

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.distributed_gradients import synchronize_optimizer_gradients
from robomimic.utils.representation_audit import RepresentationAudit, select_student_parameters

from robomimic.algo import register_algo_factory_func, PolicyAlgo


def alignment_weight_at_step(
    base_weight, epoch, first_active_epoch, origin, warmdown_steps,
    min_decay_factor, decay_power,
):
    """Evaluate the alignment schedule on a stable optimizer-step clock.

    ``epoch`` is one-based because the DROID loop treats each epoch as one
    optimizer update. The absolute origin therefore survives checkpoint resume.
    With ``min_decay_factor=0.01``, ``decay_power=1`` and a 0.1 base weight
    this is the paper's linear 0.1 -> 0.001 anneal (Eq. 4).
    """
    if origin == "absolute":
        elapsed = max(0, int(epoch) - 1)
    elif origin == "first_active":
        elapsed = int(epoch) - int(first_active_epoch)
    else:
        raise ValueError(f"Unsupported alignment schedule origin: {origin}")
    warmdown_steps = max(1, int(warmdown_steps))
    minimum = max(0.0, min(1.0, float(min_decay_factor)))
    power = max(1.0, float(decay_power))
    if elapsed >= warmdown_steps:
        return float(base_weight) * minimum
    remaining = 1.0 - float(elapsed) / float(warmdown_steps)
    return float(base_weight) * (minimum + (1.0 - minimum) * remaining ** power)


def _config_get(config, key, default=None):
    """Read an optional config key.

    A locked robomimic Config raises on attribute access to a missing key, so
    ``getattr(config, key, default)`` never returns the default. Configs are
    dicts; plain namespaces (tests) also work.
    """
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _grad_norms_by_group(optimizer) -> "OrderedDict[str, torch.Tensor]":
    """Pre-clipping gradient norm of every optimizer group, without host syncs."""
    norms = OrderedDict()
    for index, group in enumerate(optimizer.param_groups):
        grads = [parameter.grad for parameter in group["params"] if parameter.grad is not None]
        name = str(group.get("name", f"group_{index}"))
        if not grads:
            continue
        foreach_norm = getattr(torch, "_foreach_norm", None)
        per_tensor = foreach_norm(grads) if callable(foreach_norm) else [grad.norm() for grad in grads]
        norms[name] = torch.linalg.vector_norm(torch.stack([value.float() for value in per_tensor]))
    return norms


@contextmanager
def _scaled_student_lr(optimizer, scale):
    """Scale the Student groups' LR for one optimizer step, then restore it.

    Restoring right after the step keeps every LR scheduler (chainable or not)
    unaware of the Student schedule. Adam's update is proportional to the LR,
    so a scale of 0 leaves the Student weights exactly unchanged.
    """
    groups = [] if scale is None or scale == 1.0 else [
        group for group in optimizer.param_groups if group.get("name") in TorchUtils.STUDENT_GROUP_NAMES
    ]
    saved = [group["lr"] for group in groups]
    for group in groups:
        group["lr"] = group["lr"] * scale
    try:
        yield
    finally:
        for group, lr in zip(groups, saved):
            group["lr"] = lr


def _scalar_floats(values) -> "OrderedDict[str, float]":
    """Convert a mapping of scalar tensors/floats with a single device sync."""
    tensors = OrderedDict((key, value) for key, value in values.items() if torch.is_tensor(value))
    result = OrderedDict((key, float(value)) for key, value in values.items() if not torch.is_tensor(value))
    if tensors:
        stacked = torch.stack([value.detach().float().reshape(()) for value in tensors.values()]).tolist()
        result.update(zip(tensors.keys(), stacked))
    return result


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def __init__(self, *args, **kwargs):
        # Track when alignment is first active; only the "first_active"
        # schedule origin reads it. It is checkpointed for resume.
        self._alignment_start_epoch = None
        self._optimizer_stepped_since_scheduler = False
        self._nonfinite_gradient_skips = 0
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch):
        # GradScaler can reject an update after an overflow. Keep the learning
        # rate on the same optimizer-step clock as EMA and the parameters.
        if self._optimizer_stepped_since_scheduler:
            super().on_epoch_end(epoch)
        self._optimizer_stepped_since_scheduler = False

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    @property
    def uses_paper_readout(self) -> bool:
        return str(_config_get(self.algo_config, "robot_dift_readout", "legacy") or "legacy").lower() == "paper"

    def _rgb_keys(self):
        return [key for key in self.obs_shapes if ObsUtils.OBS_KEYS_TO_MODALITIES.get(key) == "rgb"]

    def _create_paper_encoder(self):
        from agents.encoders.robot_dift_stage1_encoder import RobotDIFTStage1Encoder

        rgb_keys = self._rgb_keys()
        extra = [key for key in self.obs_shapes if key not in rgb_keys]
        if extra:
            raise ValueError(f"The paper Stage-I encoder is RGB-only; unexpected observation keys: {extra}")
        rgb_config = self.obs_config.encoder.rgb
        student_kwargs = dict(rgb_config.core_kwargs.backbone_kwargs)
        readout_kwargs = dict(self.algo_config.robot_dift_paper_readout)
        clip_model_path = readout_kwargs.pop("clip_model_path", None)
        # Build on the algo's device (e.g. cuda:<local_rank>, or cpu for evaluation).
        student_kwargs["device"] = str(self.device)
        return RobotDIFTStage1Encoder(
            rgb_keys=rgb_keys,
            student=student_kwargs,
            clip_model_path=clip_model_path,
            feature_keys=tuple(student_kwargs.get("feature_key", ("us3", "us6", "us8"))),
            **readout_kwargs,
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        To = self.algo_config.horizon.observation_horizon
        if self.uses_paper_readout:
            obs_encoder = self._create_paper_encoder()
            obs_dim, goal_dim = obs_encoder.output_dim, obs_encoder.goal_dim
        else:
            observation_group_shapes = OrderedDict()
            observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
            encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
            obs_encoder = ObsNets.ObservationGroupEncoder(
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
            )
            # Replace all BatchNorm with GroupNorm to work with EMA.
            obs_encoder = replace_bn_with_gn(obs_encoder)
            obs_dim, goal_dim = obs_encoder.output_shape()[0], 0
        self.goal_dim = int(goal_dim)

        unet_config = self.algo_config.unet
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * To + self.goal_dim,
            diffusion_step_embed_dim=int(unet_config.diffusion_step_embed_dim),
            down_dims=[int(width) for width in unet_config.down_dims],
            kernel_size=int(unet_config.kernel_size),
            n_groups=int(unet_config.n_groups),
        )

        use_ddp = bool(_config_get(self.global_config.train, "use_ddp", False))
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not use_ddp and not is_distributed and torch.cuda.device_count() > 1:
            print(
                "[DiffusionPolicy] Several GPUs are visible but DDP is off; training uses one GPU. "
                "Launch with torchrun --use_ddp for multi-GPU training."
            )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net,
            })
        })
        nets = nets.float().to(self.device)
        self._setup_amp()

        # setup noise scheduler
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()

        # EMA over trainable parameters only: frozen Teacher/VAE/CLIP weights
        # never change, so shadowing them only costs memory and bandwidth.
        # use_ema_warmup makes ``power`` effective: decay = 1 - (1 + step)^-power.
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(
                parameters=self._trainable_parameters(nets),
                decay=float(_config_get(self.algo_config.ema, "max_decay", 1.0)),
                min_decay=0.0,
                use_ema_warmup=True,
                inv_gamma=float(_config_get(self.algo_config.ema, "inv_gamma", 1.0)),
                power=float(self.algo_config.ema.power),
            )

        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def _setup_amp(self):
        requested_amp = bool(_config_get(self.global_config.train, "use_amp", False))
        self.use_amp = requested_amp and torch.cuda.is_available()
        self.autocast_dtype = None
        self.grad_scaler = None
        if not self.use_amp:
            return
        amp_dtype = str(_config_get(self.global_config.train, "amp_dtype", "bfloat16")).lower()
        if amp_dtype in {"bfloat16", "bf16"}:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "bfloat16 autocast was requested, but this GPU does not support bfloat16. "
                    "Set ROBOT_DIFT_DROID_AMP_DTYPE=float16 explicitly (GradScaler is used)."
                )
            self.autocast_dtype = torch.bfloat16
        elif amp_dtype in {"float16", "fp16"}:
            self.autocast_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported train.amp_dtype={amp_dtype}")
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.autocast_dtype == torch.float16)

    def _autocast(self):
        if not (self.use_amp and self.autocast_dtype is not None and torch.cuda.is_available()):
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _create_optimizers(self):
        super()._create_optimizers()
        if any(TorchUtils.student_lr_schedule(self.optim_params["policy"])):
            names = {group.get("name") for group in self.optimizers["policy"].param_groups}
            if not names & TorchUtils.STUDENT_GROUP_NAMES:
                raise RuntimeError(
                    "student_freeze_steps/student_lr_warmup_steps need a trainable Student "
                    f"in its own 'student' optimizer group; groups are {sorted(map(str, names))}"
                )

    def _student_lr_scale(self, epoch):
        """Student LR factor for this optimizer step, or None without a Student schedule."""
        optim_params = _config_get(_config_get(self.algo_config, "optim_params", {}), "policy", {})
        freeze_steps, warmup_steps = TorchUtils.student_lr_schedule(optim_params)
        if not freeze_steps and not warmup_steps:
            return None
        return TorchUtils.student_lr_scale(max(0, int(epoch) - 1), freeze_steps, warmup_steps)

    @staticmethod
    def _trainable_parameters(module):
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    def ema_parameters(self):
        """Parameters tracked by EMA, in the order of its shadow copies."""
        parameters = self._trainable_parameters(self.nets)
        if self.ema is not None and len(parameters) != len(self.ema.shadow_params):
            raise RuntimeError(
                "Trainable parameters changed after EMA construction "
                f"({len(parameters)} now, {len(self.ema.shadow_params)} tracked)"
            )
        return parameters

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def _decode_language_entry(self, entry):
        if entry is None:
            return ""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, (bytes, bytearray)):
            try:
                return entry.decode("utf-8")
            except Exception:
                return entry.decode("latin1", errors="ignore")
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu().tolist()
        elif hasattr(entry, "tolist") and not isinstance(entry, (list, tuple, dict)):
            entry = entry.tolist()
        if isinstance(entry, (list, tuple)):
            parts = [self._decode_language_entry(e) for e in entry]
            return " ".join([p for p in parts if p])
        return str(entry)

    def _clean_prompt(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\n", " ")
        cleaned = re.sub(r"!+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        tokens = cleaned.split()
        clip_word_limit = 70
        if len(tokens) > clip_word_limit:
            tokens = tokens[:clip_word_limit]
        cleaned = " ".join(tokens)
        if len(cleaned) > 300:
            cleaned = cleaned[:300]
        return cleaned

    def _decode_language_prompts(self, raw_entries):
        if raw_entries is None:
            return None
        # Keep one prompt per batch item. Dropping an empty entry shifts every
        # later caption onto the wrong image. Fully unlabeled batches remain
        # unconditioned.
        prompts = [self._clean_prompt(self._decode_language_entry(entry)) for entry in raw_entries]
        return prompts if any(prompts) else None

    def _resolve_lang_prompts(self, lang_prompts, obs_source=None):
        if lang_prompts is not None:
            return list(lang_prompts)
        if isinstance(obs_source, dict):
            if "lang_prompts" in obs_source:
                value = obs_source["lang_prompts"]
                if isinstance(value, (list, tuple)):
                    return list(value)
            if "raw_language" in obs_source:
                decoded = self._decode_language_prompts(obs_source["raw_language"])
                if decoded:
                    return decoded
        return None

    # ------------------------------------------------------------------
    # Batches
    # ------------------------------------------------------------------

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {}
        for k, v in batch["obs"].items():
            if "raw" in k:
                continue
            input_batch["obs"][k] = v[:, :To, ...]

        lang_prompts = self._decode_language_prompts(batch["obs"].get("raw_language"))
        if lang_prompts is not None and len(lang_prompts) != batch["actions"].shape[0]:
            raise ValueError("Expected one language instruction per batch item")

        lang_dropout_prob = float(_config_get(self.algo_config, "language_dropout_prob", 0.0) or 0.0)
        if lang_prompts is not None and lang_dropout_prob > 0:
            drop_mask = torch.rand(len(lang_prompts)) < lang_dropout_prob
            lang_prompts = ["" if drop else prompt for prompt, drop in zip(lang_prompts, drop_mask.tolist())]

        if lang_prompts:
            input_batch["lang_prompts"] = lang_prompts

        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True

        for key in input_batch["obs"]:
            input_batch["obs"][key] = torch.nan_to_num(input_batch["obs"][key])
        input_batch["actions"] = torch.nan_to_num(input_batch["actions"])

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    # ------------------------------------------------------------------
    # Observation conditioning and losses
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(module):
        return module.module if hasattr(module, "module") else module

    def _encode_obs_sequence(self, obs_encoder, obs_sequence, lang_prompts=None):
        """Legacy per-frame robomimic encoder: ``[B, T, obs_dim]``."""
        To = self.algo_config.horizon.observation_horizon
        lang_cond = list(lang_prompts) if lang_prompts is not None else None
        features = []
        for t in range(To):
            obs_t = TensorUtils.index_at_time(obs_sequence, t)
            if lang_cond is not None:
                feats_t = obs_encoder(obs=obs_t, lang_cond=lang_cond)
            else:
                feats_t = obs_encoder(obs=obs_t)
            features.append(feats_t)
        return torch.stack(features, dim=1)

    def _legacy_student(self, obs_encoder):
        for module in obs_encoder.modules():
            if callable(getattr(module, "compute_alignment_loss", None)) and hasattr(module, "model"):
                return module
        return None

    def _legacy_alignment_loss(self, obs_encoder, obs, lang_prompts):
        """Current-frame Teacher alignment of the legacy per-camera encoder."""
        student = self._legacy_student(obs_encoder)
        image_keys = self._rgb_keys()
        if student is None or student.freeze_backbone or not image_keys:
            raise RuntimeError("Teacher alignment is enabled, but no trainable CleanDIFT Student was found")
        images = torch.cat([obs[key][:, -1] for key in image_keys], dim=0)
        batch = obs[image_keys[0]].shape[0]
        prompts = list(lang_prompts) if lang_prompts is not None else [""] * batch
        # One Teacher timestep per view: views are concatenated along the batch,
        # which equals averaging per-view losses of equal batch size.
        return student.compute_alignment_loss(images, prompts * len(image_keys))

    def _observation_condition(self, obs, lang_prompts, compute_alignment):
        """Return ``(global_cond, alignment_loss, alignment_metrics)``."""
        obs_encoder = self._unwrap(self._get_nets()["policy"]["obs_encoder"])
        metrics = {}
        alignment_loss = None
        if self.uses_paper_readout:
            features, alignment_loss, metrics = obs_encoder.encode_sequence(
                obs,
                lang_prompts,
                alignment=compute_alignment,
                return_raw_cosine=compute_alignment,
            )
            goal = obs_encoder.encode_language_goal(lang_prompts, batch=features.shape[0])
            global_cond = torch.cat([features.flatten(start_dim=1), goal.to(dtype=features.dtype)], dim=-1)
        else:
            features = self._encode_obs_sequence(obs_encoder, obs, lang_prompts=lang_prompts)
            global_cond = features.flatten(start_dim=1)
            if compute_alignment:
                alignment_loss = self._legacy_alignment_loss(obs_encoder, obs, lang_prompts)
        if alignment_loss is None:
            alignment_loss = torch.zeros((), device=global_cond.device)
        return global_cond, alignment_loss, metrics

    def _policy_loss(self, actions, global_cond):
        nets = self._get_nets()
        batch = actions.shape[0]
        num_noise_samples = int(self.algo_config.noise_samples)
        noise = torch.randn([num_noise_samples] + list(actions.shape), device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch,), device=self.device
        ).long()
        noisy_actions = torch.cat([
            self.noise_scheduler.add_noise(actions, noise[i], timesteps)
            for i in range(num_noise_samples)
        ], dim=0)
        noise_pred = nets["policy"]["noise_pred_net"](
            noisy_actions,
            timesteps.repeat(num_noise_samples),
            global_cond=global_cond.repeat(num_noise_samples, 1),
        )
        return F.mse_loss(noise_pred, noise.reshape(num_noise_samples * batch, *actions.shape[1:]))

    def _alignment_weight(self, epoch):
        base_weight = float(_config_get(self.algo_config, "cleandift_alignment_weight", 0.0) or 0.0)
        if base_weight <= 0.0:
            return 0.0
        if self._alignment_start_epoch is None:
            self._alignment_start_epoch = int(epoch)
        warmdown_steps = _config_get(self.algo_config, "cleandift_alignment_warmdown_steps", None)
        if warmdown_steps is None:
            total_epochs = int(_config_get(self.global_config.train, "num_epochs", 1000) or 1000)
            warmdown_frac = float(_config_get(self.algo_config, "cleandift_alignment_warmdown_frac", 0.5) or 0.0)
            warmdown_steps = max(1, int(total_epochs * warmdown_frac))
        return alignment_weight_at_step(
            base_weight,
            epoch,
            self._alignment_start_epoch,
            str(_config_get(self.algo_config, "cleandift_alignment_schedule_origin", "absolute")),
            warmdown_steps,
            float(_config_get(self.algo_config, "cleandift_alignment_min_decay_factor", 0.01)),
            float(_config_get(self.algo_config, "cleandift_alignment_decay_power", 1.0)),
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_on_batch(self, batch, epoch, validate=False, accumulation_step=0, accumulation_steps=1):
        """
        Training on a single microbatch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): optimizer-step index (one-based)
            validate (bool): if True, don't perform any learning updates.
            accumulation_step / accumulation_steps: microbatch position within
                one optimizer update.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if accumulation_steps < 1 or not 0 <= accumulation_step < accumulation_steps:
            raise ValueError("Invalid gradient accumulation step")
        finish_accumulation = accumulation_step == accumulation_steps - 1
        audit_steps = _config_get(self.algo_config, "representation_audit_steps", [])
        audit_active = not validate and int(epoch) in audit_steps
        if audit_steps and not hasattr(self, "_representation_audit") and not validate:
            self._representation_audit = RepresentationAudit(
                select_student_parameters(self._get_nets()["policy"]),
                path=os.environ.get("ROBOT_DIFT_REPRESENTATION_AUDIT_PATH"),
            )

        actions = batch["actions"]
        lang_prompts = self._resolve_lang_prompts(batch.get("lang_prompts", None), batch)
        for key in self.obs_shapes:
            if "raw" in key:
                continue
            # first two dimensions should be [B, T] for inputs
            assert batch["obs"][key].ndim - 2 == len(self.obs_shapes[key])

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            alignment_weight = self._alignment_weight(epoch)
            compute_alignment = alignment_weight > 0.0 and not validate

            if not validate and accumulation_step == 0:
                self.optimizers["policy"].zero_grad(set_to_none=True)

            with self._autocast() if not validate else nullcontext():
                global_cond, alignment_loss, alignment_metrics = self._observation_condition(
                    batch["obs"], lang_prompts, compute_alignment
                )
                policy_loss = self._policy_loss(actions, global_cond)
            total_loss = policy_loss + alignment_weight * alignment_loss

            if audit_active and accumulation_step == 0:
                self._representation_audit.gradients(
                    policy_loss, alignment_weight * alignment_loss,
                    step=epoch, microbatch_size=actions.shape[0],
                    batch={"obs": batch["obs"], "actions": actions, "lang_prompts": lang_prompts},
                )

            info["losses"] = TensorUtils.detach(OrderedDict(
                l2_loss=policy_loss,
                alignment_loss=alignment_loss,
                total_loss=total_loss,
            ))
            info["alignment_weight"] = alignment_weight
            info["alignment_metrics"] = TensorUtils.detach(alignment_metrics)

            if not validate:
                self._backward_and_step(
                    total_loss, info,
                    epoch=epoch,
                    accumulation_steps=accumulation_steps,
                    finish_accumulation=finish_accumulation,
                    audit_active=audit_active,
                )
        return info

    def _backward_and_step(self, total_loss, info, *, epoch, accumulation_steps, finish_accumulation, audit_active):
        optimizer = self.optimizers["policy"]
        scaler = self.grad_scaler if self.grad_scaler is not None and self.grad_scaler.is_enabled() else None
        backward_loss = total_loss / accumulation_steps
        if backward_loss.requires_grad:
            (scaler.scale(backward_loss) if scaler is not None else backward_loss).backward()
        if not finish_accumulation:
            return

        # Gradients are averaged across ranks once per optimizer step, so every
        # decision below (clipping, skipping a non-finite step) is rank-consistent.
        synchronize_optimizer_gradients(optimizer)
        if scaler is not None:
            scaler.unscale_(optimizer)
        group_norms = _grad_norms_by_group(optimizer)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        max_norm = _config_get(self.global_config.train, "max_grad_norm", None)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float("inf") if max_norm is None else float(max_norm)
        )
        norms = _scalar_floats(OrderedDict([("total", grad_norm), *group_norms.items()]))
        audit_before = self._representation_audit.before_update() if audit_active else None
        student_scale = self._student_lr_scale(epoch)

        with _scaled_student_lr(optimizer, student_scale):
            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            elif math.isfinite(norms["total"]):
                optimizer.step()
                optimizer_stepped = True
            else:
                # bf16/fp32 have no GradScaler: never apply an inf/NaN update.
                optimizer.zero_grad(set_to_none=True)
                optimizer_stepped = False
                self._nonfinite_gradient_skips += 1

        if optimizer_stepped and self.ema is not None:
            self.ema.step(self.ema_parameters())
        if audit_active:
            self._representation_audit.after_update(
                audit_before, optimizer, step=epoch, optimizer_stepped=optimizer_stepped,
            )
        self._optimizer_stepped_since_scheduler |= optimizer_stepped

        info["grad_norm"] = norms.pop("total")
        info["optimizer_group_grad_norms"] = norms
        info["optimizer_step"] = float(optimizer_stepped)
        info["nonfinite_gradient_skips"] = float(self._nonfinite_gradient_skips)
        if student_scale is not None:
            info["student_lr_scale"] = student_scale

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)

        if "policy" in self.optimizers:
            for idx, group in enumerate(self.optimizers["policy"].param_groups):
                group_name = str(group.get("name", f"group_{idx}")).replace("/", "_")
                log[f"Optimizer_Group_LR/{group_name}"] = group.get("lr", 0.0)

        scalars = OrderedDict(
            Loss=info["losses"]["l2_loss"],
            Total_Loss=info["losses"]["total_loss"],
            Alignment_Loss=info["losses"]["alignment_loss"],
        )
        for name, value in info.get("alignment_metrics", {}).items():
            scalars[f"Alignment/{name}"] = value
        log.update(_scalar_floats(scalars))

        weight = float(info.get("alignment_weight", 0.0))
        log["Alignment_Weight"] = weight
        if log["Total_Loss"] != 0:
            log["Alignment_Contribution_Fraction"] = abs(log["Alignment_Loss"] * weight) / abs(log["Total_Loss"])

        if "grad_norm" in info:
            log["Grad_Norm"] = info["grad_norm"]
            log["Optimizer_Step"] = info["optimizer_step"]
            log["Nonfinite_Gradient_Skips"] = info["nonfinite_gradient_skips"]
            for group_name, grad_norm in info["optimizer_group_grad_norms"].items():
                log[f"Optimizer_Group_Grad_Norms/{str(group_name).replace('/', '_')}"] = grad_norm
        if "student_lr_scale" in info:
            log["Student_LR_Scale"] = info["student_lr_scale"]

        return log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        self.obs_queue = deque(maxlen=To)
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_mode=None, eval_mode=False):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_mode: (optional) goal-image conditioning for real-robot evaluation

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        To = self.algo_config.horizon.observation_horizon
        lang_prompts = None

        if eval_mode:
            import cv2
            from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
            root_path = os.path.join(os.getcwd(), "eval_params")

            if goal_mode is not None:
                def read_goal(camera_id, side):
                    image = cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{camera_id}_{side}.png")), cv2.COLOR_BGR2RGB)
                    return torch.FloatTensor(image / 255.0).cuda().permute(2, 0, 1)[None, None]

                for key, camera_id in (
                    ("hand_camera", hand_camera_id),
                    ("varied_camera_1", varied_camera_1_id),
                    ("varied_camera_2", varied_camera_2_id),
                ):
                    for side in ("left", "right"):
                        obs_key = f"camera/image/{key}_{side}_image"
                        goal = read_goal(camera_id, side).repeat(1, To, 1, 1, 1)
                        obs_dict[obs_key] = torch.cat([obs_dict[obs_key], goal], dim=2)
            else:
                # Current language instruction for language-conditioned policies.
                with open(os.path.join(root_path, "lang_command.txt"), "r") as file:
                    lang_prompts = [file.read()]

        lang_prompts = self._resolve_lang_prompts(lang_prompts, obs_dict)

        if len(self.action_queue) == 0:
            # no actions left, run inference: [1, Ta, Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, lang_prompts=lang_prompts)
            self.action_queue.extend(action_sequence[0])

        # has action, execute from left to right: [1, Da]
        return self.action_queue.popleft().unsqueeze(0)

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict, lang_prompts=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        for key in self.obs_shapes:
            if "raw" in key:
                continue
            # first two dimensions should be [B, T] for inputs
            assert obs_dict[key].ndim - 2 == len(self.obs_shapes[key])
        lang_prompts = self._resolve_lang_prompts(lang_prompts, obs_dict)

        ema_parameters = self.ema_parameters() if self.ema is not None else None
        if self.ema is not None:
            # Temporarily evaluate the EMA weights; always restore them, even
            # when inference fails, so ranks never train from different weights.
            self.ema.store(ema_parameters)
            self.ema.copy_to(ema_parameters)
        try:
            nets = self._get_nets()
            with self._autocast():
                obs_cond, _, _ = self._observation_condition(obs_dict, lang_prompts, compute_alignment=False)
                obs_cond = obs_cond.float()
                naction = torch.randn((obs_cond.shape[0], Tp, self.ac_dim), device=self.device)
                self.noise_scheduler.set_timesteps(num_inference_timesteps)
                noise_pred_net = self._unwrap(nets["policy"]["noise_pred_net"])
                for k in self.noise_scheduler.timesteps:
                    noise_pred = noise_pred_net(sample=naction, timestep=k, global_cond=obs_cond)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred.float(),
                        timestep=k,
                        sample=naction
                    ).prev_sample
        finally:
            if self.ema is not None:
                self.ema.restore(ema_parameters)

        # process action using Ta
        start = To - 1
        end = start + Ta
        return naction[:, start:end]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "alignment_start_epoch": getattr(self, "_alignment_start_epoch", None),
            "nonfinite_gradient_skips": int(self._nonfinite_gradient_skips),
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        nets_state = model_dict["nets"]
        # Checkpoints saved from a DDP-wrapped model carry a 'module.' prefix.
        if any(k.startswith("module.") for k in nets_state.keys()):
            nets_state = OrderedDict(
                (k[len("module."):] if k.startswith("module.") else k, v) for k, v in nets_state.items()
            )
        self.nets.load_state_dict(nets_state)
        if model_dict.get("ema", None) is not None:
            if self.ema is None:
                raise RuntimeError("Checkpoint contains EMA weights, but EMA is disabled")
            self.ema.load_state_dict(model_dict["ema"])
            all_parameters = list(self.nets.parameters())
            parameters = self._trainable_parameters(self.nets)
            shadows = list(self.ema.shadow_params)
            if len(shadows) == len(all_parameters) != len(parameters):
                # EMA state saved over every parameter lists the shadows in
                # module order; keep those of the trainable tensors.
                shadows = [shadow for shadow, parameter in zip(shadows, all_parameters) if parameter.requires_grad]
            if len(shadows) != len(parameters) or any(
                shadow.shape != parameter.shape for shadow, parameter in zip(shadows, parameters)
            ):
                raise RuntimeError(
                    "Checkpoint EMA tracks a different parameter set "
                    f"({len(shadows)} shadows for {len(parameters)} trainable tensors); "
                    "it was written by an incompatible training configuration"
                )
            # Keep EMA shadows next to their parameters (checkpoints load on CPU).
            self.ema.shadow_params = [
                shadow.to(device=parameter.device, dtype=parameter.dtype)
                for shadow, parameter in zip(shadows, parameters)
            ]

        self._alignment_start_epoch = model_dict.get("alignment_start_epoch", None)
        self._nonfinite_gradient_skips = int(model_dict.get("nonfinite_gradient_skips", 0) or 0)


# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
