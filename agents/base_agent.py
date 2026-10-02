import abc
import logging
import os
import pickle
from collections import deque
from typing import Optional, List, Dict

import einops
import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
import torch.distributed as dist

from agents.utils.scaler import ActionScaler, MinMaxScaler, Scaler
from agents.encoders.improved_proprio_fusion import ImprovedProprioEncoder, FiLMFusion

# A logger for this file
log = logging.getLogger(__name__)


def _is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    try:
        return dist.get_rank() == 0
    except RuntimeError:
        return True

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """summary

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        # cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        # cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        if in_channels != 3:
            raise ValueError(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_channels = out_channels
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)


    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

class BaseAgent(nn.Module, abc.ABC):

    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
        obs_seq_len: int,  # Spatial tokens (for decoder)
        act_seq_len: int,
        cam_names: list[str],
        temporal_obs_len: int = 1,  # Temporal history length (for inference)
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        if_dift_language: bool = False,
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None
        self._scaler_loaded_from_checkpoint = False
        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition
        self.if_dift_language = if_dift_language

        # Initialize model and encoder
        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.point_encoder = PointNetEncoderXYZ(in_channels=3,
                                                out_channels=384,
                                                use_layernorm=True,
                                                final_norm='layernorm',
                                                use_projection=True).to(device)
        self.language_description = None
        # Candidate Robot-DIFT readouts already hold a frozen CLIP text tower.
        # Reuse it for the pooled policy goal instead of loading another full
        # CLIP copy. Existing encoders retain their configured language module.
        self.language_encoder = (
            None if hasattr(self.img_encoder, "encode_language_goal")
            else hydra.utils.instantiate(language_encoders).to(device)
        )
        self.model = hydra.utils.instantiate(model).to(device)

        if self.if_robot_states:
            self.proprio_encoder = ImprovedProprioEncoder(
                state_dim=state_dim,
                latent_dim=latent_dim,
            ).to(device)
            self.proprio_film = FiLMFusion(visual_dim=latent_dim, proprio_dim=latent_dim).to(device)
        else:
            self.proprio_encoder = None
            self.proprio_film = None
        self.register_buffer("robot_states_min", None)
        self.register_buffer("robot_states_max", None)
        self._robot_state_stats_loaded_from_checkpoint = False

        self.cam_names = cam_names
        self.num_cameras = len(cam_names) if cam_names is not None else 0
        if self.num_cameras > 0:
            self.camera_embed = nn.Parameter(torch.zeros(self.num_cameras, latent_dim))
            nn.init.normal_(self.camera_embed, mean=0.0, std=0.02)
        else:
            self.camera_embed = None
        self._disable_camera_embed = os.environ.get("ROBOT_DIFT_DISABLE_CAMERA_EMBED", "").lower() in {"1", "true", "yes"}
        if self._disable_camera_embed and self.camera_embed is not None and _is_rank0():
            log.info("ROBOT_DIFT_DISABLE_CAMERA_EMBED is set; camera embeddings will be loaded but not applied")
        self.alignment_cfg = getattr(self.img_encoder, "alignment_cfg", None)
        self._last_alignment_loss: Optional[torch.Tensor] = None
        # for inference
        self.rollout_step_counter = 0
        self.act_seq_len = act_seq_len
        self.obs_seq_len = obs_seq_len  # Spatial tokens for decoder
        self.temporal_obs_len = temporal_obs_len  # Temporal history for inference
        self._rollout_replan_every = self.act_seq_len
        replan_every = os.environ.get("ROBOT_DIFT_ROLLOUT_REPLAN_EVERY", "").strip()
        if replan_every:
            try:
                self._rollout_replan_every = max(1, min(self.act_seq_len, int(replan_every)))
            except ValueError:
                self._rollout_replan_every = self.act_seq_len
            if self._rollout_replan_every != self.act_seq_len and _is_rank0():
                log.info(
                    "ROBOT_DIFT_ROLLOUT_REPLAN_EVERY=%d; policy will replan before exhausting the %d-step action chunk",
                    self._rollout_replan_every,
                    self.act_seq_len,
                )

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}
        self._warned_nonfinite_rollout_actions = False
        self._warned_clipped_rollout_actions = False
        self._clip_rollout_actions = os.environ.get("ROBOT_DIFT_CLIP_ROLLOUT_ACTIONS", "").lower() in {"1", "true", "yes"}

    def train(self, mode: bool = True):
        return super().train(mode)

    def set_scaler(self, scaler):
        self.scaler = scaler
        self._move_scaler_to_device()
        self._scaler_loaded_from_checkpoint = False

    def _move_scaler_to_device(self) -> None:
        if self.scaler is None:
            return
        if hasattr(self.scaler, "device"):
            self.scaler.device = self.device
        for name, value in vars(self.scaler).items():
            if torch.is_tensor(value):
                setattr(self.scaler, name, value.to(self.device))

    def set_robot_state_stats(self, stats: dict | None) -> None:
        if stats is None or not stats:
            self.robot_states_min = None
            self.robot_states_max = None
            self._robot_state_stats_loaded_from_checkpoint = False
            return

        min_tensor = torch.as_tensor(stats.get("min"), dtype=torch.float32) if stats.get("min") is not None else None
        max_tensor = torch.as_tensor(stats.get("max"), dtype=torch.float32) if stats.get("max") is not None else None

        if min_tensor is not None:
            min_tensor = min_tensor.to(self.device)
        if max_tensor is not None:
            max_tensor = max_tensor.to(self.device)

        self.robot_states_min = min_tensor
        self.robot_states_max = max_tensor
        self._robot_state_stats_loaded_from_checkpoint = False

    def _normalize_robot_states(self, robot_states: torch.Tensor) -> torch.Tensor:
        if self.robot_states_min is None or self.robot_states_max is None:
            return robot_states
        denom = (self.robot_states_max - self.robot_states_min).clamp(min=1e-6)
        return 2.0 * (robot_states - self.robot_states_min) / denom - 1.0

    def _process_vision_output(self, vision: torch.Tensor) -> torch.Tensor:
        """
        Normalize vision encoder output to shape [B, tokens, dim].
        Supports legacy [B, num_cam, dim] and new [B, num_cam, num_tokens, dim].
        Adds camera embedding per camera when available.
        """
        if vision.dim() == 2:
            vision = vision.unsqueeze(1)  # [B,1,C]
        if vision.dim() == 3:
            vision = vision.unsqueeze(2)  # [B, num_cam, 1, dim]
        if vision.dim() != 4:
            raise ValueError(f"Unexpected vision output shape: {vision.shape}")

        B, num_cam, num_tokens, dim = vision.shape
        if (
            self.camera_embed is not None
            and not self._disable_camera_embed
            and num_cam == self.num_cameras
            and self.camera_embed.shape[-1] == dim
        ):
            camera_bias = self.camera_embed.view(1, self.num_cameras, 1, dim)
            vision = vision + camera_bias

        vision = vision.reshape(B, num_cam * num_tokens, dim)
        return vision

    # @abc.abstractmethod
    def compute_input_embeddings(self, obs_dict, alignment_context: Optional[dict] = None):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        self._last_alignment_loss = None

        #########################################
        # deal with language embedding
        #########################################
        if "lang" in obs_dict:
            self.language_description = obs_dict["lang"]
            if "lang_emb" not in obs_dict:
                if self.language_encoder is None:
                    goal = self.img_encoder.encode_language_goal(obs_dict["lang"])
                else:
                    goal = self.language_encoder(obs_dict["lang"])
                obs_dict["lang_emb"] = goal.float()

        latent_goal = obs_dict["lang_emb"]
        # latent_goal = obs_dict.get("lang_emb", None)
        # latent_goal = torch.zeros(obs_dict["agentview_image"].shape[0], 1, 512).to(self.device)           # tmp: TODO: remove this if have lang_emb

        if "point_cloud" in obs_dict and "robot0_agentview_left_image" in obs_dict:

            assert obs_dict["point_cloud"].shape[-1] in [3, 6], "Point cloud should have 3 or 6 channels"

            obs_dict["point_cloud"] = einops.rearrange(obs_dict["point_cloud"], "b t n d -> (b t) n d")

            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
            # B, T, C, H, W = obs_dict["robot0_agentview_center_image"].shape
            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            if self.if_film_condition:
                perceptual_emb, _ = self.img_encoder(obs_dict, latent_goal)
            else:
                perceptual_emb, _ = self.img_encoder(obs_dict)

            perceptual_emb = self._process_vision_output(perceptual_emb)
            self._last_alignment_loss = None
            return perceptual_emb, latent_goal

        if "point_cloud" in obs_dict and "agentview_rgb_image" in obs_dict:
            assert obs_dict["point_cloud"].shape[-1] in [3, 6], "Point cloud should have 3 or 6 channels"

            pc = einops.rearrange(obs_dict["point_cloud"], "b t n d -> (b t) n d")

            point_emb = self.point_encoder(pc)

            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
            # B, T, C, H, W = obs_dict["robot0_agentview_center_image"].shape

            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            if self.if_film_condition:
                perceptual_emb, _ = self.img_encoder(obs_dict, latent_goal)
            else:
                perceptual_emb, _ = self.img_encoder(obs_dict)

            perceptual_emb = self._process_vision_output(perceptual_emb)
            point_emb = point_emb.view(B, T, -1)
            perceptual_emb = torch.cat([perceptual_emb, point_emb], dim=1)

            self._last_alignment_loss = None
            return perceptual_emb, latent_goal

        #########################################
        elif self.cam_names is not None:
            # Handle both training (5D) and inference (4D or 5D) cases
            first_image = obs_dict[f"{self.cam_names[0]}_image"]
            image_batch = first_image
            if len(image_batch.shape) == 4:
                vision_batch = image_batch.shape[0]
                temporal_len = 1
            else:
                vision_batch = image_batch.shape[0]
                temporal_len = image_batch.shape[1]

            alignment_weight = 0.0
            alignment_images: Dict[str, torch.Tensor] = {}
            alignment_sample_indices = None
            alignment_captions = None

            if alignment_context is not None:
                alignment_weight = float(alignment_context.get("weight", 0.0) or 0.0)
                alignment_sample_indices = alignment_context.get("sample_indices")
                if "lang" in obs_dict and isinstance(obs_dict["lang"], list):
                    alignment_captions = obs_dict["lang"]
                    if alignment_weight > 0.0:
                        alignment_views = alignment_context.get("views")
                        if alignment_views is None or alignment_views == "all":
                            target_cameras = self.cam_names
                        else:
                            target_cameras = [
                                name for name in self.cam_names if name in alignment_views
                            ]
                            if not target_cameras:
                                target_cameras = self.cam_names
                        for camera in target_cameras:
                            camera_key = f"{camera}_image"
                            camera_obs = obs_dict[camera_key]
                            if camera_obs.ndim == 5:
                                alignment_images[camera_key] = camera_obs[:, -1].detach().clone()

            if len(first_image.shape) == 4:
                B, C, H, W = first_image.shape
                T = 1
                for camera in self.cam_names:
                    obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].unsqueeze(1)
            else:
                B, T, C, H, W = first_image.shape

            # Reshape for processing
            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            encoder_alignment_ctx = None
            if alignment_weight > 0.0 and alignment_images:
                encoder_alignment_ctx = {
                    "weight": alignment_weight,
                    "images": alignment_images,
                    "captions": alignment_captions,
                    "sample_indices": alignment_sample_indices,
                }

            if self.if_film_condition:
                perceptual_emb, alignment_loss = self.img_encoder(obs_dict, latent_goal, alignment_context=encoder_alignment_ctx)
            elif self.if_dift_language:
                perceptual_emb, alignment_loss = self.img_encoder(obs_dict, self.language_description, alignment_context=encoder_alignment_ctx)
            else:
                perceptual_emb, alignment_loss = self.img_encoder(obs_dict, alignment_context=encoder_alignment_ctx)

            perceptual_emb = self._process_vision_output(perceptual_emb)
            if alignment_weight > 0.0 and alignment_loss is not None:
                weight_tensor = alignment_loss.new_tensor(alignment_weight)
                weighted_alignment = alignment_loss * weight_tensor
                self._last_alignment_loss = (alignment_loss, weighted_alignment)
            else:
                self._last_alignment_loss = None

            BT, num_cameras, visual_dim = perceptual_emb.shape
            if vision_batch is None:
                vision_batch = BT
                temporal_len = 1
            if vision_batch * temporal_len != BT:
                raise RuntimeError("Mismatch between inferred batch/time dims and encoder output.")

            # Restore explicit [B, T, num_cam, dim] tokens.
            perceptual_tokens = perceptual_emb.view(vision_batch, temporal_len, num_cameras, visual_dim)

            if (
                self.if_robot_states
                and self.proprio_encoder is not None
                and self.proprio_film is not None
                and "robot_states" in obs_dict
            ):
                robot_states = obs_dict["robot_states"]
                if robot_states.shape[1] > temporal_len:
                    robot_states = robot_states[:, :temporal_len]
                proprio_feat = self.proprio_encoder(robot_states)  # [B, T, latent_dim]
                perceptual_tokens = self.proprio_film(
                    perceptual_tokens.view(vision_batch, temporal_len * num_cameras, visual_dim),
                    proprio_feat.view(vision_batch, temporal_len, visual_dim),
                )
                perceptual_tokens = perceptual_tokens.view(vision_batch, temporal_len, num_cameras, visual_dim)

            # Flatten back to per-token sequence for downstream decoder.
            total_views = perceptual_tokens.shape[2]
            perceptual_emb = perceptual_tokens.view(vision_batch, temporal_len * total_views, visual_dim)
        else:
            raise NotImplementedError("Either use point clouds or images as input.")

        return perceptual_emb, latent_goal

    @abc.abstractmethod
    def forward(self, obs_dict: dict[str, torch.Tensor], actions=None, alignment_context=None) -> torch.Tensor:
        """
        Forward pass of the model
        """
        pass

    def reset(self):
        """Resets the context of the model."""
        self.rollout_step_counter = 0
        self.obs_seq: dict[str, deque[torch.Tensor]] = {}
        self._warned_nonfinite_rollout_actions = False
        self._warned_clipped_rollout_actions = False

    def _sanitize_rollout_actions(self, pred_action_seq: torch.Tensor) -> torch.Tensor:
        """Clean and bound rollout actions before they reach the simulator."""
        pred_action_seq = pred_action_seq.to(self.device, dtype=torch.float32)

        if not torch.isfinite(pred_action_seq).all():
            if _is_rank0() and not self._warned_nonfinite_rollout_actions:
                log.warning(
                    "Detected non-finite rollout actions; replacing NaN/Inf values before simulation."
                )
                self._warned_nonfinite_rollout_actions = True
            pred_action_seq = torch.nan_to_num(
                pred_action_seq,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        if self._clip_rollout_actions and self.scaler is not None and hasattr(self.scaler, "clip_action"):
            clipped_action_seq = self.scaler.clip_action(pred_action_seq)
            if (
                _is_rank0()
                and not self._warned_clipped_rollout_actions
                and not torch.allclose(clipped_action_seq, pred_action_seq)
            ):
                max_delta = (clipped_action_seq - pred_action_seq).abs().max().item()
                log.warning(
                    "Clipped rollout actions to scaler bounds before simulation (max delta %.4f).",
                    max_delta,
                )
                self._warned_clipped_rollout_actions = True
            pred_action_seq = clipped_action_seq

        return pred_action_seq

    @torch.no_grad()
    def predict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # initialize self.obs_seq if empty
        if not self.obs_seq:
            for key in obs_dict.keys():
                self.obs_seq[key] = deque(maxlen=self.temporal_obs_len)

        for key in obs_dict.keys():
            current_value = obs_dict[key]

            if key == "robot_states":
                current_value = self._normalize_robot_states(current_value)

            self.obs_seq[key].append(current_value)

            if key == "lang":
                continue
            obs_dict[key] = torch.concat(list(self.obs_seq[key]), dim=1)

            if obs_dict[key].shape[1] < self.temporal_obs_len:
                # left pad repeated first element
                pad = einops.repeat(
                    obs_dict[key][:, 0],
                    "b ... -> b t ...",
                    t=self.temporal_obs_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)

        if self.rollout_step_counter == 0:
            self.eval()

            # predict action sequence
            pred_action_seq = self(obs_dict)[:, :self.act_seq_len]
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            pred_action_seq = self._sanitize_rollout_actions(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[:, self.rollout_step_counter]  # TODO:for isaac it returns a batch of actions

        self.rollout_step_counter += 1
        if self.rollout_step_counter >= self._rollout_replan_every:
            self.rollout_step_counter = 0

        # Ensure the action has the correct shape: [batch_size, action_dim]
        # If the output is squeezed to 1D, we need to unsqueeze it
        if current_action.dim() == 1:
            current_action = current_action.unsqueeze(0)

        # For environment step, we need to squeeze the batch dimension to get [action_dim]
        if current_action.shape[0] == 1:
            current_action = current_action.squeeze(0)

        return current_action

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load pretrained weights for the entire agent
        """
        if os.path.isfile(weights_path):
            path = weights_path
        else:
            candidates = []
            if sv_name is None:
                candidates.extend(["model_state_dict.pth", "last_model.pth"])
            else:
                candidates.append(f"{sv_name}.pth")
            path = None
            for filename in candidates:
                candidate = os.path.join(weights_path, filename)
                if os.path.isfile(candidate):
                    path = candidate
                    break
            if path is None:
                expected = ", ".join(candidates)
                raise FileNotFoundError(
                    f"Could not find any of [{expected}] under checkpoint path: {weights_path}"
                )
        state_dict = torch.load(path, weights_only=True)

        robot_stats = {}
        for key in ("robot_states_min", "robot_states_max"):
            if key in state_dict:
                robot_stats[key] = state_dict.pop(key)

        incompatible = self.load_state_dict(state_dict, strict=False)
        missing_keys = [
            key for key in getattr(incompatible, "missing_keys", [])
            if key not in robot_stats
        ]
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint incompatibility for {path}: "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

        if "robot_states_min" in robot_stats and robot_stats["robot_states_min"] is not None:
            self.robot_states_min = robot_stats["robot_states_min"].to(self.device)
        if "robot_states_max" in robot_stats and robot_stats["robot_states_max"] is not None:
            self.robot_states_max = robot_stats["robot_states_max"].to(self.device)
        self._robot_state_stats_loaded_from_checkpoint = bool(robot_stats)
        checkpoint_dir = os.path.dirname(path)
        scaler_path = os.path.join(checkpoint_dir, "model_scaler.pkl")
        if os.path.isfile(scaler_path):
            self.load_model_scaler(checkpoint_dir)
        if _is_rank0():
            log.info("Loaded pre-trained model from %s", path)

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the weights of the entire agent
        """
        path = os.path.join(
            store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth"
        )
        torch.save(self.state_dict(), path)
        if _is_rank0():
            log.info(f"Model saved to: {store_path}")

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        """
        Store the model scaler inside the store path as model_scaler.pkl
        """
        save_path = os.path.join(
            store_path, "model_scaler.pkl" if sv_name is None else sv_name
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        if _is_rank0():
            log.info(f"Model scaler saved to: {save_path}")

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        """
        Load the model scaler from the weights path
        """
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = pickle.load(f)
        self._move_scaler_to_device()
        self._scaler_loaded_from_checkpoint = True
        if _is_rank0():
            log.info("Loaded model scaler")

    def get_params(self, wandb_log=True):

        total_params = sum(p.numel() for p in self.parameters())

        if wandb_log:
            wandb.log({"model parameters": total_params})

        if _is_rank0():
            log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler

    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (self.state_dict(), self.get_scaler)

    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)
