import logging
import json
import hashlib
import os
from pathlib import Path

import cv2
import einops
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from omegaconf import ListConfig
from robocasa.utils.env_utils import create_env
from tqdm import tqdm

from agents.base_agent import BaseAgent
from environments.utils.robocasa_aliases import normalize_env_name
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


class RoboCasaSim(BaseSim):
    def __init__(
        self,
        env_name: str,
        camera_names: ListConfig[str],
        img_height: int,
        img_width: int,
        num_episode: int,
        max_step_per_episode: int,
        seed: int,
        device: str,
        render: bool = True,
        n_cores: int = 1,
        if_vision: bool = False,
        global_action: bool = False,
        start_episode: int = 0,
        layout_ids: list[int] | None = None,
        style_ids: list[int] | None = None,
        obj_instance_split: str | None = None,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.num_episode = num_episode
        self.start_episode = max(0, int(start_episode))
        self.max_step_per_episode = max_step_per_episode
        self.camera_names = list(camera_names)
        self.global_action = global_action
        self.layout_ids = list(layout_ids) if layout_ids is not None else None
        self.style_ids = list(style_ids) if style_ids is not None else None
        self.obj_instance_split = obj_instance_split

        # Normalize env names (Hydra may pass a ListConfig, list, or str).
        env_iterable = (
            env_name
            if isinstance(env_name, (list, tuple, ListConfig))
            else [env_name]
        )
        # Keep (alias, robosuite name) pairs so logging stays friendly but we
        # instantiate the actual registered environment.
        self.env_name = [
            (str(env), normalize_env_name(str(env))) for env in env_iterable
        ]

        self.img_height = img_height
        self.img_width = img_width

    @staticmethod
    def _sanitize_policy_action(action: np.ndarray, env_name: str, episode_idx: int, step_idx: int) -> np.ndarray:
        """Keep simulator-facing actions finite without changing normal eval behavior."""
        action = np.asarray(action)
        if not np.isfinite(action).all():
            log.warning(
                "Non-finite action detected in RoboCasa sim; sanitizing. env=%s episode=%d step=%d action=%s",
                env_name,
                episode_idx,
                step_idx,
                action,
            )
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        clip_actions = os.environ.get("ROBOT_DIFT_CLIP_SIM_ACTIONS", "").lower() in {"1", "true", "yes"}
        if clip_actions and action.shape[0] >= 7:
            clipped = action.copy()
            clipped[:7] = np.clip(clipped[:7], -1.1, 1.1)
            action = clipped

        return action

    @staticmethod
    def _maybe_log_action_debug(
        action: np.ndarray, env_name: str, episode_idx: int, step_idx: int,
        sim_state: np.ndarray | None = None,
    ) -> None:
        every = os.environ.get("ROBOT_DIFT_LOG_SIM_ACTION_EVERY", "").strip()
        if not every:
            return
        try:
            every_n = int(every)
        except ValueError:
            return
        if every_n <= 0:
            return
        if step_idx != 0 and ((step_idx + 1) % every_n) != 0:
            return
        state = np.asarray(sim_state) if sim_state is not None else np.asarray([])
        log.info(
            "[SimAction] env=%s episode=%d step=%d min=%.4f max=%.4f l2=%.4f "
            "state_finite=%s state_absmax=%.4f action=%s",
            env_name,
            episode_idx,
            step_idx,
            float(action.min()),
            float(action.max()),
            float(np.linalg.norm(action)),
            bool(np.isfinite(state).all()),
            float(np.max(np.abs(state))) if state.size else 0.0,
            np.array2string(action, precision=4, suppress_small=False),
        )

    def _maybe_reseed_policy_rng(self, env_name: str, episode_idx: int) -> None:
        reseed = os.environ.get("ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if not reseed:
            return

        policy_seed = int(self.seed) + int(episode_idx)
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        log.info(
            "[SimEpisodeSeed] env=%s episode=%d policy_seed=%d",
            env_name,
            episode_idx,
            policy_seed,
        )

    def test_agent(
        self,
        agent: BaseAgent,
        step: int | None = None,
        save_videos: bool = False,
        video_dir: str | None = None,
        video_fps: int = 20,
    ):

        metrics = {}
        video_root = self._prepare_video_dir(agent, save_videos, video_dir)

        for env_alias, env_real in self.env_name:

            print(f"Initializing environment {env_alias} (robosuite id: {env_real})")

            self._init_env(
                env_real,
                self.img_width,
                self.img_height,
                self.render,
            )

            env_video_dir = None
            if video_root is not None:
                env_video_dir = video_root / self._sanitize_name(env_alias)
                env_video_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0
            task_completion_hold_count = -1

            episode_logs = []

            if self.start_episode > 0:
                log.info(
                    "[SimEpisodeSkip] env=%s seed=%s skipping %d resets before evaluation",
                    env_alias,
                    self.seed,
                    self.start_episode,
                )
                for _ in range(self.start_episode):
                    self.env.reset()

            for local_i in range(self.num_episode):
                i = self.start_episode + local_i
                obs = self.env.reset()
                self._maybe_reseed_policy_rng(env_alias, i)
                episode_meta = self.env.get_ep_meta()
                layout_id = episode_meta.get("layout_id")
                style_id = episode_meta.get("style_id")
                reset_state_sha256 = None
                if os.environ.get("ROBOT_DIFT_EPISODE_LOG_PATH"):
                    reset_state = np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64)
                    reset_state_sha256 = hashlib.sha256(reset_state.tobytes()).hexdigest()
                if self.layout_ids is not None and layout_id not in self.layout_ids:
                    raise RuntimeError(f"RoboCasa reset produced layout_id={layout_id} outside {self.layout_ids}")
                if self.style_ids is not None and style_id not in self.style_ids:
                    raise RuntimeError(f"RoboCasa reset produced style_id={style_id} outside {self.style_ids}")
                log.info(
                    "[SimScene] env=%s episode=%d layout_id=%s style_id=%s obj_instance_split=%s",
                    env_alias, i, layout_id, style_id, self.obj_instance_split,
                )

                if self.render:
                    self.env.render()

                agent.reset()
                # Reset stored prev_action for simulation alignment
                agent._last_action_sim = torch.zeros(1, 1, 7, device=self.device)

                lang = episode_meta["lang"]

                print(f"episode {i}: ", lang)

                episode_success = False
                steps_taken = 0

                episode_video_path = None
                if env_video_dir is not None:
                    episode_video_path = env_video_dir / f"episode_{i:03d}.mp4"
                video_writer = None
                frames_written = 0

                for j in tqdm(
                    range(self.max_step_per_episode),
                    leave=False,
                    dynamic_ncols=True,
                    desc=f"{env_alias} episode {i}",
                ):
                    obs_dict = {}
                    obs_dict["lang"] = lang

                    gripper_qpos = obs["robot0_gripper_qpos"]
                    if gripper_qpos.ndim and gripper_qpos.shape[-1] > 1:
                        gripper_qpos = gripper_qpos[..., :1]
                    gripper_state = torch.from_numpy(gripper_qpos).float()
                    gripper_state = einops.rearrange(
                        gripper_state, "d -> 1 1 d"
                    ).to(self.device)

                    gripper_qvel = obs.get("robot0_gripper_qvel")
                    if gripper_qvel is None:
                        gripper_qvel = np.zeros_like(gripper_qpos)
                    elif gripper_qvel.ndim and gripper_qvel.shape[-1] > 1:
                        gripper_qvel = gripper_qvel[..., :1]
                    gripper_vel = torch.from_numpy(gripper_qvel).float()
                    gripper_vel = einops.rearrange(
                        gripper_vel, "d -> 1 1 d"
                    ).to(self.device)

                    joint_pos_raw = obs.get("robot0_joint_pos")
                    if joint_pos_raw is None:
                        # RoboCasa环境只提供sin/cos值,需要用atan2恢复原始角度
                        joint_pos_cos = obs.get("robot0_joint_pos_cos")
                        joint_pos_sin = obs.get("robot0_joint_pos_sin")
                        if joint_pos_cos is None or joint_pos_sin is None:
                            raise ValueError("Cannot find robot0_joint_pos, robot0_joint_pos_cos, or robot0_joint_pos_sin!")

                        # 使用atan2恢复原始角度值 (范围 [-π, π])
                        joint_pos_raw = np.arctan2(joint_pos_sin, joint_pos_cos)

                    # Optional debug prints were removed to keep simulation output clean

                    # Take only the arm joints (first 7), matching training data
                    # PandaMobile may have extra joints for the mobile base
                    joint_pos_np = joint_pos_raw[..., :7]
                    joint_pos = torch.from_numpy(joint_pos_np).float()
                    joint_pos = einops.rearrange(joint_pos, "d -> 1 1 d").to(self.device)

                    joint_vel_np = obs.get("robot0_joint_vel")
                    if joint_vel_np is None:
                        joint_vel_np = np.zeros_like(joint_pos_np)
                    joint_vel = torch.from_numpy(joint_vel_np[..., :7]).float()
                    joint_vel = einops.rearrange(joint_vel, "d -> 1 1 d").to(self.device)

                    eef_pos_np = obs.get("robot0_eef_pos")
                    if eef_pos_np is None:
                        raise ValueError("robot0_eef_pos missing in simulation obs")
                    eef_pos = torch.from_numpy(eef_pos_np).float()
                    eef_pos = einops.rearrange(eef_pos, "d -> 1 1 d").to(self.device)

                    eef_quat_np = obs.get("robot0_eef_quat")
                    if eef_quat_np is None:
                        raise ValueError("robot0_eef_quat missing in simulation obs")
                    eef_quat = torch.from_numpy(eef_quat_np).float()
                    eef_quat = einops.rearrange(eef_quat, "d -> 1 1 d").to(self.device)

                    prev_action = getattr(agent, "_last_action_sim", None)
                    if prev_action is None:
                        prev_action = torch.zeros(1, 1, 7, device=self.device)
                        agent._last_action_sim = prev_action

                    robot_state = torch.cat(
                        [
                            gripper_state,
                            gripper_vel,
                            joint_pos,
                            joint_vel,
                            eef_pos,
                            eef_quat,
                            prev_action,
                        ],
                        dim=-1,
                    )

                    obs_dict["robot_states"] = robot_state

                    for cam_name in self.camera_names:

                        # cv2.imshow(f"{cam_name}_image", obs[f"{cam_name}_image"])
                        # cv2.waitKey(1)

                        rgb = (
                            torch.from_numpy(obs[f"{cam_name}_image"].copy())
                            .float()
                            .permute(2, 0, 1)
                            / 255.0
                        )
                        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(self.device)
                        obs_dict[f"{cam_name}_image"] = rgb

                    action = agent.predict(obs_dict).cpu().numpy()

                    # Update prev_action buffer (use only first 7 dims before padding)
                    prev_action_tensor = (
                        torch.from_numpy(action.reshape(1, -1)[:, :7])
                        .float()
                        .view(1, 1, 7)
                        .to(self.device)
                    )
                    agent._last_action_sim = prev_action_tensor

                    # Ensure action is 1D: if shape is [1, 7], squeeze to [7]
                    if action.ndim > 1:
                        action = action.squeeze(0)

                    # Model outputs 7-dim arm actions: [dx, dy, dz, dax, day, daz, gripper]
                    # Need to append 5-dim mobile base actions (constant in dataset: [0, 0, 0, 0, -1])
                    # Environment expects [dx, dy, dz, dax, day, daz, gripper, base_x, base_y, base_theta, base_gripper, base_lift]
                    action = np.concatenate(
                        [action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])]
                    )

                    if self.global_action:
                        action = self._get_local_action(action)

                    action = self._sanitize_policy_action(action, env_alias, i, j)
                    self._maybe_log_action_debug(
                        action, env_alias, i, j,
                        self.env.sim.get_state().flatten()
                        if os.environ.get("ROBOT_DIFT_LOG_SIM_ACTION_EVERY") else None,
                    )

                    obs, _, done, _ = self.env.step(action)

                    if self.render:
                        self.env.render()

                    if episode_video_path is not None:
                        video_writer, wrote = self._record_video_frame(
                            video_writer,
                            episode_video_path,
                            video_fps,
                            obs,
                        )
                        if wrote:
                            frames_written += 1

                    if self.env._check_success():
                        if task_completion_hold_count > 0:
                            task_completion_hold_count -= (
                                1  # latched state, decrement count
                            )
                        else:
                            task_completion_hold_count = (
                                10  # reset count on first success timestep
                            )
                    else:
                        task_completion_hold_count = (
                            -1
                        )  # null the counter if there's no success

                    if task_completion_hold_count == 0:
                        success_count += 1
                        episode_success = True
                        done = True

                    steps_taken = j + 1

                    if done:
                        break

                if video_writer is not None:
                    video_writer.release()
                    if frames_written > 0 and episode_video_path is not None:
                        final_video_path = self._finalize_episode_video(
                            episode_video_path, episode_success
                        )
                        log.info(
                            "Saved simulation video: %s",
                            final_video_path,
                        )
                    elif episode_video_path is not None and episode_video_path.exists():
                        episode_video_path.unlink(missing_ok=True)

                episode_logs.append({
                    "episode": i,
                    "steps": steps_taken,
                    "success": episode_success,
                })
                episode_log_path = os.environ.get("ROBOT_DIFT_EPISODE_LOG_PATH")
                if episode_log_path:
                    record = {
                        "task": env_alias,
                        "episode": int(i),
                        "training_epoch": None if step is None else int(step),
                        "seed": int(self.seed),
                        "layout_id": None if layout_id is None else int(layout_id),
                        "style_id": None if style_id is None else int(style_id),
                        "obj_instance_split": self.obj_instance_split,
                        "reset_state_sha256": reset_state_sha256,
                        "policy_seed": (
                            int(self.seed) + int(i)
                            if os.environ.get("ROBOT_DIFT_RESEED_POLICY_EACH_EPISODE", "").lower()
                            in {"1", "true", "yes"}
                            else None
                        ),
                        "steps": int(steps_taken),
                        "success": bool(episode_success),
                    }
                    path = Path(episode_log_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                log.info(
                    "[SimEpisode] env=%s episode=%d steps=%d success=%s",
                    env_alias,
                    i,
                    steps_taken,
                    episode_success,
                )

            success_rate = success_count / self.num_episode
            print(f"Success rate: {success_rate}")

            metrics[f"{env_alias}_average_success"] = success_rate

            self.env.close()

        return metrics

    def _get_local_action(self, global_action: np.ndarray) -> np.ndarray:
        base_mat = self.env.sim.data.get_site_xmat(
            f"mobilebase{self.env.robots[0].idn}_center"
        )

        global_action_pos = global_action[:3]
        global_action_axis_angle = global_action[3:6]
        global_action_mat = T.quat2mat(T.axisangle2quat(global_action_axis_angle))

        local_action_pos = base_mat.T @ global_action_pos
        local_action_mat = base_mat.T @ global_action_mat @ base_mat
        local_action_axis_angle = T.quat2axisangle(T.mat2quat(local_action_mat))

        local_action = np.concatenate(
            [local_action_pos, local_action_axis_angle, global_action[6:]]
        )
        return local_action

    def _init_env(
        self,
        env_name,
        img_width,
        img_height,
        render,
    ):
        base_env = create_env(
            env_name=env_name,
            camera_widths=img_width,
            camera_heights=img_height,
            camera_names=self.camera_names,
            render_onscreen=render,
            seed=self.seed,
            layout_ids=self.layout_ids,
            style_ids=self.style_ids,
            obj_instance_split=self.obj_instance_split,
        )
        hard_reset_override = os.environ.get("ROBOT_DIFT_ROBOCASA_HARD_RESET", "").strip().lower()
        if hard_reset_override in {"0", "false", "no"} and hasattr(base_env, "hard_reset"):
            base_env.hard_reset = False
            log.info("Set RoboCasa/robosuite hard_reset=False via ROBOT_DIFT_ROBOCASA_HARD_RESET")
        elif hard_reset_override in {"1", "true", "yes"} and hasattr(base_env, "hard_reset"):
            base_env.hard_reset = True
            log.info("Set RoboCasa/robosuite hard_reset=True via ROBOT_DIFT_ROBOCASA_HARD_RESET")

        self.env = RobosuiteWrapper(base_env)

    def _prepare_video_dir(
        self,
        agent: BaseAgent,
        save_videos: bool,
        requested_dir: str | None,
    ) -> Path | None:
        if not save_videos or not self.camera_names:
            return None

        base_dir = requested_dir or getattr(agent, "working_dir", None) or self.working_dir
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        video_dir = base_path / "sim_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def _record_video_frame(
        self,
        writer,
        video_path: Path,
        fps: int,
        obs,
    ):
        frame = self._compose_video_frame(obs)
        if frame is None:
            return writer, False

        if writer is None:
            writer = self._create_video_writer(video_path, fps, frame.shape)
            if writer is None:
                return None, False

        writer.write(frame)
        return writer, True

    def _compose_video_frame(self, obs):
        frames = []
        for cam_name in self.camera_names:
            cam_key = f"{cam_name}_image"
            if cam_key not in obs:
                continue
            frame = np.asarray(obs[cam_key])
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames.append(frame)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return np.concatenate(frames, axis=1)

    def _create_video_writer(self, video_path: Path, fps: int, frame_shape):
        if frame_shape is None:
            return None
        height, width = frame_shape[:2]
        if height == 0 or width == 0:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
