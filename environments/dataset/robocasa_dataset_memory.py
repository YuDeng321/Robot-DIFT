import json
import os
import uuid

import h5py
import numpy as np
import torch
from termcolor import cprint

from environments.dataset.base_dataset import TrajectoryDataset
from environments.utils.robocasa_aliases import normalize_env_names


def _demo_sort_key(name: str):
    prefix, _, suffix = name.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return 0, int(suffix)
    return 1, name


class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        cam_names: list[str],
        env_name: list[str],
        data_directory: os.PathLike,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        max_len_data: int = 256,
        window_size: int = 1,
        global_action: bool = False,
        max_demos_per_task: int | None = None,
        obs_seq_len: int = 1,
        return_frame_ids: bool = False,
    ):
        if obs_seq_len < 1 or obs_seq_len > window_size:
            raise ValueError("obs_seq_len must be between 1 and window_size")
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        # Normalize aliases so downstream folder lookup succeeds.
        self.env_name = normalize_env_names(env_name)
        self.cam_names = cam_names
        self.obs_seq_len = obs_seq_len
        self.return_frame_ids = bool(return_frame_ids)
        self.frame_cache_namespace = uuid.uuid4().int & ((1 << 63) - 1) if self.return_frame_ids else None

        self.action_key = "global_actions" if global_action else "actions"

        self.slices = []

        self.data = {}

        for cam_name in self.cam_names:
            self.data[cam_name] = []
        self.data["lang"] = []
        self.data["action"] = []
        self.data["robot_states"] = []

        i = 0
        self.envs_data = []
        for env in self.env_name:
            if 'PnP' in env:
                data_dir = os.path.join(data_directory, "kitchen_pnp", env)
            elif 'Door' in env:
                data_dir = os.path.join(data_directory, "kitchen_doors", env)
            elif 'Drawer' in env:
                data_dir = os.path.join(data_directory, "kitchen_drawer", env)
            elif 'Coffee' in env:
                data_dir = os.path.join(data_directory, "kitchen_coffee", env)
            elif 'Stove' in env:
                data_dir = os.path.join(data_directory, "kitchen_stove", env)
            elif 'Microwave' in env:
                data_dir = os.path.join(data_directory, "kitchen_microwave", env)
            elif 'Sink' in env:
                data_dir = os.path.join(data_directory, "kitchen_sink", env)
            elif 'Navigate' in env:
                data_dir = os.path.join(data_directory, "kitchen_navigate", env)
            else:
                raise ValueError(f"Unknown environment: {env}")

            data_dir = os.path.join(data_dir, os.listdir(data_dir)[0], "demo_gentex_im128_randcams.hdf5")

            env_data = h5py.File(data_dir, "r")
            env_data = env_data["data"]

            demo_names = sorted(env_data.keys(), key=_demo_sort_key)
            num_available_demos = len(demo_names)
            if max_demos_per_task is not None:
                demo_names = demo_names[:max_demos_per_task]
            print(
                f"[RobocasaDataset] {env}: using {len(demo_names)}/{num_available_demos} demos"
            )

            for demo in demo_names:

                demo_length = env_data[demo].attrs["num_samples"]

                if demo_length - self.window_size < 0:
                    print(
                        f"Ignored short sequence #{i}: len={demo_length}, window={self.window_size}"
                    )
                else:
                    self.slices += [
                        (i, start, start + self.window_size)
                        for start in range(demo_length - self.window_size + 1)
                    ]  # slice indices follow convention [start, end)

                self.data["lang"].append(json.loads(env_data[demo].attrs["ep_meta"])["lang"])
                actions = env_data[demo][self.action_key][:, :self.action_dim]
                self.data["action"].append(actions)

                gripper_qpos = env_data[demo]["obs"]["robot0_gripper_qpos"][:, :1]
                gripper_qvel = env_data[demo]["obs"].get("robot0_gripper_qvel")
                if gripper_qvel is None:
                    gripper_qvel = np.zeros_like(gripper_qpos)
                else:
                    gripper_qvel = gripper_qvel[:, :1]

                joint_pos = env_data[demo]["obs"].get("robot0_joint_pos")
                if joint_pos is None:
                    joint_pos_cos = env_data[demo]["obs"].get("robot0_joint_pos_cos")
                    joint_pos_sin = env_data[demo]["obs"].get("robot0_joint_pos_sin")
                    if joint_pos_cos is None or joint_pos_sin is None:
                        raise ValueError("Joint positions not found in dataset")
                    joint_pos = np.arctan2(joint_pos_sin[:], joint_pos_cos[:])
                else:
                    joint_pos = joint_pos[:]
                joint_pos = joint_pos[:, :7]

                joint_vel = env_data[demo]["obs"].get("robot0_joint_vel")
                if joint_vel is None:
                    joint_vel = np.zeros_like(joint_pos)
                else:
                    joint_vel = joint_vel[:, :7]

                eef_pos = env_data[demo]["obs"].get("robot0_eef_pos")
                if eef_pos is None:
                    raise ValueError("robot0_eef_pos missing from dataset")
                eef_pos = eef_pos[:]

                eef_quat = env_data[demo]["obs"].get("robot0_eef_quat")
                if eef_quat is None:
                    raise ValueError("robot0_eef_quat missing from dataset")
                eef_quat = eef_quat[:]

                prev_action = np.zeros_like(actions)
                if len(prev_action) > 1:
                    prev_action[1:] = actions[:-1]

                robot_state = np.concatenate(
                    [
                        gripper_qpos,
                        gripper_qvel,
                        joint_pos,
                        joint_vel,
                        eef_pos,
                        eef_quat,
                        prev_action,
                    ],
                    axis=1,
                )
                self.data["robot_states"].append(robot_state)

                for cam_name in self.cam_names:
                    self.data[cam_name].append(env_data[demo]["obs"][f"{cam_name}_image"])

                i += 1

        cprint(f"Using dataset: {data_directory}", "green")
        cprint(f"Using action key: {self.action_key}", "blue")

        # Compute robot state bounds and store for downstream normalization
        all_robot_states = np.concatenate(self.data["robot_states"], axis=0)
        self.robot_states_min = torch.from_numpy(all_robot_states.min(0)).float()
        self.robot_states_max = torch.from_numpy(all_robot_states.max(0)).float()
        self.robot_states_denom = torch.clamp(self.robot_states_max - self.robot_states_min, min=1e-6)
        self.robot_state_bounds = {
            "min": self.robot_states_min.clone(),
            "max": self.robot_states_max.clone(),
        }
        cprint("Robot states bounds computed (for [-1, 1] normalization):", "cyan")
        cprint(f"  Min: {self.robot_states_min.numpy()}", "cyan")
        cprint(f"  Max: {self.robot_states_max.numpy()}", "cyan")

    def get_seq_length(self, idx):
        return self.data["action"][idx].shape[0]

    def get_all_actions(self):
        result = []

        for action in self.data["action"]:
            result.append(torch.from_numpy(action))

        return torch.cat(result, dim=0).to(self.device)

    def get_all_observations(self):
        result = [torch.from_numpy(rs) for rs in self.data["robot_states"]]
        return torch.cat(result, dim=0).to(self.device)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        action = torch.from_numpy(self.data["action"][i][start:end]).float()

        obs = {}

        for cam_name in self.cam_names:
            rgb = (
                torch.from_numpy(self.data[cam_name][i][start:start+self.obs_seq_len])
                .float()
                .permute(0, 3, 1, 2)
                / 255.0
            )
            obs[f"{cam_name}_image"] = rgb

        obs["lang"] = self.data["lang"][i]
        if self.return_frame_ids:
            # Stable within this dataset instance; only used for opt-in
            # deterministic frozen-encoder feature caching.
            obs["_robot_dift_frame_ids"] = torch.stack(
                (
                    torch.full((self.obs_seq_len,), self.frame_cache_namespace, dtype=torch.long),
                    torch.full((self.obs_seq_len,), i, dtype=torch.long),
                    torch.arange(start, start + self.obs_seq_len, dtype=torch.long),
                ),
                dim=-1,
            )

        robot_states = torch.from_numpy(
            self.data["robot_states"][i][start:start+self.obs_seq_len]
        ).float()
        robot_states = self._normalize_robot_states(robot_states)
        obs["robot_states"] = robot_states

        return obs, action, torch.ones(action.shape[0])  # TODO is this mask correct?

    def _normalize_robot_states(self, robot_states: torch.Tensor) -> torch.Tensor:
        return 2.0 * (robot_states - self.robot_states_min) / self.robot_states_denom - 1.0
