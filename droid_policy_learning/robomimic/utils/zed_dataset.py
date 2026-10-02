"""
ZED Dataset for DROID Training Framework

将ZED的HDF5+SVO数据格式适配为RLDS风格，与DROID训练框架无缝集成。

使用方法:
    from robomimic.utils.zed_dataset import make_zed_dataset

    dataset = make_zed_dataset(
        data_directory='data',
        window_size=2,
        future_action_window_size=15,
        image_size=(256, 256),
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
import numpy as np
import cv2
import torch

# 获取项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from example_reader import TrajectoryReader


class ZEDRLDSDataset:
    """
    ZED数据集的RLDS适配器

    输出格式与DROID RLDS一致，可直接用于robomimic训练:
    {
        "observation": {
            "image_primary": [T, H, W, 3] uint8,
            "image_secondary": [T, H, W, 3] uint8,
            "proprio": [T, 7] float32,
            "pad_mask": [T] bool,
        },
        "action": [T, 10] float32,  # 3(pos) + 6(rot6d) + 1(gripper)
        "task": {
            "language_instruction": str,
        }
    }
    """

    # DROID标准摄像头配置
    # 11022812: hand_camera (手眼相机，跟随机械臂末端)
    # 24285872: varied_camera (外部固定相机)
    HAND_CAMERA = '11022812'
    VARIED_CAMERA_1 = '24285872'

    def __init__(
        self,
        data_directory: str,
        window_size: int = 2,
        future_action_window_size: int = 15,
        subsample_length: int = 100,
        image_size: tuple = (256, 256),
        stereo_mode: str = 'left',
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.data_directory = data_directory
        self.window_size = window_size
        self.future_action_window_size = future_action_window_size
        self.subsample_length = subsample_length
        self.image_size = image_size
        self.stereo_mode = stereo_mode
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        # 初始化随机数生成器（必须在_compute_statistics之前）
        self.rng = np.random.RandomState(seed)

        self.trajectory_paths = self._scan_trajectories()
        print(f"✓ Found {len(self.trajectory_paths)} ZED trajectories in {data_directory}")

        self.dataset_statistics = self._compute_statistics()

    def _scan_trajectories(self) -> List[Path]:
        """扫描所有HDF5轨迹文件"""
        data_path = Path(self.data_directory)
        trajectory_paths = []

        # 查找所有HDF5文件（.h5, .hdf5）
        for pattern in ["*.h5", "*.hdf5"]:
            for hdf5_file in sorted(data_path.rglob(pattern)):
                trajectory_paths.append(hdf5_file)

        if len(trajectory_paths) == 0:
            print(f"Warning: No HDF5 files found in {data_path}")
            print("Expected file patterns: *.h5 or *.hdf5")

        return trajectory_paths

    def _compute_statistics(self) -> List[Dict[str, Any]]:
        """计算数据集统计信息 - 返回每个轨迹的统计信息列表"""
        sample_size = min(50, len(self.trajectory_paths))
        if sample_size == 0:
            return [self._default_statistics()]

        sample_paths = self.rng.choice(self.trajectory_paths, sample_size, replace=False)

        stats_list = []
        total_transitions = 0

        for traj_path in sample_paths:
            try:
                svo_dir = traj_path.parent
                if not any(svo_dir.glob("*.svo*")):
                    svo_subdir = traj_path.parent / "SVO"
                    if svo_subdir.exists():
                        svo_dir = svo_subdir

                reader = TrajectoryReader(str(traj_path), svo_dir=str(svo_dir), read_images=False)
                T = reader.length()
                total_transitions += T

                # 采样几个时间步来估计统计量
                sample_indices = np.linspace(0, T-1, min(10, T), dtype=int)
                traj_actions = []
                traj_states = []

                for idx in sample_indices:
                    timestep = reader.read_timestep(index=idx)
                    if 'action' in timestep:
                        action = timestep['action']
                        if isinstance(action, dict) and 'cartesian_position' in action and 'gripper_position' in action:
                            cart_pos = np.array(action['cartesian_position'])  # [6]
                            gripper_pos = float(action['gripper_position'])   # scalar
                            # 构造动作: [pos(3), rot_euler(3), gripper(1)]
                            action_array = np.concatenate([
                                cart_pos[:3],   # position
                                cart_pos[3:6],  # rotation
                                [gripper_pos]   # gripper
                            ])
                            traj_actions.append(action_array)
                        elif isinstance(action, (list, np.ndarray)):
                            action_array = np.array(action)
                            if action_array.shape[0] == 7:
                                traj_actions.append(action_array)

                    if 'observations' in timestep and 'robot_state' in timestep['observations']:
                        state = timestep['observations']['robot_state']
                        if isinstance(state, dict) and 'joint_positions' in state:
                            state_array = np.array(state['joint_positions'])
                        elif isinstance(state, (list, np.ndarray)):
                            state_array = np.array(state)
                        else:
                            state_array = None

                        if state_array is not None and len(state_array) >= 7:
                            traj_states.append(state_array[:7])

                if traj_actions and traj_states:
                    actions_concat = np.array(traj_actions)
                    proprio_concat = np.array(traj_states)

                    action_stats = {
                        "mean": np.mean(actions_concat, axis=0),
                        "std": np.std(actions_concat, axis=0) + 1e-6,
                        "min": np.min(actions_concat, axis=0),
                        "max": np.max(actions_concat, axis=0),
                        "q01": np.percentile(actions_concat, 1, axis=0),
                        "q99": np.percentile(actions_concat, 99, axis=0),
                    }

                    proprio_stats = {
                        "mean": np.mean(proprio_concat, axis=0),
                        "std": np.std(proprio_concat, axis=0) + 1e-6,
                        "min": np.min(proprio_concat, axis=0),
                        "max": np.max(proprio_concat, axis=0),
                    }

                    traj_stats = {
                        "action": action_stats,
                        "proprio": proprio_stats,
                        "num_trajectories": 1,  # 每个轨迹算一个数据集
                        "num_transitions": T,
                    }
                    stats_list.append(traj_stats)
                else:
                    # 如果无法提取动作或状态，使用默认统计信息
                    default_stats = self._default_statistics()
                    default_stats["num_transitions"] = T
                    stats_list.append(default_stats)

                reader.close()

            except Exception as e:
                print(f"Warning: Failed to load {traj_path}: {e}")
                continue

        if not stats_list:
            stats_list = [self._default_statistics()]

        return stats_list

    def _default_statistics(self):
        """默认统计信息"""
        action_dim = 7  # 修改为7维动作
        return {
            "action": {
                "mean": np.zeros(action_dim, dtype=np.float32),
                "std": np.ones(action_dim, dtype=np.float32),
                "min": -np.ones(action_dim, dtype=np.float32),
                "max": np.ones(action_dim, dtype=np.float32),
                "q01": -np.ones(action_dim, dtype=np.float32),
                "q99": np.ones(action_dim, dtype=np.float32),
            },
            "proprio": {
                "mean": np.zeros(7, dtype=np.float32),
                "std": np.ones(7, dtype=np.float32),
                "min": -np.ones(7, dtype=np.float32),
                "max": np.ones(7, dtype=np.float32),
            },
            "num_trajectories": 1,
            "num_transitions": 1000,
        }

    def _euler_to_rot6d(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转6D旋转表示"""
        def euler_to_matrix(roll, pitch, yaw):
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])
            return R

        original_shape = euler.shape
        euler_flat = euler.reshape(-1, 3)

        rot6d_list = []
        for e in euler_flat:
            R = euler_to_matrix(e[0], e[1], e[2])
            rot6d = R[:, :2].T.flatten()
            rot6d_list.append(rot6d)

        rot6d = np.stack(rot6d_list, axis=0)
        rot6d = rot6d.reshape(*original_shape[:-1], 6)
        return rot6d

    def _process_stereo_image(self, img: np.ndarray) -> np.ndarray:
        """处理ZED立体图像"""
        if img.shape[1] == 2560:
            if self.stereo_mode == 'left':
                img = img[:, :1280, :]
            elif self.stereo_mode == 'right':
                img = img[:, 1280:, :]
        return img

    def _load_trajectory(self, traj_path: Path) -> Dict[str, np.ndarray]:
        """加载单条轨迹 - 从HDF5和SVO文件读取"""
        return self._load_trajectory_from_hdf5_svo(traj_path)

    def _load_trajectory_from_hdf5_svo(self, traj_path: Path) -> Dict[str, np.ndarray]:
        """直接从HDF5和SVO文件加载轨迹（优化版本，不需要预提取PNG）"""
        import h5py

        # SVO文件在recordings/SVO子目录
        svo_dir = traj_path.parent / "recordings" / "SVO"
        if not svo_dir.exists():
            svo_dir = traj_path.parent  # 备用：同目录

        with h5py.File(traj_path, 'r') as f:
            # 读取元数据
            metadata = {}
            for key in f.attrs.keys():
                metadata[key] = f.attrs[key]

            # 读取动作和状态
            T = f['action/cartesian_position'].shape[0]

            # 子采样 - 这是关键优化！只读取需要的帧
            if T > self.subsample_length:
                indices = np.linspace(0, T-1, self.subsample_length, dtype=int)
                T = self.subsample_length
            else:
                indices = np.arange(T)

            # 读取动作
            cart_pos = f['action/cartesian_position'][indices]  # [T, 6]
            gripper = f['action/gripper_position'][indices]      # [T]

            # 构造动作: [pos(3), rot_euler(3), gripper(1)]
            actions_raw = np.concatenate([
                cart_pos[:, :3],   # position
                cart_pos[:, 3:6],  # rotation
                gripper.reshape(-1, 1)  # gripper
            ], axis=1)

            # 读取状态
            joint_pos = f['observation/robot_state/joint_positions'][indices]  # [T, 7]

            # 动作格式转换
            actions_pos = actions_raw[:, :3]
            actions_rot_euler = actions_raw[:, 3:6]
            actions_gripper = actions_raw[:, 6:7]
            actions_rot6d = self._euler_to_rot6d(actions_rot_euler)
            actions = np.concatenate([actions_pos, actions_rot6d, actions_gripper], axis=-1)

            # 读取SVO图像 - 只读取子采样的帧
            image_primary_list = []
            image_secondary_list = []

            hand_camera_svo = svo_dir / f"{self.HAND_CAMERA}.svo2"
            varied_camera_1_svo = svo_dir / f"{self.VARIED_CAMERA_1}.svo2"

            # 批量提取需要的帧（比逐帧提取快）
            for idx in indices:
                # 手眼相机 (hand_camera)
                if hand_camera_svo.exists():
                    frame = self._read_svo_frame_cached(hand_camera_svo, idx)
                    if frame is not None:
                        frame = self._process_stereo_image(frame)
                        frame = cv2.resize(frame, self.image_size[::-1])
                        image_primary_list.append(frame)
                    else:
                        image_primary_list.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
                else:
                    image_primary_list.append(np.zeros((*self.image_size, 3), dtype=np.uint8))

                # 外部固定相机 (varied_camera_1)
                if varied_camera_1_svo.exists():
                    frame = self._read_svo_frame_cached(varied_camera_1_svo, idx)
                    if frame is not None:
                        frame = self._process_stereo_image(frame)
                        frame = cv2.resize(frame, self.image_size[::-1])
                        image_secondary_list.append(frame)
                    else:
                        image_secondary_list.append(image_primary_list[-1].copy())
                else:
                    image_secondary_list.append(image_primary_list[-1].copy())

            image_primary = np.array(image_primary_list)
            image_secondary = np.array(image_secondary_list)
            proprio = joint_pos.astype(np.float32)
            pad_mask = np.ones(T, dtype=bool)

            # 任务描述
            language_instruction = metadata.get('current_task', traj_path.stem)
            if isinstance(language_instruction, bytes):
                language_instruction = language_instruction.decode('utf-8')

            return {
                "observation": {
                    "image_primary": image_primary,
                    "image_secondary": image_secondary,
                    "proprio": proprio,
                    "pad_mask": pad_mask,
                },
                "action": actions.astype(np.float32),
                "task": {
                    "language_instruction": str(language_instruction),
                },
            }

    def _read_svo_frame_cached(self, svo_file: Path, frame_idx: int) -> np.ndarray:
        """读取SVO帧（带简单缓存）"""
        import subprocess

        try:
            # 使用ffmpeg提取单帧
            cmd = [
                'ffmpeg',
                '-i', str(svo_file),
                '-vf', f'select=eq(n\\,{frame_idx})',
                '-vframes', '1',
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                # 解码原始RGB数据
                # ZED相机分辨率720×2560 (立体)
                frame = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = frame.reshape((720, 2560, 3))
                return frame
            else:
                return None

        except Exception as e:
            return None

    def _load_trajectory_from_svo(self, traj_path: Path) -> Dict[str, np.ndarray]:
        """从SVO文件加载轨迹（慢速，仅作为备选）"""
        # SVO文件在recordings/SVO子目录
        svo_dir = traj_path.parent / "recordings" / "SVO"
        if not svo_dir.exists():
            svo_dir = traj_path.parent  # 备用：同目录

        # 创建TrajectoryReader (如果没有SVO文件也没关系，图像会是零数组)
        reader = TrajectoryReader(str(traj_path), svo_dir=str(svo_dir) if svo_dir.exists() else None, read_images=True)

        try:
            T = reader.length()

            # 子采样
            if T > self.subsample_length:
                indices = np.linspace(0, T-1, self.subsample_length, dtype=int).astype(int)
                T = self.subsample_length
            else:
                indices = np.arange(T)

            # 预分配数组
            actions_list = []
            states_list = []
            image_primary_list = []
            image_secondary_list = []

            for idx in indices:
                timestep = reader.read_timestep(index=idx)

                # 读取动作 - 使用cartesian_position和gripper_position
                if 'action' in timestep:
                    action_data = timestep['action']
                    cart_pos = action_data.get('cartesian_position', np.zeros(6))  # [x,y,z,rx,ry,rz]
                    gripper = action_data.get('gripper_position', 0.0)

                    # 构造动作: [pos(3), rot_euler(3), gripper(1)]
                    action_array = np.concatenate([
                        cart_pos[:3],  # position
                        cart_pos[3:6],  # rotation (euler angles)
                        [gripper]  # gripper
                    ])
                    actions_list.append(action_array)
                else:
                    actions_list.append(np.zeros(7))

                # 读取状态 - 使用robot_state/joint_position
                if 'observation' in timestep and 'robot_state' in timestep['observation']:
                    robot_state = timestep['observation']['robot_state']
                    joint_pos = robot_state.get('joint_positions', np.zeros(7))
                    joint_vel = robot_state.get('joint_velocity', np.zeros(7))

                    # 构造状态: 简化版只用joint_position
                    state_array = np.array(joint_pos)
                    states_list.append(state_array)
                else:
                    states_list.append(np.zeros(7))

                # 读取图像
                if 'observation' in timestep and 'image' in timestep['observation']:
                    images = timestep['observation']['image']

                    # 主相机 (11022812)
                    if self.PRIMARY_CAMERA in images:
                        img = images[self.PRIMARY_CAMERA]
                        img = self._process_stereo_image(img)
                        img = cv2.resize(img, self.image_size[::-1])
                        image_primary_list.append(img)
                    else:
                        image_primary_list.append(np.zeros((*self.image_size, 3), dtype=np.uint8))

                    # 副相机 (24285872)
                    if self.SECONDARY_CAMERA in images:
                        img = images[self.SECONDARY_CAMERA]
                        img = self._process_stereo_image(img)
                        img = cv2.resize(img, self.image_size[::-1])
                        image_secondary_list.append(img)
                    else:
                        # 如果没有副相机，复制主相机
                        image_secondary_list.append(image_primary_list[-1].copy())
                else:
                    # 没有图像数据，使用零数组
                    image_primary_list.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
                    image_secondary_list.append(np.zeros((*self.image_size, 3), dtype=np.uint8))

            # 转换为numpy数组
            actions_raw = np.array(actions_list)  # [T, 7]
            states_raw = np.array(states_list)    # [T, 7]

            # 动作格式转换: [pos, rot_euler, gripper] -> [pos, rot6d, gripper]
            actions_pos = actions_raw[:, :3]
            actions_rot_euler = actions_raw[:, 3:6]
            actions_gripper = actions_raw[:, 6:7]
            actions_rot6d = self._euler_to_rot6d(actions_rot_euler)
            actions = np.concatenate([actions_pos, actions_rot6d, actions_gripper], axis=-1)  # [T, 10]

            # Proprio: 使用joint_position
            proprio = states_raw  # [T, 7]

            # 图像数组
            image_primary = np.array(image_primary_list)
            image_secondary = np.array(image_secondary_list)

            pad_mask = np.ones(T, dtype=bool)

            # 从元数据获取任务描述
            metadata = reader.get_metadata()
            if 'current_task' in metadata and metadata['current_task']:
                language_instruction = str(metadata['current_task'])
            else:
                task_name = traj_path.stem
                language_instruction = f"Perform task: {task_name}"

            return {
                "observation": {
                    "image_primary": image_primary,
                    "image_secondary": image_secondary,
                    "proprio": proprio.astype(np.float32),
                    "pad_mask": pad_mask,
                },
                "action": actions.astype(np.float32),
                "task": {
                    "language_instruction": language_instruction,
                },
            }

        finally:
            reader.close()

    def _create_windows(self, trajectory: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """创建窗口化样本"""
        T = trajectory["action"].shape[0]
        total_window = self.window_size + self.future_action_window_size

        if T < total_window:
            pad_length = total_window - T
            for key in ["image_primary", "image_secondary", "proprio", "pad_mask"]:
                traj_data = trajectory["observation"][key]
                pad_shape = (pad_length,) + traj_data.shape[1:]
                pad_data = np.zeros(pad_shape, dtype=traj_data.dtype)
                trajectory["observation"][key] = np.concatenate([traj_data, pad_data], axis=0)

            action_pad = np.zeros((pad_length, trajectory["action"].shape[1]), dtype=trajectory["action"].dtype)
            trajectory["action"] = np.concatenate([trajectory["action"], action_pad], axis=0)
            trajectory["observation"]["pad_mask"][T:] = False
            T = total_window

        windows = []
        for start_idx in range(T - total_window + 1):
            end_obs = start_idx + self.window_size
            end_action = start_idx + total_window

            window = {
                "observation": {
                    "image_primary": trajectory["observation"]["image_primary"][start_idx:end_obs],
                    "image_secondary": trajectory["observation"]["image_secondary"][start_idx:end_obs],
                    "proprio": trajectory["observation"]["proprio"][start_idx:end_obs],
                    "pad_mask": trajectory["observation"]["pad_mask"][start_idx:end_action],
                },
                "action": trajectory["action"][start_idx:end_action],
                "task": trajectory["task"],
            }
            windows.append(window)

        return windows

    def as_numpy_iterator(self) -> Iterator[Dict[str, np.ndarray]]:
        """返回numpy迭代器"""
        shuffled_indices = self.rng.permutation(len(self.trajectory_paths))
        buffer = []

        for idx in shuffled_indices:
            traj_path = self.trajectory_paths[idx]

            try:
                trajectory = self._load_trajectory(traj_path)
                windows = self._create_windows(trajectory)
                buffer.extend(windows)

                while len(buffer) >= self.shuffle_buffer_size:
                    pop_idx = self.rng.randint(0, len(buffer))
                    yield buffer.pop(pop_idx)

            except Exception as e:
                print(f"Warning: Failed to process {traj_path}: {e}")
                continue

        self.rng.shuffle(buffer)
        for sample in buffer:
            yield sample

    def map(self, transform_fn, num_parallel_calls=None):
        """应用transform"""
        return ZEDRLDSDatasetMapped(self, transform_fn)


class ZEDRLDSDatasetMapped:
    """应用了transform的ZED数据集"""

    def __init__(self, base_dataset: ZEDRLDSDataset, transform_fn):
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn
        self.dataset_statistics = base_dataset.dataset_statistics

    def as_numpy_iterator(self) -> Iterator[Dict[str, Any]]:
        for sample in self.base_dataset.as_numpy_iterator():
            yield self.transform_fn(sample)


def make_zed_dataset(
    data_directory: str,
    window_size: int = 2,
    future_action_window_size: int = 15,
    subsample_length: int = 100,
    image_size: tuple = (256, 256),
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
):
    """
    创建ZED数据集

    Args:
        data_directory: ZED数据目录
        window_size: 观测窗口 (observation_horizon)
        future_action_window_size: 未来动作窗口 (prediction_horizon - 1)
        subsample_length: 轨迹子采样长度
        image_size: 图像尺寸 (H, W)
        shuffle_buffer_size: shuffle缓冲区大小
        seed: 随机种子

    Returns:
        dataset: ZED RLDS数据集
        statistics: 数据集统计信息
    """
    dataset = ZEDRLDSDataset(
        data_directory=data_directory,
        window_size=window_size,
        future_action_window_size=future_action_window_size,
        subsample_length=subsample_length,
        image_size=image_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )

    return dataset, dataset.dataset_statistics
