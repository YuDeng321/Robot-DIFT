"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg

def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action_dict"]["cartesian_position"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_position"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


DROID_RLDS_IMAGE_SLOTS = ("primary", "secondary", "tertiary")


def _slot_to_transformed_obs_key(slot: str) -> str:
    return f"image_{slot}"


def robomimic_transform(
    trajectory: Dict[str, Any],
    normalize_to_neg_one_one: bool = False,
    include_proprio: bool = True,
    view_dropout_prob: float = 0.0,
    obs_camera_keys: Optional[List[str]] = None,
    rlds_image_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Transform trajectory to robomimic format.

    Args:
        trajectory: Input trajectory dict
        normalize_to_neg_one_one: If True, normalize images to [-1, 1] (for diffusion models like CleanDIFT)
                                  If False, normalize to [0, 1] (default for ResNet etc.)
        include_proprio: If True, include proprioceptive state in obs dict
                        If False, only include images (pure RGB mode for CleanDIFT)
        obs_camera_keys: Robomimic observation keys to emit. Defaults to the
                         historical two exterior-camera keys.
        rlds_image_keys: Transformed RLDS observation keys to read. These are
                         usually ["image_primary", "image_secondary", ...],
                         where the suffixes match image_obs_keys passed to Octo.
    """
    if obs_camera_keys is None:
        obs_camera_keys = [
            "camera/image/varied_camera_1_left_image",
            "camera/image/varied_camera_2_left_image",
        ]
    if rlds_image_keys is None:
        rlds_image_keys = [_slot_to_transformed_obs_key(slot) for slot in DROID_RLDS_IMAGE_SLOTS[:len(obs_camera_keys)]]
    if len(obs_camera_keys) != len(rlds_image_keys):
        raise ValueError(
            f"obs_camera_keys and rlds_image_keys must have the same length, got "
            f"{len(obs_camera_keys)} and {len(rlds_image_keys)}"
        )
    if len(obs_camera_keys) < 2:
        raise ValueError(f"At least two camera views are required, got {obs_camera_keys}")

    image_dict = {}
    for obs_key, source_key in zip(obs_camera_keys, rlds_image_keys):
        # Convert images from uint8 [0, 255] to float
        img = tf.cast(trajectory["observation"][source_key], tf.float32) / 255.

        # For diffusion-based encoders (CleanDIFT), convert [0, 1] to [-1, 1]
        if normalize_to_neg_one_one:
            img = img * 2.0 - 1.0
        image_dict[obs_key] = img

    # View dropout: randomly blank one camera view per trajectory
    if view_dropout_prob and view_dropout_prob > 0:
        drop = tf.less(tf.random.uniform([], 0.0, 1.0), float(view_dropout_prob))
        drop_value = -1.0 if normalize_to_neg_one_one else 0.0
        drop_idx = tf.random.uniform([], minval=0, maxval=len(obs_camera_keys), dtype=tf.int32)
        updated = {}
        for i, obs_key in enumerate(obs_camera_keys):
            img = image_dict[obs_key]
            fill_img = tf.ones_like(img) * tf.cast(drop_value, img.dtype)
            should_drop = tf.logical_and(drop, tf.equal(drop_idx, i))
            updated[obs_key] = tf.where(should_drop, fill_img, img)
        image_dict = updated

    # Build obs dict - start with images and language
    obs_dict = {
        "raw_language": trajectory["task"]["language_instruction"],
        "pad_mask": trajectory["observation"]["pad_mask"][..., None],
    }
    obs_dict.update(image_dict)

    # Add proprio if requested and available
    # Some datasets omit proprio; guard access to avoid KeyErrors when running RGB-only
    has_proprio = include_proprio and ("proprio" in trajectory.get("observation", {}))
    if has_proprio:
        obs_dict["robot_state/cartesian_position"] = trajectory["observation"]["proprio"][..., :6]
        obs_dict["robot_state/gripper_position"] = trajectory["observation"]["proprio"][..., -1:]

    return {
        "obs": obs_dict,
        "actions": trajectory["action"][1:],
    }

DROID_TO_RLDS_OBS_KEY_MAP = {
    # Robomimic config中的camera key -> RLDS数据集中的observation key
    "camera/image/hand_camera_left_image": "wrist_image_left",           # DROID标准手眼相机
    "camera/image/wrist_image_left": "wrist_image_left",                 # 真机数据手眼相机别名
    "camera/image/varied_camera_1_left_image": "exterior_image_1_left", # 外部相机1
    "camera/image/varied_camera_2_left_image": "exterior_image_2_left", # 外部相机2
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}


def real_robot_transform(
    trajectory: Dict[str, Any],
    normalize_to_neg_one_one: bool = False,
    include_proprio: bool = True,
) -> Dict[str, Any]:
    """
    Transform function for real robot data (ZED cameras + HDF5 format).

    Converts from real robot dataset format to DROID training format.

    Args:
        trajectory: Raw trajectory from real robot dataset
        normalize_to_neg_one_one: If True, normalize images to [-1, 1] instead of [0, 1]
        include_proprio: If True, include proprioceptive state

    Returns:
        Transformed trajectory compatible with DROID training
    """
    # 真实机器人数据集已经是uint8格式，转换为float
    img1 = trajectory["observation"]["image_primary"].astype(np.float32) / 255.
    img2 = trajectory["observation"]["image_secondary"].astype(np.float32) / 255.

    if normalize_to_neg_one_one:
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0

    # 构建obs dict
    obs_dict = {
        "camera/image/varied_camera_1_left_image": img1,
        "camera/image/varied_camera_2_left_image": img2,
        "raw_language": trajectory["task"]["language_instruction"],
        "pad_mask": trajectory["observation"]["pad_mask"][..., None],
    }

    # 添加proprio
    if include_proprio and "proprio" in trajectory["observation"]:
        obs_dict["robot_state/cartesian_position"] = trajectory["observation"]["proprio"][..., :6]
        obs_dict["robot_state/gripper_position"] = trajectory["observation"]["proprio"][..., -1:]

    return {
        "obs": obs_dict,
        "actions": trajectory["action"][1:],  # 去掉第一帧，与DROID一致
    }


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
        shuffle_buffer_size=None,
        dataset_length=None,  # 允许手动指定数据集长度
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._shuffle_buffer_size = shuffle_buffer_size
        self._dataset_length = dataset_length  # 缓存数据集长度

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        # 如果手动指定了数据集长度，直接使用
        if self._dataset_length is not None:
            return self._dataset_length

        # Try to get dataset statistics from various possible attributes
        dataset_stats = None
        if hasattr(self._rlds_dataset, 'dataset_statistics'):
            dataset_stats = self._rlds_dataset.dataset_statistics
        elif hasattr(self._rlds_dataset, '_dataset_statistics'):
            dataset_stats = self._rlds_dataset._dataset_statistics

        if dataset_stats is None:
            # 对于没有统计信息的数据集，使用shuffle_buffer_size作为估计
            # 用户应该通过dataset_length参数提供正确的长度
            print(f"⚠️  Warning: Dataset statistics not available.")
            print(f"   Using shuffle_buffer_size ({self._shuffle_buffer_size}) as fallback estimate.")
            print(f"   For accurate epoch counting, please specify dataset length via --dataset_length parameter.")
            return self._shuffle_buffer_size if hasattr(self, '_shuffle_buffer_size') else 100000

        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in dataset_stats
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)
