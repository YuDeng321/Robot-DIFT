"""Entry point for training diffusion policies with optional CleanDIFT encoder."""

import argparse
import glob
import json
import numpy as np
import random
import time
import os
import shutil
import psutil
import sys
import socket
import traceback
import tqdm
import gc
from datetime import timedelta

from collections import OrderedDict
from typing import Dict, List, Optional

# Suppress TensorFlow logging before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file as load_safetensors
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tensorflow as tf
# The RLDS pipeline runs on CPUs; prevent each DDP rank from reserving PyTorch's GPU.
tf.config.set_visible_devices([], "GPU")
from PIL import Image
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Further suppress TF logging
tf.get_logger().setLevel('ERROR')

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.action_utils as ActionUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.distributed_gradients import broadcast_training_state
from robomimic.utils.robot_dift_export import (
    export_robot_dift_encoder as _export_robot_dift_encoder,
    find_cleandift_encoder as _find_cleandift_encoder,
    unwrap_parallel as _unwrap_parallel,
)
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.rlds_utils import droid_dataset_transform, robomimic_transform, real_robot_transform, DROID_RLDS_IMAGE_SLOTS, DROID_TO_RLDS_OBS_KEY_MAP, DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP, TorchRLDSDataset
# Lazy import zed_dataset only when needed (requires example_reader)
# from robomimic.utils.zed_dataset import make_zed_dataset

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec


def _preprocessed_float_rgb_processor(obs):
    """Keep RLDS-preprocessed float RGB images and only convert HWC to CHW."""
    obs = ObsUtils.TU.to_float(obs)
    return ObsUtils.batch_image_hwc_to_chw(obs)


def compute_rlds_dataset_length(data_path, dataset_names):
    """
    快速计算RLDS数据集的总步数（不加载图像数据）

    Args:
        data_path: 数据集根目录
        dataset_names: 数据集名称列表

    Returns:
        总步数
    """
    import tensorflow_datasets as tfds

    total_steps = 0
    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_path, dataset_name)
        # 检查是否存在版本子目录
        if os.path.exists(dataset_path):
            subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            if subdirs and subdirs[0].replace('.', '').isdigit():  # 版本号格式如 "1.0.0"
                dataset_path = os.path.join(dataset_path, subdirs[0])

        try:
            builder = tfds.builder_from_directory(dataset_path)
            dataset = builder.as_dataset(split='train')

            # 只读取episode元数据，不加载图像
            dataset_steps = 0
            for episode in dataset:
                # 计算steps数量（不实际加载数据）
                steps = episode['steps']
                # 使用cardinality()获取steps数量（快速，不遍历）
                num_steps = steps.cardinality().numpy()
                if num_steps < 0:  # cardinality未知，需要遍历
                    num_steps = sum(1 for _ in steps)
                dataset_steps += num_steps

            print(f"  Dataset '{dataset_name}': {dataset_steps} steps")
            total_steps += dataset_steps
        except Exception as e:
            print(f"  Warning: Could not compute length for dataset '{dataset_name}': {e}")
            return None

    return total_steps


def visualize_trajectories(images, gt_actions, pred_actions, camera_names, epoch, config):
    """
    可视化真实轨迹和预测轨迹

    Args:
        images: dict, 包含不同相机的图像 {camera_name: tensor [B, T, C, H, W]}
        gt_actions: tensor [B, T, action_dim]，真实动作轨迹
        pred_actions: tensor [B, T, action_dim]，预测动作轨迹
        camera_names: list，相机名称列表
        epoch: int，当前epoch数
        config: 配置对象

    Returns:
        wandb.Image 对象列表
    """
    import wandb

    vis_images = []
    batch_size = gt_actions.shape[0]
    num_vis = min(4, batch_size)  # 最多可视化4个样本

    for b_idx in range(num_vis):
        # 创建figure
        num_cameras = len(camera_names)
        fig, axes = plt.subplots(1, num_cameras + 1, figsize=(6 * (num_cameras + 1), 6))
        if num_cameras == 1:
            axes = [axes]

        # 绘制每个相机的图像
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_name in images:
                # 取第一帧图像 [C, H, W]
                img = images[cam_name][b_idx, 0].cpu().numpy()
                # 转换为 [H, W, C]
                if img.shape[0] == 3:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
                # 如果是 [-1, 1] 范围，转换到 [0, 1]
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = np.clip(img, 0, 1)

                axes[cam_idx].imshow(img)
                axes[cam_idx].set_title(f'Camera {cam_idx + 1}', fontsize=12)
                axes[cam_idx].axis('off')

        # 绘制动作轨迹对比（XY平面）
        ax_traj = axes[-1]
        gt_traj = gt_actions[b_idx].cpu().numpy()  # [T, action_dim]
        pred_traj = pred_actions[b_idx].cpu().numpy()  # [T, action_dim]

        # 绘制 XY 轨迹（假设前2维是 X, Y 位置）
        ax_traj.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', linewidth=2, markersize=4, label='Ground Truth', alpha=0.8)
        ax_traj.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--s', linewidth=2, markersize=4, label='Predicted', alpha=0.8)

        # 标记起点和终点
        ax_traj.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=10, label='Start')
        ax_traj.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'g^', markersize=10, label='GT End')
        ax_traj.plot(pred_traj[-1, 0], pred_traj[-1, 1], 'r^', markersize=10, label='Pred End')

        ax_traj.set_xlabel('X Position', fontsize=12)
        ax_traj.set_ylabel('Y Position', fontsize=12)
        ax_traj.set_title(f'Trajectory (XY Plane) - Sample {b_idx + 1}', fontsize=12)
        ax_traj.legend(loc='best', fontsize=10)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.axis('equal')

        plt.tight_layout()

        # 转换为numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        vis_images.append(wandb.Image(img_array, caption=f"Epoch {epoch} - Sample {b_idx + 1}"))
        plt.close(fig)

    return vis_images


DEFAULT_CLUSTER_COLORS = torch.tensor([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
    [0, 128, 128],
    [220, 190, 255],
], dtype=torch.float32) / 255.0


class DummyDataLogger:
    """Dummy logger for non-rank-0 processes in DDP training."""
    def __init__(self):
        pass


def _project_feature_map(feature_map: torch.Tensor) -> List[torch.Tensor]:
    projections: List[torch.Tensor] = []
    if feature_map is None or feature_map.ndim != 4:
        return projections

    for sample in feature_map:
        sample_cpu = sample.detach().to(torch.float32).cpu()
        C, H, W = sample_cpu.shape
        flat = sample_cpu.reshape(C, H * W).transpose(0, 1)
        flat = flat - flat.mean(dim=0, keepdim=True)
        if flat.numel() == 0:
            continue
        q = min(3, flat.shape[1])
        try:
            _, _, V = torch.pca_lowrank(flat, q=q)
        except RuntimeError:
            continue
        proj = flat @ V[:, :q]
        if q < 3:
            pad = torch.zeros(flat.shape[0], 3 - q)
            proj = torch.cat([proj, pad], dim=1)
        proj = proj.reshape(H, W, 3)
        proj_min = proj.amin(dim=(0, 1), keepdim=True)
        proj_max = proj.amax(dim=(0, 1), keepdim=True)
        proj = proj - proj_min
        denom = proj_max - proj_min
        denom[denom < 1e-6] = 1.0
        proj = proj / denom
        projections.append(proj.permute(2, 0, 1).contiguous())

    return projections


def _get_cluster_palette(num_clusters: int) -> torch.Tensor:
    if num_clusters <= DEFAULT_CLUSTER_COLORS.shape[0]:
        return DEFAULT_CLUSTER_COLORS[:num_clusters]
    extra = num_clusters - DEFAULT_CLUSTER_COLORS.shape[0]
    g = torch.Generator().manual_seed(0)
    random_colors = torch.rand((extra, 3), generator=g)
    return torch.cat([DEFAULT_CLUSTER_COLORS, random_colors], dim=0)


def _cluster_assignments(sample_cpu: torch.Tensor, num_clusters: int, num_iters: int = 15) -> torch.Tensor:
    """
    Args:
        sample_cpu: Tensor (C, H, W)
    Returns:
        assignments: Tensor (H, W) of cluster ids
    """
    C, H, W = sample_cpu.shape
    flat = sample_cpu.reshape(C, H * W).transpose(0, 1)  # (N, C)
    N = flat.shape[0]
    if num_clusters <= 1 or N < num_clusters:
        return torch.zeros((H, W), dtype=torch.long)

    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(N, generator=g)[:num_clusters]
    centers = flat[indices].clone()

    for _ in range(num_iters):
        distances = torch.cdist(flat.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
        assignments = distances.argmin(dim=1)
        new_centers = []
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                new_centers.append(flat[mask].mean(dim=0))
            else:
                new_centers.append(centers[k])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers

    distances = torch.cdist(flat.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
    assignments = distances.argmin(dim=1)
    return assignments.reshape(H, W)


def _build_progress_postfix(log: Dict[str, float], prefix: str = "train") -> OrderedDict:
    """
    Build a compact set of metrics for tqdm progress bars.
    """
    postfix = OrderedDict()

    loss_key = "Total_Loss" if "Total_Loss" in log else "Loss"
    if loss_key in log:
        try:
            postfix[f"{prefix}_loss"] = f"{float(log[loss_key]):.4f}"
        except (TypeError, ValueError):
            pass

    time_key = None
    # Prefer full epoch timing if available
    if "Time_Epoch" in log:
        time_key = "Time_Epoch"
    elif "Time_Train_Batch" in log:
        time_key = "Time_Train_Batch"
    if time_key is not None:
        try:
            postfix[f"{prefix}_time_s"] = f"{float(log[time_key]):.2f}"
        except (TypeError, ValueError):
            pass

    if prefix == "train":
        lr_values = []
        for key, value in log.items():
            if key.startswith("Optimizer/") and key.endswith("_lr"):
                try:
                    lr_values.append(float(value))
                except (TypeError, ValueError):
                    pass
        if lr_values:
            try:
                postfix["lr_max"] = f"{max(lr_values):.2e}"
            except (TypeError, ValueError):
                pass

    return postfix


def _prepare_feature_vis_state(config, use_neg_one_one_norm: bool, device: torch.device):
    feature_vis_cfg = getattr(config.experiment, "feature_vis", None)
    if not feature_vis_cfg:
        return None

    if isinstance(feature_vis_cfg, dict):
        cfg = dict(feature_vis_cfg)
    else:
        try:
            cfg = dict(feature_vis_cfg)
        except TypeError:
            cfg = feature_vis_cfg

    image_dir = cfg.get("image_dir")
    if not image_dir:
        return None
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        return None

    image_keys = cfg.get("image_keys")
    if not image_keys:
        image_keys = list(config.observation.modalities.obs.rgb)

    max_images = int(cfg.get("max_images_per_key", 8))
    apply_token_mapper = bool(cfg.get("apply_token_mapper", False))
    grid_nrow = cfg.get("grid_nrow")
    log_prefix = cfg.get("log_prefix", "FeatureVis")
    frequency = cfg.get("frequency")
    if frequency is not None:
        try:
            frequency = int(frequency)
        except (TypeError, ValueError):
            frequency = None
    if frequency is not None and frequency <= 0:
        frequency = None
    num_clusters = cfg.get("num_clusters", 6)
    try:
        num_clusters = int(num_clusters)
    except (TypeError, ValueError):
        num_clusters = 6
    num_clusters = max(2, num_clusters)

    images: Dict[str, torch.Tensor] = {}
    loaded_paths: Dict[str, List[str]] = {}

    for key in image_keys:
        prefix = key.replace("/", "_")
        pattern = os.path.join(image_dir, f"{prefix}*.png")
        paths = sorted(glob.glob(pattern))
        if not paths:
            continue
        selected = paths[:max_images]
        tensors: List[torch.Tensor] = []
        for path in selected:
            with Image.open(path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0
            if use_neg_one_one_norm:
                arr = arr * 2.0 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
        if not tensors:
            continue
        stacked = torch.stack(tensors, dim=0).to(device)
        images[key] = stacked
        loaded_paths[key] = selected


    if not images:
        return None

    return {
        "images": images,
        "apply_token_mapper": apply_token_mapper,
        "grid_nrow": grid_nrow,
        "log_prefix": log_prefix,
        "frequency": frequency,
        "normalize_to_neg_one_one": use_neg_one_one_norm,
        "num_clusters": num_clusters,
        "last_epoch": None,
        "paths": loaded_paths,
    }


def _log_feature_maps(model, feature_vis_state, data_logger, epoch: int, device: torch.device):
    if feature_vis_state is None:
        return

    nets = model.nets.module if isinstance(model.nets, DDP) else model.nets
    obs_encoder = nets["policy"]["obs_encoder"]
    cleandift_encoder, _ = _find_cleandift_encoder(obs_encoder)
    if cleandift_encoder is None or not hasattr(cleandift_encoder, "extract_feature_map"):
        return

    prefix = feature_vis_state.get("log_prefix", "FeatureVis")
    apply_token_mapper = feature_vis_state.get("apply_token_mapper", False)
    grid_nrow = feature_vis_state.get("grid_nrow")
    normalize_neg_one_one = feature_vis_state.get("normalize_to_neg_one_one", False)

    encoder = _unwrap_parallel(cleandift_encoder)
    was_training = encoder.training
    logged_any = False

    with torch.no_grad():
        encoder.eval()
        for key, images in feature_vis_state["images"].items():
            inputs = images if images.device == device else images.to(device)
            fmap = encoder.extract_feature_map(
                inputs,
                apply_token_mapper=apply_token_mapper,
            )
            fmap_cpu = fmap.detach().cpu()
            projections = _project_feature_map(fmap_cpu)
            if not projections:
                continue
            cluster_k = int(feature_vis_state.get("num_clusters", 6))
            palette = _get_cluster_palette(cluster_k)

            pair_samples: List[torch.Tensor] = []
            for idx, proj in enumerate(projections):
                if idx >= inputs.shape[0]:
                    break
                orig = inputs[idx].detach().to(torch.float32)
                if orig.is_cuda:
                    orig = orig.cpu()
                if normalize_neg_one_one:
                    orig = (orig + 1.0) * 0.5
                orig = torch.clamp(orig, 0.0, 1.0)
                proj_tensor = torch.clamp(proj, 0.0, 1.0)
                if proj_tensor.shape[-2:] != orig.shape[-2:]:
                    proj_tensor = F.interpolate(
                        proj_tensor.unsqueeze(0),
                        size=orig.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                sample_cpu = fmap_cpu[idx]
                sample_up = F.interpolate(
                    sample_cpu.unsqueeze(0),
                    size=orig.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                assign = _cluster_assignments(sample_up, cluster_k)
                assign_flat = assign.reshape(-1).long()
                assign_flat = torch.clamp(assign_flat, max=palette.shape[0] - 1)
                cluster_colors = palette[assign_flat].reshape(assign.shape[0], assign.shape[1], 3)
                cluster_tensor = cluster_colors.permute(2, 0, 1).contiguous()
                if cluster_tensor.shape[-2:] != orig.shape[-2:]:
                    cluster_tensor = F.interpolate(
                        cluster_tensor.unsqueeze(0),
                        size=orig.shape[-2:],
                        mode="nearest",
                    ).squeeze(0)
                cluster_tensor = torch.clamp(cluster_tensor, 0.0, 1.0)
                overlay = torch.clamp(0.45 * orig + 0.55 * cluster_tensor, 0.0, 1.0)

                separator = torch.ones(
                    (3, orig.shape[-2], 2), dtype=orig.dtype, device=orig.device
                )
                pair = torch.cat([orig, separator, proj_tensor, separator, overlay], dim=2)
                pair_samples.append(pair)

            if not pair_samples:
                continue

            tensor = torch.stack(pair_samples, dim=0)
            nrow = grid_nrow if grid_nrow is not None and grid_nrow > 0 else min(4, tensor.size(0))
            grid = make_grid(tensor, nrow=nrow, padding=2)
            grid = torch.clamp(grid, 0.0, 1.0)
            img = grid.permute(1, 2, 0).numpy()
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            clean_key = key.replace("camera/image/", "").replace("/", "_")
            tag = f"{prefix}/{clean_key}"
            data_logger.record(tag, img_uint8, epoch, data_type='image')
            logged_any = True

    if was_training:
        encoder.train()
    if logged_any:
        feature_vis_state["last_epoch"] = epoch




def train(config, device):
    """
    Train a model using the algorithm.
    """

    # Initialize DDP if enabled
    use_ddp = getattr(config.train, 'use_ddp', False)
    is_distributed = dist.is_available() and dist.is_initialized()

    if use_ddp and not is_distributed:
        # Initialize process group
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Set NCCL timeout to 30 minutes (1800 seconds) for slow data loading
        # Default is 10 minutes (600 seconds)
        # This is especially important when using many GPUs with large shuffle buffers
        timeout_minutes = 30
        timeout = timedelta(minutes=timeout_minutes)

        dist.init_process_group(
            backend=backend,
            timeout=timeout,
        )
        is_distributed = True


    else:
        rank = dist.get_rank() if is_distributed else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

    # Set device based on local_rank for DDP, or use the passed device for single GPU
    if use_ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    # else: keep the device passed to train() function

    # first set seeds (add rank to seed for different data order on each GPU)
    rank_seed = int(config.train.seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    tf.random.set_seed(rank_seed)
    if config.train.get("rlds_deterministic", False):
        tf.config.experimental.enable_op_determinism()
    if rank == 0:
        print(f"[Seed] Python/NumPy/Torch/TF base={config.train.seed}; rank_offset=rank", flush=True)

    # set num workers
    torch.set_num_threads(1)

    if rank == 0:
        print("\n============= New Training Run with Config =============")
        if use_ddp:
            print(f"[DDP] Running on {world_size} GPUs")

        verbose_config = getattr(config.experiment, 'verbose_config', False)

        if verbose_config:
            print(config)
        else:
            print(f"Experiment: {config.experiment.name}")
            print(f"Algorithm: {config.algo_name}")
            print(f"Output Dir: {config.train.output_dir}")
            print(f"Batch Size: {config.train.batch_size}")
            print(f"Learning Rate: {config.algo.optim_params.policy.learning_rate.initial}")
            if hasattr(config.observation.encoder, 'rgb') and hasattr(config.observation.encoder.rgb, 'core_kwargs'):
                if 'backbone_class' in config.observation.encoder.rgb.core_kwargs:
                    print(f"Visual Encoder: {config.observation.encoder.rgb.core_kwargs.backbone_class}")
            print("(Use --verbose_config to see full configuration)")

        print("")

    # Generate a shared timestamp for all ranks
    if is_distributed:
        # Rank 0 generates timestamp and broadcasts to all ranks
        if rank == 0:
            shared_timestamp = time.time()
        else:
            shared_timestamp = 0.0  # Initialize with a dummy value

        # Broadcast timestamp from rank 0 to all other ranks
        if torch.cuda.is_available():
            timestamp_tensor = torch.tensor([shared_timestamp], dtype=torch.float64).cuda(local_rank)
            dist.broadcast(timestamp_tensor, src=0)
            shared_timestamp = timestamp_tensor.cpu().item()
        else:
            timestamp_tensor = torch.tensor([shared_timestamp], dtype=torch.float64)
            dist.broadcast(timestamp_tensor, src=0)
            shared_timestamp = timestamp_tensor.item()

        # Store shared timestamp for use in get_exp_dir
        import datetime
        time_str = datetime.datetime.fromtimestamp(shared_timestamp).strftime('%Y%m%d%H%M%S')
    else:
        time_str = None

    # Create experiment directories (get_exp_dir now handles DDP synchronization)
    auto_remove = getattr(config.experiment, "auto_remove_exp_dir", False)
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(
        config, auto_remove_exp_dir=auto_remove, shared_time_str=time_str
    )

    # Override checkpoint directory if custom path is specified (only rank 0 creates it)
    if hasattr(config.train, 'checkpoint_dir') and config.train.checkpoint_dir is not None:
        ckpt_dir = config.train.checkpoint_dir
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"✓ Using custom checkpoint directory: {ckpt_dir}")
    else:
        if rank == 0:
            print(f"✓ Using default checkpoint directory: {ckpt_dir}")

    # Synchronize after checkpoint directory creation
    if is_distributed:
        dist.barrier()

    # Only rank 0 redirects stdout to file
    if config.experiment.logging.terminal_output_to_txt and rank == 0:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    ds_format = config.train.data_format
    use_neg_one_one_norm = False

    if ds_format == "real_robot" or ds_format == "zed":
        # 真实机器人数据格式处理 (Real Robot Data: HDF5 + SVO from ZED cameras)
        # 向后兼容：zed 作为 real_robot 的别名
        print("\n" + "=" * 60)
        print("Loading Real Robot Dataset")
        print("=" * 60)

        # 真实机器人数据不需要env_meta，直接设置为None
        env_meta = None
        obs_normalization_stats = None

        # 懒加载 zed_dataset 模块
        from robomimic.utils.zed_dataset import make_zed_dataset

        # 创建真实机器人数据集
        dataset, real_robot_statistics = make_zed_dataset(
            data_directory=config.train.data_path,
            window_size=config.algo.horizon.observation_horizon,
            future_action_window_size=config.algo.horizon.prediction_horizon - 1,
            subsample_length=config.train.subsample_length,
            image_size=tuple(config.observation.image_dim),
            shuffle_buffer_size=config.train.shuffle_buffer_size,
            seed=config.train.seed,
        )

        # 组合数据集统计信息
        real_robot_statistics = combine_dataset_statistics(real_robot_statistics)

        # 检查是否使用CleanDIFT
        use_neg_one_one_norm = False
        if hasattr(config.observation.encoder, 'rgb') and hasattr(config.observation.encoder.rgb, 'core_kwargs'):
            backbone_class = config.observation.encoder.rgb.core_kwargs.get('backbone_class', '')
            if 'CleanDIFT' in backbone_class or 'DIFT' in backbone_class:
                use_neg_one_one_norm = True
                ObsUtils.ImageModality.set_obs_processor(_preprocessed_float_rgb_processor)
                if rank == 0:
                    print(
                        "[DROID] CleanDIFT/DIFT encoder detected: using identity RGB "
                        "processor for RLDS-preprocessed float images.",
                        flush=True,
                    )
            else:
                ObsUtils.ImageModality.set_obs_processor(None)

        has_proprio = len(config.observation.modalities.obs.low_dim) > 0

        # 应用transform
        dataset = dataset.map(
            lambda traj: real_robot_transform(
                traj,
                normalize_to_neg_one_one=use_neg_one_one_norm,
                include_proprio=has_proprio,
            )
        )

        pytorch_dataset = TorchRLDSDataset(dataset)
        train_loader = DataLoader(
            pytorch_dataset,
            batch_size=config.train.batch_size,
            num_workers=0,
        )

        # 预加载第一个batch
        if rank == 0:
            print("\nLoading first batch from ZED dataset...")
            sys.stdout.flush()

        data_loader_iter = iter(train_loader)
        rlds_batch = next(data_loader_iter)

        if rank == 0:
            print("✓ First batch loaded successfully!")
            print("=" * 60 + "\n")
            sys.stdout.flush()

        if use_ddp:
            dist.barrier()

        # 动作统计信息
        action_stats = real_robot_statistics["action"]
        action_config = config.train.action_config
        action_normalization_stats = action_stats_to_normalization_stats(
            {k: action_stats for k in config.train.action_keys},
            action_config
        )

        # 获取shape metadata
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=None,
            batch=rlds_batch,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True,
            config=config
        )

    elif ds_format == "droid_rlds":
        # # load basic metadata from training file
        # print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=None, ds_format=ds_format)
        obs_normalization_stats = None

        obs_modalities = config.observation.modalities.obs.rgb
        # NOTE: Must be at least 2 cam for now, can clean this up later.
        if len(obs_modalities) < 2 or len(obs_modalities) > len(DROID_RLDS_IMAGE_SLOTS):
            raise ValueError(
                f"DROID RLDS training expects 2-{len(DROID_RLDS_IMAGE_SLOTS)} cameras, "
                f"got {len(obs_modalities)}: {obs_modalities}"
            )

        # 验证相机key是否在映射表中
        for cam_key in obs_modalities:
            if cam_key not in DROID_TO_RLDS_OBS_KEY_MAP:
                available_keys = list(DROID_TO_RLDS_OBS_KEY_MAP.keys())
                raise KeyError(
                    f"Camera key '{cam_key}' not found in DROID_TO_RLDS_OBS_KEY_MAP.\n"
                    f"Available keys: {available_keys}\n"
                    f"Your config has: {obs_modalities}\n"
                    f"Hint: Use --cameras to specify correct camera names, e.g.:\n"
                    f"  For DROID: --cameras hand_camera_left varied_camera_1_left\n"
                    f"  For real robot: --cameras wrist_left varied_camera_1_left"
                )

        ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
        action_config = config.train.action_config
        is_abs_action = [True] * ac_dim

        image_slots = list(DROID_RLDS_IMAGE_SLOTS[:len(obs_modalities)])
        image_obs_keys = {
            slot: DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[i]]
            for i, slot in enumerate(image_slots)
        }
        transformed_image_keys = [f"image_{slot}" for slot in image_slots]

        if rank == 0:
            print("\nDROID RLDS camera mapping:")
            for out_key, slot, source_key, transformed_key in zip(
                obs_modalities, image_slots, image_obs_keys.values(), transformed_image_keys
            ):
                print(f"  {out_key} <= RLDS '{source_key}' as '{transformed_key}'")

        BASE_DATASET_KWARGS = {
                "data_dir": config.train.data_path,
                "image_obs_keys": image_obs_keys,
                "state_obs_keys": [DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[obs_key] for obs_key in config.observation.modalities.obs.low_dim],
                "language_key": "language_instruction",
                "norm_skip_keys":  ["proprio"],
                "action_proprio_normalization_type": "bounds",
                "absolute_action_mask": is_abs_action,
                "action_normalization_mask": is_abs_action,
                "standardize_fn": droid_dataset_transform,
         }

        dataset_names = config.train.dataset_names
        filter_functions = [[ModuleSpec.create(
                                "robomimic.utils.rlds_utils:filter_success"
                                )] if d_name == "droid" else [] \
                            for d_name in dataset_names]
        dataset_kwargs_list = [
            {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS} for d_name, f_functions in zip(dataset_names, filter_functions)
        ]
        # Compute combined normalization stats
        per_dataset_statistics = [
            make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list
        ]
        combined_dataset_statistics = combine_dataset_statistics(per_dataset_statistics)
        if rank == 0:
            # Record exactly which data this run saw: the TFDS version and the
            # post-filter trajectory/transition counts differ between copies.
            import tensorflow_datasets as tfds

            for dataset_kwargs, statistics in zip(dataset_kwargs_list, per_dataset_statistics):
                version = tfds.builder(dataset_kwargs["name"], data_dir=dataset_kwargs["data_dir"]).info.version
                print(
                    f"[DROID] dataset={dataset_kwargs['name']} version={version} "
                    f"data_dir={dataset_kwargs['data_dir']} "
                    f"trajectories={statistics.get('num_trajectories')} "
                    f"transitions={statistics.get('num_transitions')}",
                    flush=True,
                )

        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            config.train.sample_weights,
            train=True,
            shuffle_buffer_size=config.train.shuffle_buffer_size,
            batch_size=None,  # batching will be handled in PyTorch Dataloader object
            balance_weights=False,
            dataset_statistics=combined_dataset_statistics,
            traj_transform_kwargs=dict(
                # NOTE(Ashwin): window_size and future_action_window_size may break if
                # not using diffusion policy
                window_size=config.algo.horizon.observation_horizon,
                future_action_window_size=config.algo.horizon.prediction_horizon-1,
                subsample_length=config.train.subsample_length,
                skip_unlabeled=True,    # skip all trajectories without language
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs=dict(
                ),
                resize_size={slot: config.observation.image_dim for slot in image_slots},
                num_parallel_calls=config.train.num_parallel_calls,
            ),
            traj_transform_threads=config.train.traj_transform_threads,
            traj_read_threads=config.train.traj_read_threads,
        )
        # Note: If we have separated statistics for multiple datasets, use the first one (assumed to be DROID)
        # Otherwise, use the combined dataset statistics.
        rlds_dataset_stats = dataset.dataset_statistics[0] if isinstance(dataset.dataset_statistics, list) else dataset.dataset_statistics
        action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats["action"], config.train.action_keys, config.train.action_shapes)
        action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)

        # 计算数据集的实际长度（用于正确的epoch计数）
        # 仅在明确启用传统epoch模式时计算（通过环境变量或配置标志）
        dataset_length = None
        compute_exact_length = os.environ.get("COMPUTE_DATASET_LENGTH", "false").lower() == "true"

        if compute_exact_length:
            if rank == 0:
                print("\n" + "="*60)
                print("Computing exact dataset length for traditional epoch training...")
                print("="*60)
                dataset_length = compute_rlds_dataset_length(config.train.data_path, config.train.dataset_names)
                if dataset_length:
                    print(f"✓ Total dataset length: {dataset_length} steps")
                    print(f"  With batch_size={config.train.batch_size}: {dataset_length//config.train.batch_size} batches per epoch")

                    # 检查shuffle_buffer_size是否足够大
                    if config.train.shuffle_buffer_size < dataset_length:
                        print(f"\n⚠️  WARNING: shuffle_buffer_size ({config.train.shuffle_buffer_size}) < dataset_length ({dataset_length})")
                        print(f"   This means each epoch will NOT see all data!")
                        print(f"   Recommendation: Set shuffle_buffer_size >= {dataset_length} for true epoch training")
                    else:
                        print(f"✓ shuffle_buffer_size ({config.train.shuffle_buffer_size}) >= dataset_length")
                        print(f"  Good! Each epoch will see all data.")
                    print("="*60 + "\n")
                else:
                    print("⚠️  Could not compute dataset length")
                    print("="*60 + "\n")
                    dataset_length = 0  # 设为0表示未知

            # Broadcast dataset_length to all ranks
            if use_ddp:
                dataset_length_tensor = torch.tensor(dataset_length if dataset_length else 0, dtype=torch.long)
                if torch.cuda.is_available():
                    dataset_length_tensor = dataset_length_tensor.cuda()
                dist.broadcast(dataset_length_tensor, src=0)
                dataset_length = int(dataset_length_tensor.item()) if dataset_length_tensor.item() > 0 else None

        # Check if using CleanDIFT or other diffusion-based encoders that need [-1, 1] normalization
        use_neg_one_one_norm = False
        if hasattr(config.observation.encoder, 'rgb') and hasattr(config.observation.encoder.rgb, 'core_kwargs'):
            backbone_class = str(config.observation.encoder.rgb.core_kwargs.get('backbone_class', ''))
            if 'CleanDIFT' in backbone_class or 'DIFT' in backbone_class:
                use_neg_one_one_norm = True
                ObsUtils.ImageModality.set_obs_processor(_preprocessed_float_rgb_processor)
                if rank == 0:
                    print(
                        "[DROID] CleanDIFT/DIFT encoder detected: using identity RGB "
                        "processor for RLDS-preprocessed float images.",
                        flush=True,
                    )
            else:
                ObsUtils.ImageModality.set_obs_processor(None)

        # Check if we have low_dim observations (proprio)
        has_proprio = len(config.observation.modalities.obs.low_dim) > 0
        view_dropout_prob = float(getattr(config.train, "view_dropout_prob", 0.0) or 0.0)

        # Apply appropriate normalization based on encoder type
        dataset = dataset.map(
            lambda traj: robomimic_transform(
                traj,
                normalize_to_neg_one_one=use_neg_one_one_norm,
                include_proprio=has_proprio,
                view_dropout_prob=view_dropout_prob,
                obs_camera_keys=list(obs_modalities),
                rlds_image_keys=transformed_image_keys,
            ),
            num_parallel_calls=config.train.traj_transform_threads
        )

        pytorch_dataset = TorchRLDSDataset(
            dataset,
            shuffle_buffer_size=config.train.shuffle_buffer_size,
            dataset_length=dataset_length  # 传递实际的数据集长度
        )
        train_loader = DataLoader(
            pytorch_dataset,
            batch_size=config.train.batch_size,
            num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        )

        # Pre-load first batch to ensure all ranks have data ready before training starts
        # This prevents NCCL timeout due to uneven data loading speeds across ranks
        if rank == 0:
            print("\n" + "=" * 60)
            print("RLDS Data Loading")
            print("=" * 60)
            print(f"Shuffle buffer size: {config.train.shuffle_buffer_size}")
            print("TensorFlow is filling the shuffle buffer...")
            print("This may take 15-30 minutes depending on buffer size.")
            print("(The progress bar below is from TF data pipeline)")
            print("=" * 60 + "\n")
            sys.stdout.flush()

        data_loader_iter = iter(train_loader)
        rlds_batch = next(data_loader_iter)

        # 保存固定的可视化batch（用于每个epoch的轨迹可视化）
        if rank == 0:
            print("\n" + "=" * 60)
            print("Saving fixed visualization batch...")
            print("=" * 60)

            # 递归深拷贝函数
            def deep_copy_batch(obj, max_samples=4):
                """递归复制batch数据，只保留前max_samples个样本"""
                if isinstance(obj, dict):
                    return {k: deep_copy_batch(v, max_samples) for k, v in obj.items()}
                elif torch.is_tensor(obj):
                    # 克隆tensor并只保留前N个样本
                    cloned = obj.clone()
                    if len(cloned) > max_samples:
                        cloned = cloned[:max_samples]
                    return cloned
                else:
                    return obj

            # 深拷贝一份用于可视化
            vis_batch = deep_copy_batch(rlds_batch, max_samples=4)

            # 检查保存的数据
            num_samples = vis_batch['actions'].shape[0] if torch.is_tensor(vis_batch.get('actions')) else 0
            num_cameras = len(vis_batch.get('obs', {})) if isinstance(vis_batch.get('obs'), dict) else 0

            print(f"✓ Saved visualization batch:")
            print(f"  - Samples: {num_samples}")
            print(f"  - Cameras: {num_cameras}")
            if isinstance(vis_batch.get('obs'), dict):
                for cam_name in vis_batch['obs'].keys():
                    if torch.is_tensor(vis_batch['obs'][cam_name]):
                        print(f"  - {cam_name}: {vis_batch['obs'][cam_name].shape}")
            print("  Will visualize predictions at each epoch")
            print("=" * 60 + "\n")
            sys.stdout.flush()
        else:
            vis_batch = None

        if rank == 0:
            print("\n" + "=" * 60)
            print("✓ First batch loaded successfully!")
            print("=" * 60 + "\n")
            sys.stdout.flush()

        # Synchronize all ranks after first batch is loaded
        if use_ddp:
            dist.barrier()

        # For RLDS, get batch from train loader to compute shapes
        # (already loaded above)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=None,
            batch=rlds_batch,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True,
            config = config
        )

        # RLDS format does not have validation loader
        valid_loader = None

    else:
        # make sure the dataset exists
        eval_dataset_cfg = config.train.data[0]
        dataset_path = os.path.expanduser(eval_dataset_cfg["path"])

        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)


        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            batch=None,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True,
            config = config
        )
        # load training data
        trainset, validset = TrainUtils.load_data_for_training(
            config, obs_keys=shape_meta["all_obs_keys"])

        # Use DistributedSampler for DDP
        if use_ddp:
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_sampler = trainset.get_dataset_sampler()

        # # maybe retreve statistics for normalizing observations
        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()

        # maybe retreve statistics for normalizing actions
        action_normalization_stats = trainset.get_action_normalization_stats()

        # initialize data loaders
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )

    # determine number of steps per epoch (use full loader when not explicitly set)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    if train_num_steps is None:
        train_num_steps = len(train_loader)
    if valid_loader is not None and valid_num_steps is None:
        valid_num_steps = len(valid_loader)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs (typically only rank 0 does rollouts)
        if rank == 0:
            env_names = [env_meta["env_name"]]

            if config.experiment.additional_envs is not None:
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            for env_name in env_names:
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                )
                env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
                envs[env.name] = env

    # setup for a new training run - only rank 0 creates real loggers
    if rank == 0:
        data_logger = DataLogger(
            log_dir,
            config,
            log_tb=config.experiment.logging.log_tb,
            log_wandb=config.experiment.logging.log_wandb,
        )
    else:
        # Other ranks use dummy logger
        data_logger = DummyDataLogger()

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file (only rank 0)
    if rank == 0:
        with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
            json.dump(config, outfile, indent=4)

    # if checkpoint is specified, load in model weights
    ckpt_path = config.experiment.ckpt_path
    # Load checkpoint if resuming training
    resume_epoch = 0
    resume_best_valid_loss = None
    if ckpt_path is not None:
        reset_optimizer_on_resume = bool(
            getattr(config.experiment, "reset_optimizer_on_resume", False)
        )
        if rank == 0:
            print(f"Loading checkpoint from: {ckpt_path}")
            if reset_optimizer_on_resume:
                print("↻ Resetting optimizer/scheduler states after loading model weights")
            sys.stdout.flush()
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)

        # Load model weights
        model.deserialize(ckpt_dict["model"])
        if rank == 0:
            print("✓ Loaded model weights")
            sys.stdout.flush()

        # Load optimizer states for resume training. Some staged finetunes
        # intentionally change trainable parameter groups between stages.
        if reset_optimizer_on_resume:
            if rank == 0:
                print("↻ Skipping optimizer states - starting with fresh optimizer")
        elif "optimizer_states" in ckpt_dict:
            model.load_optimizer_states(ckpt_dict["optimizer_states"])
            if rank == 0:
                print(f"✓ Loaded optimizer states: {list(ckpt_dict['optimizer_states'].keys())}")
        else:
            if rank == 0:
                print("⚠ No optimizer states found - starting with fresh optimizer")

        # Load scheduler states for resume training
        if reset_optimizer_on_resume:
            if rank == 0:
                print("↻ Skipping scheduler states - starting with fresh scheduler")
        elif "scheduler_states" in ckpt_dict:
            model.load_scheduler_states(ckpt_dict["scheduler_states"])
            if rank == 0:
                print(f"✓ Loaded scheduler states: {list(ckpt_dict['scheduler_states'].keys())}")
        else:
            if rank == 0:
                print("⚠ No scheduler states found - starting with fresh scheduler")

        if (
            not reset_optimizer_on_resume
            and "amp_scaler_state" in ckpt_dict
            and hasattr(model, "grad_scaler")
            and model.grad_scaler is not None
        ):
            model.grad_scaler.load_state_dict(ckpt_dict["amp_scaler_state"])
            if rank == 0:
                print("✓ Loaded AMP grad scaler state")

        # Load training state
        if "epoch" in ckpt_dict:
            resume_epoch = ckpt_dict["epoch"]
            if rank == 0:
                print(f"✓ Resume from epoch: {resume_epoch}")

        if "best_valid_loss" in ckpt_dict:
            resume_best_valid_loss = ckpt_dict["best_valid_loss"]
            if rank == 0:
                print(f"✓ Best validation loss: {resume_best_valid_loss}")

        # Keep the configured freeze state of a legacy encoder after loading weights.
        from robomimic.models.cleandift_backbone import CleanDIFTConv
        for module in model.nets.modules():
            if isinstance(module, CleanDIFTConv):
                module._sync_freeze_state()

    # print("\n============= Model Summary =============")
    # print(model)  # print model summary - commented out to reduce verbosity
    # print("")

    # Algo calls individual children of model.nets rather than its forward.
    # A DDP wrapper around the ModuleDict would therefore never see a forward
    # and cannot reliably reduce gradients. Broadcast once here; the policy
    # averages actual optimizer gradients after local accumulation.
    if use_ddp:
        broadcast_training_state(model.nets, getattr(model, "ema", None))
        if rank == 0:
            print("[DDP] Using explicit optimizer-gradient synchronization", flush=True)

    ##### ------------------------------------------------------------------------------------ ######

    # TODO(Ashwin): Support loading validation splits for RLDS
    if ds_format != "droid_rlds" and config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins (only rank 0)
    if rank == 0:
        flush_warnings()

    feature_vis_state = None
    if rank == 0:
        feature_vis_state = _prepare_feature_vis_state(config, use_neg_one_one_norm, device)
        if feature_vis_state is not None:
            initial_epoch = resume_epoch if resume_epoch > 0 else 0
            _log_feature_maps(model, feature_vis_state, data_logger, epoch=initial_epoch, device=device)

    # main training loop
    # Initialize training state
    if resume_best_valid_loss is not None:
        best_valid_loss = resume_best_valid_loss
    else:
        best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch already set above (defaulting to full loader)
    data_loader_iter = iter(train_loader)

    # Resume from checkpoint epoch if available
    start_epoch = resume_epoch + 1 if resume_epoch > 0 else 1
    if rank == 0 and resume_epoch > 0:
        print(f"Resuming training from epoch {start_epoch} (previous {resume_epoch}, best val {best_valid_loss})")

    # Print training start confirmation
    if rank == 0:
        print("\n" + "=" * 60)
        print("🚀 TRAINING STARTING")
        print("=" * 60)
        print(f"  Total epochs: {config.train.num_epochs}")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Steps per epoch: {train_num_steps}")
        accumulation = int(getattr(config.train, "gradient_accumulation_steps", 1) or 1)
        print(f"  Microbatch size per GPU: {config.train.batch_size}")
        print(f"  Gradient accumulation steps: {accumulation}")
        print(f"  Effective global batch size: {config.train.batch_size * world_size * accumulation}")
        print("=" * 60 + "\n")
        sys.stdout.flush()

    if rank == 0:
        initial_progress = max(start_epoch - 1, 0)
        total_epochs = max(config.train.num_epochs, 0)
        progress_bar = tqdm.tqdm(
            range(start_epoch, config.train.num_epochs + 1),
            total=total_epochs,
            initial=min(initial_progress, total_epochs),
            desc="Training",
            dynamic_ncols=True,
            file=sys.stdout,  # Force output to stdout
            position=0,       # Keep at position 0
            leave=True,       # Keep the progress bar after completion
        )
        epoch_iterable = progress_bar
    else:
        progress_bar = None
        epoch_iterable = range(start_epoch, config.train.num_epochs + 1)

    for epoch in epoch_iterable:  # epoch numbers start at 1
        if progress_bar is not None:
            progress_bar.set_description(f"Epoch {epoch}")
            postfix_metrics = OrderedDict()
        else:
            postfix_metrics = None

        if rank == 0 and epoch == 1:
            print(f"\n[Train] Epoch {epoch}: Starting run_epoch...", flush=True)

        # model.nets might be wrapped with DDP, but model itself is always Algo object
        # run_epoch expects Algo object, so just pass model as is
        train_step_log, data_loader_iter = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
            data_loader_iter=data_loader_iter,
            is_ddp=use_ddp,
            rank=rank,
        )

        if rank == 0 and epoch == 1:
            print(f"[Train] Epoch {epoch}: run_epoch completed!", flush=True)

        model.on_epoch_end(epoch)

        # ========================================
        # 轨迹可视化（每个epoch）
        # ========================================
        enable_vis = os.environ.get("ENABLE_TRAJECTORY_VIS", "false").lower() == "true"
        # 自适应可视化频率：前10个epoch每次都可视化，之后每10个epoch可视化一次
        vis_freq = 1 if epoch <= 10 else max(1, config.train.num_epochs // 100)

        if rank == 0 and vis_batch is not None and enable_vis and epoch % vis_freq == 0:
            try:
                print(f"\n[Visualization] Generating trajectory visualization for epoch {epoch}...")
                # 设置评估模式（兼容DDP包装的model）
                model.set_eval()

                # 重要：重置模型状态（初始化action_queue等推理所需的状态）
                model.reset()

                with torch.no_grad():
                    # 准备输入数据（深拷贝避免修改原始vis_batch）
                    def recursive_copy(obj):
                        """递归复制dict或tensor"""
                        if isinstance(obj, dict):
                            return {k: recursive_copy(v) for k, v in obj.items()}
                        elif torch.is_tensor(obj):
                            return obj.clone()
                        else:
                            return obj

                    vis_obs = recursive_copy(vis_batch['obs'])
                    vis_gt_actions = vis_batch['actions'].clone() if torch.is_tensor(vis_batch['actions']) else vis_batch['actions']

                    # 预测动作
                    vis_input = {'obs': vis_obs}
                    vis_output = model.get_action(vis_input)
                    vis_pred_actions = vis_output if torch.is_tensor(vis_output) else vis_output['actions']

                    # 获取相机图像
                    camera_images = {}
                    obs_modalities = config.observation.modalities.obs.rgb
                    for cam_name in obs_modalities:
                        if cam_name in vis_obs:
                            camera_images[cam_name] = vis_obs[cam_name]

                    # 生成可视化
                    vis_images = visualize_trajectories(
                        images=camera_images,
                        gt_actions=vis_gt_actions,
                        pred_actions=vis_pred_actions,
                        camera_names=obs_modalities,
                        epoch=epoch,
                        config=config
                    )

                    # 上传到wandb
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"trajectory_visualization": vis_images}, step=epoch)
                        print(f"✓ Uploaded {len(vis_images)} trajectory visualizations to wandb (epoch {epoch})")

                # 恢复训练模式
                model.set_train()
            except Exception as e:
                print(f"\n⚠️  Warning: Trajectory visualization failed at epoch {epoch}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {e}")
                print(f"   Model type: {type(model)}")
                print(f"   Has nets: {hasattr(model, 'nets')}")
                if hasattr(model, 'nets'):
                    print(f"   Nets type: {type(model.nets)}")
                print(f"   Has set_eval: {hasattr(model, 'set_eval')}")
                print(f"   Has get_action: {hasattr(model, 'get_action')}")
                import traceback
                traceback.print_exc()
                # 确保模型回到训练模式
                try:
                    model.set_train()
                except:
                    pass

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        # Only rank 0 prints and logs
        if rank == 0:
            for k, v in train_step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)
        if postfix_metrics is not None:
            postfix_metrics.update(_build_progress_postfix(train_step_log, prefix="train"))

        # Evaluate the model on validation set
        if config.experiment.validate:
            # model itself is Algo object, just pass it directly
            with torch.no_grad():
                valid_step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    is_ddp=use_ddp,
                    rank=rank,
                )

            # Only rank 0 logs and prints
            if rank == 0:
                for k, v in valid_step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                    else:
                        data_logger.record("Valid/{}".format(k), v, epoch)
            if postfix_metrics is not None:
                postfix_metrics.update(_build_progress_postfix(valid_step_log, prefix="val"))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in valid_step_log
            if valid_check and (best_valid_loss is None or (valid_step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = valid_step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts (only rank 0)

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if rank == 0 and config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            # model itself is Algo object, just pass it directly
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # check if we need to save model MSE (only rank 0)
        #TODO(Ashwin): support MSE Logging with RLDS dataloading
        if rank == 0 and ds_format != "droid_rlds":
            should_save_mse = False
            if config.experiment.mse.enabled:
                if config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0:
                    should_save_mse = True
                if config.experiment.mse.on_save_ckpt and should_save_ckpt:
                    should_save_mse = True
            if should_save_mse:
                if config.experiment.mse.visualize:
                    save_vis_dir = os.path.join(vis_dir, epoch_ckpt_name)
                else:
                    save_vis_dir = None
                # model itself is Algo object, just pass it directly
                mse_log, vis_log = model.compute_mse_visualize(
                    trainset,
                    validset,
                    num_samples=config.experiment.mse.num_samples,
                    savedir=save_vis_dir,
                )
                for k, v in mse_log.items():
                    data_logger.record("{}".format(k), v, epoch)

                for k, v in vis_log.items():
                    data_logger.record("{}".format(k), v, epoch, data_type='image')


        elif rank == 0 and ds_format == "droid_rlds":
            should_save_batch_samples = False
            # TODO(Ashwin): eventually clean up to use different config parameters for
            # batch visualization vs. mse visualization
            if config.experiment.mse.enabled:
                if config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0:
                    should_save_batch_samples = True
                if config.experiment.mse.on_save_ckpt and should_save_ckpt:
                    should_save_batch_samples = True
            if should_save_batch_samples:
                if config.experiment.mse.visualize:
                    save_vis_dir = os.path.join(vis_dir, epoch_ckpt_name)
                else:
                    save_vis_dir = None
                # model itself is Algo object, just pass it directly
                vis_log = model.compute_batch_visualize(
                    batch=rlds_batch,
                    num_samples=config.experiment.mse.num_samples,
                    savedir=save_vis_dir,
                )

                for k, v in vis_log.items():
                    data_logger.record("{}".format(k), v, epoch, data_type='image')

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        # Only rank 0 handles video files (since only rank 0 creates them)
        if rank == 0:
            should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
            if video_paths is not None and not should_save_video:
                for env_name in video_paths:
                    os.remove(video_paths[env_name])

        if rank == 0 and feature_vis_state is not None:
            freq = feature_vis_state.get("frequency")
            last_epoch = feature_vis_state.get("last_epoch")
            feature_vis_due = False
            if should_save_ckpt and (last_epoch is None or epoch != last_epoch):
                feature_vis_due = True
            elif freq is not None and freq > 0:
                if epoch % freq == 0 and (last_epoch is None or epoch != last_epoch):
                    feature_vis_due = True
            if feature_vis_due:
                _log_feature_maps(model, feature_vis_state, data_logger, epoch=epoch, device=device)

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        # Only rank 0 saves checkpoints
        if should_save_ckpt and rank == 0:
            def _trace_checkpoint_memory(phase):
                if os.environ.get("ROBOT_DIFT_CHECKPOINT_MEMORY_TRACE") != "1" or not torch.cuda.is_available():
                    return
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[Checkpoint memory] phase={phase} epoch={epoch} "
                    f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.2f} "
                    f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.2f} "
                    f"reserved_gib={torch.cuda.memory_reserved() / 2**30:.2f} "
                    f"free_gib={free / 2**30:.2f} total_gib={total / 2**30:.2f}",
                    flush=True,
                )

            _trace_checkpoint_memory("before_policy_save")
            # Note: model is Algo object, model.nets might be wrapped with DDP
            # save_model will call model.serialize() which handles DDP unwrapping internally
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                optimizers=model.get_optimizer_states(),
                lr_schedulers=model.get_scheduler_states(),
            )
            _trace_checkpoint_memory("after_policy_save")

            # Also export the Stage-I Student (and its readout) for transfer.
            save_cleandift_dir = getattr(config.experiment, 'save_cleandift_dir', None)
            if save_cleandift_dir:
                require_encoder_export = bool(getattr(config.experiment, "require_encoder_export", False))
                try:
                    for saved_dir in _export_robot_dift_encoder(
                        model,
                        config,
                        epoch,
                        save_cleandift_dir,
                        save_ema=bool(getattr(config.experiment, "save_cleandift_ema", False)),
                        trace=_trace_checkpoint_memory,
                    ):
                        print(f"Saved Robot-DIFT encoder checkpoint to: {saved_dir}")
                except Exception as e:
                    if require_encoder_export:
                        raise RuntimeError(
                            f"Required Robot-DIFT encoder export failed at epoch {epoch} "
                            f"under {save_cleandift_dir}"
                        ) from e
                    print(f"Failed to save encoder: {e}")

        # Synchronize all processes before continuing
        if is_distributed:
            dist.barrier()

        if torch.cuda.is_available() and os.environ.get("ROBOT_DIFT_LOG_RANK_GPU_MEMORY") == "1":
            torch.cuda.synchronize(device)
            gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info(device)
            print(
                f"[GPU memory] epoch={epoch} rank={rank} "
                f"allocated_gib={torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} "
                f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} "
                f"reserved_gib={torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} "
                f"peak_reserved_gib={torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} "
                f"free_gib={gpu_free_bytes / (1024 ** 3):.2f} "
                f"total_gib={gpu_total_bytes / (1024 ** 3):.2f}",
                flush=True,
            )

        # Finally, log memory usage in MB (only rank 0)
        if rank == 0:
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            if postfix_metrics is not None:
                postfix_metrics["mem_mb"] = f"{mem_usage}"
            if torch.cuda.is_available():
                # Lifetime peaks expose whether a larger per-rank microbatch
                # fits with enough headroom for checkpoint export and rollout.
                gpu_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                gpu_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                data_logger.record("System/GPU Peak Allocated (GiB)", gpu_peak_gib, epoch)
                data_logger.record("System/GPU Peak Reserved (GiB)", gpu_reserved_gib, epoch)
                if postfix_metrics is not None:
                    postfix_metrics["gpu_peak_gib"] = f"{gpu_peak_gib:.1f}"

        if progress_bar is not None and postfix_metrics is not None:
            progress_bar.set_postfix(postfix_metrics, refresh=False)

    # Final evaluation rollouts after training completes (only rank 0)
    if rank == 0 and config.experiment.rollout.enabled and len(envs) > 0:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ROLLOUTS")
        print("=" * 80)

        rollout_model = RolloutPolicy(
            model,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        final_epoch_label = "final"
        all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
            policy=rollout_model,
            envs=envs,
            horizon=config.experiment.rollout.horizon,
            use_goals=config.use_goals,
            num_episodes=config.experiment.rollout.n,
            render=False,
            video_dir=video_dir if config.experiment.render_video else None,
            epoch=final_epoch_label,
            video_skip=config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
        )

        for env_name, rollout_logs in all_rollout_logs.items():
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/FinalRollout_{env_name}_{k[5:]}", v, config.train.num_epochs)
                else:
                    data_logger.record(f"FinalRollout/{k}/{env_name}", v, config.train.num_epochs, log_stats=True)

            print(f"\nFinal Rollout Results ({env_name}):")
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    # terminate logging (only rank 0)
    if rank == 0:
        data_logger.close()
        progress_bar.close()

    # Clean up DDP
    if is_distributed:
        dist.destroy_process_group()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if config.train.data_format != "droid_rlds" and args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 200
        config.experiment.mse.every_n_epochs = 2
        config.experiment.save.every_n_epochs = 1

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)
