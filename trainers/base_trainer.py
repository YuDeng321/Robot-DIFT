"""Shared utilities for all main scripts.

This trainer supports both single-GPU and multi-GPU (DDP) training.
"""

import os
import inspect
import pickle
import random
import logging
import wandb
from contextlib import nullcontext
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from agents.utils.scaler import Scaler, ActionScaler, MinMaxScaler
from agents.utils.ema import ExponentialMovingAverage
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


def optimizer_updates_per_epoch(num_batches: int, accumulation_steps: int, full_groups_only: bool) -> int:
    """Count optimizer updates, optionally dropping an incomplete final group."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    usable = num_batches - num_batches % accumulation_steps if full_groups_only else num_batches
    if usable < 1:
        raise ValueError("No full optimizer update fits in one training epoch")
    return (usable + accumulation_steps - 1) // accumulation_steps


def make_linear_lr_scheduler(optimizer, total_updates: int, end_factor: float):
    """Linearly decay each optimizer LR after updates to a stated endpoint."""
    if total_updates < 1 or not 0.0 <= end_factor <= 1.0:
        raise ValueError("linear LR requires positive updates and end_factor in [0, 1]")
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: 1.0 - (1.0 - end_factor) * min(update, total_updates) / total_updates,
    )

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("All items in batch are None.")
    return default_collate(batch)

class BaseTrainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size: int = 512,
            val_batch_size: int = 512,
            num_workers: int = 8,
            device: str = 'cpu',
            epoch: int = 100,
            scale_data: bool = True,
            scaler_type: str = None,
            eval_every_n_epochs: int = 50,
            sim_eval_every_n_epochs: int = 0,
            simulation_cfg: DictConfig = None,
            obs_seq_len: int = 1,
            decay_ema: float = 0.999,
            ema_power: Optional[float] = None,
            if_use_ema: bool = False,
            use_ddp: bool = True,
            scale_lr_by_world_size: bool = False,
            use_sync_bn: bool = False,
            alignment_cfg: DictConfig | None = None,
            use_amp: bool = True,
            gradient_accumulation_steps: int = 1,
            accumulate_steps: Optional[int] = None,
            full_accumulation_batches: bool = False,
            max_train_batches_per_epoch: Optional[int] = None,
            lr_scheduler_type: Optional[str] = None,
            linear_lr_end_factor: float = 0.1,
            # Visualization settings
            viz_interval: int = 10,
            num_viz_samples: int = 5,
    ):
        """Initialize."""

        # DDP env
        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_dist else 0

        self.use_ddp = use_ddp and self.is_dist
        self.scale_lr_by_world_size = scale_lr_by_world_size
        self.use_sync_bn = use_sync_bn

        # Device
        if torch.cuda.is_available() and (device.startswith('cuda')):
            self.device = torch.device(f"cuda:{self.local_rank}" if self.use_ddp else 'cuda')
        else:
            self.device = torch.device('cpu')

        self.use_amp = bool(use_amp and torch.cuda.is_available() and self.device.type == 'cuda')
        self._amp_dtype = torch.bfloat16
        grad_scaler_enabled = self.use_amp and self._amp_dtype == torch.float16
        self.grad_scaler = GradScaler("cuda", enabled=grad_scaler_enabled)
        if accumulate_steps is not None:
            if gradient_accumulation_steps not in (1, int(accumulate_steps)):
                raise ValueError("accumulate_steps and gradient_accumulation_steps disagree")
            gradient_accumulation_steps = int(accumulate_steps)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps or 1))
        self.accumulate_steps = self.gradient_accumulation_steps  # legacy constructor attribute
        self.full_accumulation_batches = bool(full_accumulation_batches)
        if max_train_batches_per_epoch is not None and int(max_train_batches_per_epoch) < 1:
            raise ValueError("max_train_batches_per_epoch must be positive")
        self.max_train_batches_per_epoch = (
            None if max_train_batches_per_epoch is None else int(max_train_batches_per_epoch)
        )
        self.lr_scheduler_type = lr_scheduler_type
        self.linear_lr_end_factor = float(linear_lr_end_factor)

        # Datasets and loaders
        self.trainset = hydra.utils.instantiate(trainset)
        # self.valset = hydra.utils.instantiate(valset)

        if self.is_dist:
            train_sampler = DistributedSampler(
                self.trainset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_sampler = None

        # IMPORTANT: DataLoader.num_workers is per-process already; don't divide by world_size
        # Keep the user-provided workers per rank to avoid dataloader bottlenecks when scaling GPUs.
        # If you need to cap total CPU usage, tune this value in the config directly.
        # num_workers stays as-is per rank.
        num_workers = int(num_workers)

        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
            collate_fn=safe_collate,
            drop_last=True,
        )

        # Meta
        self.obs_seq_len = obs_seq_len
        self.eval_every_n_epochs = eval_every_n_epochs
        self.sim_eval_every_n_epochs = int(sim_eval_every_n_epochs) if sim_eval_every_n_epochs else 0
        self.simulation_cfg = simulation_cfg
        self._simulation = None
        self.epoch = epoch
        self.working_dir = os.getcwd()
        self.scaler_type = scaler_type
        self.decay_ema = decay_ema
        self.ema_power = ema_power
        self.if_use_ema = if_use_ema
        self.alignment_cfg = OmegaConf.to_container(alignment_cfg, resolve=True) if alignment_cfg is not None else None
        self._alignment_weight = 0.0
        self._alignment_freq = 0.0

        # Visualization settings
        self.viz_interval = viz_interval
        self.num_viz_samples = num_viz_samples
        self._visualizer = None

        # Scaler (use trainer device)
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), scale_data, str(self.device))
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), scale_data, str(self.device))

        if (not self.is_dist) or self.rank == 0:
            log.info("Number of training samples: {}".format(len(self.trainset)))
            effective_batch = train_batch_size * self.world_size * self.gradient_accumulation_steps
            log.info(
                "Train batch per rank=%s, world_size=%s, gradient_accumulation_steps=%s, effective batch=%s",
                train_batch_size,
                self.world_size,
                self.gradient_accumulation_steps,
                effective_batch,
            )

    def _init_visualizer(self, agent):
        """Initialize visualizer for real robot training"""
        if self._visualizer is not None:
            return

        # Only initialize on rank 0
        if self.is_dist and self.rank != 0:
            return

        try:
            from trainers.visualization_utils import RealRobotVisualizer

            # Get action dim from agent or trainset
            action_dim = getattr(agent, 'action_dim', 7)
            if hasattr(self.trainset, 'action_dim'):
                action_dim = self.trainset.action_dim

            # Get camera IDs if available
            camera_ids = getattr(self.trainset, 'camera_ids', None)

            self._visualizer = RealRobotVisualizer(
                num_fixed_samples=self.num_viz_samples,
                action_dim=action_dim,
                device=str(self.device),
                camera_ids=camera_ids,
            )

            # Initialize with training samples
            self._visualizer.initialize_fixed_samples(self.trainset)

            log.info(f"✓ Visualizer initialized (interval: {self.viz_interval} epochs)")
        except Exception as e:
            log.warning(f"Failed to initialize visualizer: {e}")
            self._visualizer = None

    def _run_visualization(self, agent, num_epoch: int):
        """Run visualization at specified intervals"""
        if self._visualizer is None:
            return

        # Only run on rank 0
        if self.is_dist and self.rank != 0:
            return

        # Check if we should visualize
        if (num_epoch + 1) % self.viz_interval != 0 and num_epoch != 0:
            return

        try:
            # Unwrap DDP if needed
            model = agent.module if isinstance(agent, DDP) else agent

            # Use EMA model if available
            if self.if_use_ema:
                self.ema_helper.store(model.parameters())
                self.ema_helper.copy_to(model.parameters())

            log.info(f"Running visualization at epoch {num_epoch + 1}...")
            self._visualizer.log_fixed_predictions(
                model,
                num_epoch + 1,
                scaler=self.scaler
            )

            # Restore original weights if EMA was used
            if self.if_use_ema:
                self.ema_helper.restore(model.parameters())

        except Exception as e:
            log.warning(f"Visualization failed at epoch {num_epoch + 1}: {e}")
            import traceback
            traceback.print_exc()

    def main(self, agent):
        """Run main training/testing pipeline."""
        keep_checkpoint_scaler = (
            getattr(agent, "_scaler_loaded_from_checkpoint", False)
            and os.environ.get("ROBOT_DIFT_OVERRIDE_CHECKPOINT_SCALER", "").lower()
            not in {"1", "true", "yes"}
        )
        if keep_checkpoint_scaler:
            log.info(
                "Keeping action scaler from checkpoint. "
                "Set ROBOT_DIFT_OVERRIDE_CHECKPOINT_SCALER=1 to use current dataset scaler."
            )
        else:
            agent.set_scaler(self.scaler)
        if hasattr(agent, "set_robot_state_stats"):
            bounds = getattr(self.trainset, "robot_state_bounds", None)
            if bounds is not None:
                keep_checkpoint_stats = (
                    getattr(agent, "_robot_state_stats_loaded_from_checkpoint", False)
                    and os.environ.get("ROBOT_DIFT_OVERRIDE_CHECKPOINT_ROBOT_STATS", "").lower()
                    not in {"1", "true", "yes"}
                )
                if keep_checkpoint_stats:
                    log.info(
                        "Keeping robot-state normalization stats from checkpoint. "
                        "Set ROBOT_DIFT_OVERRIDE_CHECKPOINT_ROBOT_STATS=1 to use current dataset bounds."
                    )
                else:
                    agent.set_robot_state_stats(bounds)

        # define optimizer
        if getattr(agent, 'use_lr_scheduler', False):
            self.optimizer, self.scheduler = agent.configure_optimizers()
        else:
            self.optimizer = agent.configure_optimizers()
        # Snapshot flag before potential DDP wrap to avoid attribute access on wrapper later
        self.use_lr_scheduler = getattr(agent, 'use_lr_scheduler', False)
        # Optionally scale LR by number of GPUs (linear scaling rule)
        if self.is_dist and self.scale_lr_by_world_size:
            for pg in self.optimizer.param_groups:
                if 'lr' in pg and isinstance(pg['lr'], (float, int)):
                    pg['lr'] *= self.world_size

        if self.lr_scheduler_type is not None:
            if self.use_lr_scheduler:
                raise ValueError("Trainer and agent cannot both configure an LR scheduler")
            if self.lr_scheduler_type != "linear":
                raise ValueError(f"Unsupported trainer LR scheduler: {self.lr_scheduler_type}")
            updates_per_epoch = optimizer_updates_per_epoch(
                min(len(self.train_dataloader), self.max_train_batches_per_epoch)
                if self.max_train_batches_per_epoch is not None else len(self.train_dataloader),
                self.gradient_accumulation_steps,
                self.full_accumulation_batches,
            )
            self.scheduler = make_linear_lr_scheduler(
                self.optimizer, updates_per_epoch * self.epoch, self.linear_lr_end_factor,
            )
            self.use_lr_scheduler = True

        # Wrap with DDP if enabled
        if self.use_ddp:
            # Convert to CUDA device per local rank if available
            if torch.cuda.is_available():
                # Optionally convert BN to SyncBN for multi-GPU stability
                if self.use_sync_bn:
                    agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
                # Ensure we set a proper CUDA device string, e.g., 'cuda:0'
                self.device = torch.device(f"cuda:{self.local_rank}")
                agent = agent.to(self.device)
                agent = DDP(
                    agent,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
            else:
                if self.use_sync_bn:
                    agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
                agent = DDP(agent, find_unused_parameters=True)
        else:
            # move to device if single-process
            if torch.cuda.is_available():
                agent = agent.to(self.device)

        # Initialize EMA after model is placed/wrapped so devices match
        if self.if_use_ema:
            target = agent.module if isinstance(agent, DDP) else agent
            self.ema_helper = ExponentialMovingAverage(
                target.parameters(), self.decay_ema, str(self.device), power=self.ema_power,
            )

        # Initialize visualizer (only on rank 0)
        self._init_visualizer(agent)

        progress_iter = range(self.epoch)
        # Only show tqdm on rank 0
        if (not self.is_dist) or self.rank == 0:
            progress_iter = tqdm(progress_iter)

        for num_epoch in progress_iter:

            epoch_loss = torch.tensor(0.0).to(self.device)
            epoch_loss_base = torch.tensor(0.0).to(self.device)
            alignment_weight, alignment_freq = self._alignment_parameters(num_epoch)
            self._alignment_weight = alignment_weight
            self._alignment_freq = alignment_freq
            alignment_loss_epoch_sum = 0.0
            alignment_loss_epoch_count = 0.0

            # Ensure different shuffles across ranks each epoch
            if self.is_dist and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(num_epoch)

            self.optimizer.zero_grad(set_to_none=True)
            num_batches = len(self.train_dataloader)
            if self.max_train_batches_per_epoch is not None:
                num_batches = min(num_batches, self.max_train_batches_per_epoch)
            if self.full_accumulation_batches:
                num_batches -= num_batches % self.gradient_accumulation_steps
                if num_batches == 0:
                    raise ValueError("No full gradient-accumulation group fits in an epoch")
            for batch_idx, data in enumerate(self.train_dataloader):
                if batch_idx >= num_batches:
                    break
                obs_dict, action, mask = data
                if self.full_accumulation_batches and action.shape[0] != self.train_dataloader.batch_size:
                    raise ValueError(
                        "A partial microbatch would violate the configured effective global batch size"
                    )

                alignment_context = None
                if alignment_weight > 0.0:
                    should_compute_alignment = alignment_freq >= 1.0 or random.random() < alignment_freq
                    if should_compute_alignment:
                        batch_size = action.shape[0]
                        sample_indices = None
                        if alignment_freq < 1.0 and batch_size > 0:
                            sample_count = max(1, int(math.ceil(alignment_freq * batch_size)))
                            if sample_count < batch_size:
                                sample_indices = torch.randperm(batch_size)[:sample_count].tolist()

                        alignment_context = {
                            "weight": alignment_weight,
                            "sample_indices": sample_indices,
                        }
                        target_agent = agent.module if isinstance(agent, DDP) else agent
                        alignment_views = getattr(target_agent.img_encoder, "alignment_views", None) if hasattr(target_agent, "img_encoder") else None
                        if alignment_views is not None:
                            alignment_context["views"] = alignment_views
                # put data on cuda
                for camera in obs_dict.keys():
                    if camera == 'lang':
                        continue

                    obs_dict[camera] = obs_dict[camera].to(self.device)

                    # Apply temporal slicing to images
                    if 'rgb' in camera or 'image' in camera:
                        obs_dict[camera] = obs_dict[camera][:, :self.obs_seq_len].contiguous()
                    # Also apply temporal slicing to robot_states
                    elif camera == 'robot_states':
                        obs_dict[camera] = obs_dict[camera][:, :self.obs_seq_len].contiguous()

                action = self.scaler.scale_output(action)
                action = action[:, self.obs_seq_len - 1:, :].contiguous()

                accum_group_start = (
                    batch_idx // self.gradient_accumulation_steps
                ) * self.gradient_accumulation_steps
                accum_group_end = min(
                    accum_group_start + self.gradient_accumulation_steps,
                    num_batches,
                )
                accum_group_size = max(1, accum_group_end - accum_group_start)
                update_optimizer = (batch_idx + 1) == accum_group_end

                batch_loss, batch_alignment = self.train_one_step(
                    agent,
                    obs_dict,
                    action,
                    alignment_context=alignment_context,
                    update_optimizer=update_optimizer,
                    loss_divisor=accum_group_size,
                    sync_gradients=update_optimizer,
                )
                raw_batch_alignment = None
                weighted_batch_alignment = None
                if batch_alignment is not None:
                    raw_batch_alignment, weighted_batch_alignment = batch_alignment

                if (
                    alignment_weight > 0.0
                    and weighted_batch_alignment is not None
                    and ((not self.is_dist) or self.rank == 0)
                    and num_epoch == 0
                    and batch_idx == 0
                ):
                    log.warning("[Alignment] weighted loss (batch): %.6f", weighted_batch_alignment.item())

                epoch_loss += batch_loss
                base_loss = batch_loss
                if weighted_batch_alignment is not None:
                    base_loss = base_loss - weighted_batch_alignment
                epoch_loss_base += base_loss
                if raw_batch_alignment is not None:
                    alignment_loss_epoch_sum += raw_batch_alignment.item()
                    alignment_loss_epoch_count += 1

            epoch_loss = epoch_loss / num_batches
            epoch_loss_base = epoch_loss_base / num_batches

            # Reduce loss across ranks for logging
            if self.is_dist:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss = epoch_loss / self.world_size
                dist.all_reduce(epoch_loss_base, op=dist.ReduceOp.SUM)
                epoch_loss_base = epoch_loss_base / self.world_size

            alignment_stats = torch.tensor(
                [alignment_loss_epoch_sum, alignment_loss_epoch_count],
                device=self.device,
                dtype=torch.float32,
            )
            if self.is_dist:
                dist.all_reduce(alignment_stats, op=dist.ReduceOp.SUM)
            alignment_avg = None
            if alignment_stats[1].item() > 0:
                alignment_avg = (alignment_stats[0] / alignment_stats[1]).item()

            if (not self.is_dist) or self.rank == 0:
                log_payload = {
                    "train_loss": epoch_loss_base.item(),
                    "epoch": num_epoch,
                    "train_loss_total": epoch_loss.item(),
                }
                if alignment_avg is not None:
                    log_payload["alignment_loss"] = alignment_avg
                unwrapped = agent.module if isinstance(agent, DDP) else agent
                encoder = getattr(unwrapped, "img_encoder", None)
                if encoder is not None and getattr(encoder, "feature_cache_max_bytes", 0) > 0:
                    cache_hits = int(encoder.feature_cache_hits)
                    cache_misses = int(encoder.feature_cache_misses)
                    log_payload.update({
                        "student_cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
                        "student_cache_gib": encoder._feature_cache_bytes / (1024 ** 3),
                    })
                    log.info(
                        "Rank-0 Student cache: hits=%d misses=%d entries=%d GiB=%.2f",
                        cache_hits, cache_misses, len(encoder._feature_cache),
                        encoder._feature_cache_bytes / (1024 ** 3),
                    )
                wandb.log(log_payload)
                log.info("Epoch {}: Mean train loss (base) is {}".format(num_epoch, epoch_loss_base.item()))

            # Run visualization at specified intervals
            self._run_visualization(agent, num_epoch)

            save_every = int(os.environ.get("ROBOT_DIFT_SAVE_EVERY_N_EPOCHS", "0") or 0)
            if save_every > 0 and ((num_epoch + 1) % save_every == 0):
                if (not self.is_dist) or self.rank == 0:
                    extra_ckpt_dir = os.environ.get("ROBOT_DIFT_POLICY_OUTPUT_DIR")
                    if extra_ckpt_dir:
                        module_to_save = agent.module if isinstance(agent, DDP) else agent
                        epoch_ckpt_dir = os.path.join(extra_ckpt_dir, f"epoch_{num_epoch + 1:04d}")
                        os.makedirs(epoch_ckpt_dir, exist_ok=True)
                        module_to_save.store_model_weights(epoch_ckpt_dir, sv_name="last_model")
                        module_to_save.store_model_scaler(epoch_ckpt_dir)
                        log.info("Saved intermediate model checkpoint to %s", epoch_ckpt_dir)

            if self._should_run_simulation(num_epoch):
                self._run_simulation(agent, current_epoch=num_epoch + 1)

        if (not self.is_dist) or self.rank == 0:
            log.info("training done")

        if self.if_use_ema:
            to_store = agent.module if isinstance(agent, DDP) else agent
            self.ema_helper.store(to_store.parameters())
            self.ema_helper.copy_to(to_store.parameters())

        # Save only on rank 0
        if (not self.is_dist) or self.rank == 0:
            # If model was wrapped in DDP, unwrap
            module_to_save = agent.module if isinstance(agent, DDP) else agent
            module_to_save.store_model_weights(module_to_save.working_dir, sv_name='last_model')
            module_to_save.store_model_scaler(module_to_save.working_dir)
            extra_ckpt_dir = os.environ.get("ROBOT_DIFT_POLICY_OUTPUT_DIR")
            if extra_ckpt_dir:
                os.makedirs(extra_ckpt_dir, exist_ok=True)
                module_to_save.store_model_weights(extra_ckpt_dir, sv_name='last_model')
                module_to_save.store_model_scaler(extra_ckpt_dir)
        # or send weight out of the class

    def train_one_step(
        self,
        agent: BaseAgent,
        obs_dict,
        action,
        alignment_context: Optional[dict] = None,
        update_optimizer: bool = True,
        loss_divisor: int = 1,
        sync_gradients: bool = True,
    ):
        """Run a single training step."""
        agent.train()

        sync_context = agent.no_sync() if isinstance(agent, DDP) and not sync_gradients else nullcontext()
        with sync_context:
            forward_out = agent(obs_dict, action, alignment_context=alignment_context)

            if isinstance(forward_out, tuple):
                loss, alignment_info = forward_out
            else:
                loss = forward_out
                alignment_info = None

            raw_alignment = None
            weighted_alignment = None
            if alignment_info is not None:
                if isinstance(alignment_info, tuple):
                    raw_alignment, weighted_alignment = alignment_info
                else:
                    weighted_alignment = alignment_info
                    raw_alignment = alignment_info

            loss_for_backward = loss / max(1, int(loss_divisor))
            if self.use_amp:
                self.grad_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        if update_optimizer:
            did_optimizer_step = True
            if self.use_amp:
                scale_before = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                # GradScaler lowers its scale when it skipped optimizer.step()
                # after nonfinite gradients. Disabled scalers keep scale 1.
                did_optimizer_step = self.grad_scaler.get_scale() >= scale_before
            else:
                self.optimizer.step()
            if did_optimizer_step and self.use_lr_scheduler:
                self.scheduler.step()
            if did_optimizer_step and self.if_use_ema:
                target = agent.module if isinstance(agent, DDP) else agent
                self.ema_helper.update(target.parameters())
            self.optimizer.zero_grad(set_to_none=True)

        raw_alignment_detached = raw_alignment.detach() if raw_alignment is not None else None
        weighted_alignment_detached = weighted_alignment.detach() if weighted_alignment is not None else None

        alignment_return = None
        if raw_alignment_detached is not None or weighted_alignment_detached is not None:
            alignment_return = (raw_alignment_detached, weighted_alignment_detached)

        return loss.detach(), alignment_return

    @staticmethod
    def _unwrap_module(module):
        return module.module if isinstance(module, DDP) else module

    def _alignment_parameters(self, epoch: int) -> tuple[float, float]:
        if not self.alignment_cfg:
            return 0.0, 0.0

        weight_base = float(self.alignment_cfg.get("weight", 0.0) or 0.0)
        if weight_base <= 0.0:
            return 0.0, 0.0

        warmdown_frac = float(self.alignment_cfg.get("warmdown_frac", 0.5) or 0.0)
        warmdown_epochs = max(1, int(self.epoch * warmdown_frac)) if warmdown_frac > 0 else 1
        min_decay_factor = float(self.alignment_cfg.get("min_decay_factor", 0.001) or 0.0)

        if epoch < warmdown_epochs:
            normalized_epoch = epoch / warmdown_epochs
            decay_factor = min_decay_factor + (1.0 - min_decay_factor) * ((1.0 - normalized_epoch) ** 2)
        else:
            decay_factor = min_decay_factor

        weight = weight_base * decay_factor

        freq_high = float(self.alignment_cfg.get("freq", 1.0) or 0.0)
        freq_low = float(self.alignment_cfg.get("freq_min", freq_high) or 0.0)
        drop_ratio = float(self.alignment_cfg.get("freq_drop_ratio", 0.01) or 0.0)
        drop_threshold = weight_base * drop_ratio

        freq = freq_low if weight <= drop_threshold else freq_high
        freq = max(0.0, min(1.0, freq))

        return weight, freq

    def _should_run_simulation(self, num_epoch: int) -> bool:
        if self.sim_eval_every_n_epochs <= 0:
            return False
        if self.simulation_cfg is None:
            return False
        return ((num_epoch + 1) % self.sim_eval_every_n_epochs) == 0

    def _ensure_simulation(self):
        if self._simulation is None and self.simulation_cfg is not None:
            self._simulation = hydra.utils.instantiate(self.simulation_cfg)
        return self._simulation

    def _call_simulation(self, simulation, agent, current_epoch: int):
        sim_fn = simulation.test_agent
        kwargs = {}
        try:
            signature = inspect.signature(sim_fn)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            if "step" in signature.parameters:
                kwargs["step"] = current_epoch
            elif "epoch" in signature.parameters:
                kwargs["epoch"] = current_epoch

        if kwargs:
            try:
                return sim_fn(agent, **kwargs)
            except TypeError:
                log.debug("Simulation hook does not accept %s; retrying without kwargs.", list(kwargs.keys()))

        return sim_fn(agent)

    def _run_simulation(self, agent, current_epoch: int) -> None:
        if self.is_dist:
            dist.barrier()

        try:
            if (not self.is_dist) or self.rank == 0:
                simulation = self._ensure_simulation()
                if simulation is None:
                    return

                model = self._unwrap_module(agent)
                was_training = model.training

                try:
                    if self.if_use_ema:
                        self.ema_helper.store(model.parameters())
                        self.ema_helper.copy_to(model.parameters())

                    if os.environ.get("ROBOT_DIFT_SAVE_SIM_EMA", "0").lower() in {"1", "true", "yes"}:
                        output_root = os.environ.get("ROBOT_DIFT_POLICY_OUTPUT_DIR")
                        if not output_root:
                            raise ValueError("ROBOT_DIFT_SAVE_SIM_EMA requires ROBOT_DIFT_POLICY_OUTPUT_DIR")
                        eval_checkpoint_dir = os.path.join(output_root, f"epoch_{current_epoch:04d}_eval")
                        os.makedirs(eval_checkpoint_dir, exist_ok=True)
                        model.store_model_weights(eval_checkpoint_dir, sv_name="last_model")
                        model.store_model_scaler(eval_checkpoint_dir)
                        log.info("Saved exact simulation policy checkpoint to %s", eval_checkpoint_dir)

                    model.eval()
                    with torch.no_grad():
                        metrics = self._call_simulation(simulation, model, current_epoch)
                finally:
                    if was_training:
                        model.train()
                    if self.if_use_ema:
                        self.ema_helper.restore(model.parameters())

                if metrics is None:
                    metrics = {}
                elif not isinstance(metrics, dict):
                    metrics = {"simulation_metric": metrics}

                metrics.setdefault("simulation_epoch", current_epoch)
                metrics.setdefault("epoch", current_epoch)
                log.info("Simulation evaluation at epoch %d: %s", current_epoch, metrics)
                wandb.log(metrics)
        finally:
            if self.is_dist:
                dist.barrier()

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None
