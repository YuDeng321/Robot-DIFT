"""Distributed state and gradient synchronization for the DROID Algo loop.

The robomimic Algo calls individual children of ``model.nets`` rather than
``model.nets.forward``. Wrapping the ModuleDict in DDP therefore bypasses DDP's
forward/reducer protocol. This module synchronizes the actual optimizer
gradients once per optimizer step, after local gradient accumulation.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def _world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def broadcast_training_state(module: nn.Module, ema=None) -> None:
    """Start every rank from rank zero's model, buffers, and EMA weights."""
    if _world_size() == 1:
        return
    for name, tensor in module.state_dict().items():
        if dist.get_backend() == "nccl" and tensor.device.type != "cuda":
            raise RuntimeError(f"NCCL cannot broadcast CPU model state: {name}")
        dist.broadcast(tensor, src=0)
    if ema is not None:
        for index, tensor in enumerate(ema.shadow_params):
            if dist.get_backend() == "nccl" and tensor.device.type != "cuda":
                raise RuntimeError(f"NCCL cannot broadcast CPU EMA state: {index}")
            dist.broadcast(tensor, src=0)


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


@torch.no_grad()
def synchronize_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    *,
    max_bucket_bytes: int = 25 * 1024 * 1024,
) -> None:
    """Average accumulated gradients, including rank-local unused parameters.

    A single reduced presence vector keeps collective order identical when a
    parameter receives a gradient on only some ranks. Gradients are reduced in
    bounded, same-device/dtype buckets to limit temporary memory.
    """
    world_size = _world_size()
    if world_size == 1:
        return
    if max_bucket_bytes < 1:
        raise ValueError("max_bucket_bytes must be positive")
    parameters = _optimizer_parameters(optimizer)
    if not parameters:
        return
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("Distributed DROID optimizer parameters must share one device")
    device = parameters[0].device
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.uint8,
        device=device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)

    bucket: list[torch.Tensor] = []
    bucket_bytes = 0
    bucket_dtype = None

    def flush() -> None:
        nonlocal bucket, bucket_bytes, bucket_dtype
        if bucket:
            dist.all_reduce_coalesced(bucket, op=dist.ReduceOp.SUM)
            for gradient in bucket:
                gradient.div_(world_size)
        bucket = []
        bucket_bytes = 0
        bucket_dtype = None

    for parameter, active in zip(parameters, presence.tolist()):
        if not active:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad
        if gradient.is_sparse:
            raise TypeError("Sparse gradients are unsupported by DROID gradient buckets")
        size = gradient.numel() * gradient.element_size()
        if bucket and (gradient.dtype != bucket_dtype or bucket_bytes + size > max_bucket_bytes):
            flush()
        bucket.append(gradient)
        bucket_bytes += size
        bucket_dtype = gradient.dtype
    flush()
