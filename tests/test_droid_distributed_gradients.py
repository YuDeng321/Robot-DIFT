"""Two-rank gradient synchronization contract for the DROID Algo loop."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "droid_policy_learning"))

from robomimic.utils.distributed_gradients import (  # noqa: E402
    broadcast_training_state,
    synchronize_optimizer_gradients,
)


class _TwoParameterModel(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.common = nn.Parameter(torch.tensor([1.0 + 8.0 * rank]))
        self.rank_local = nn.Parameter(torch.tensor([3.0 + 8.0 * rank]))


class _EMA:
    def __init__(self, rank: int):
        self.shadow_params = [torch.tensor([5.0 + rank])]


def _train_worker(rank: int, store: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        model = _TwoParameterModel(rank)
        ema = _EMA(rank)
        broadcast_training_state(model, ema)
        assert model.common.item() == 1.0
        assert model.rank_local.item() == 3.0
        assert ema.shadow_params[0].item() == 5.0

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        # Accumulate two local microbatches before one global optimizer step.
        for _ in range(2):
            loss = model.common * (rank + 1) / 2
            if rank == 1:
                loss = loss + model.rank_local
            loss.backward()
        synchronize_optimizer_gradients(optimizer, max_bucket_bytes=4)
        assert model.common.grad.item() == pytest.approx(1.5)
        assert model.rank_local.grad.item() == pytest.approx(1.0)
        optimizer.step()
        assert model.common.item() == pytest.approx(0.85)
        assert model.rank_local.item() == pytest.approx(2.9)
        gathered = [torch.zeros(2) for _ in range(2)]
        dist.all_gather(
            gathered,
            torch.stack((model.common.detach()[0], model.rank_local.detach()[0])),
        )
        assert torch.equal(gathered[0], gathered[1])

        # A single-rank overflow must reach every rank before GradScaler checks
        # gradients; otherwise one rank can step while another skips.
        optimizer.zero_grad(set_to_none=True)
        model.common.grad = torch.tensor([float("inf") if rank == 0 else 1.0])
        synchronize_optimizer_gradients(optimizer, max_bucket_bytes=4)
        assert torch.isinf(model.common.grad).all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_ranks_average_accumulated_and_rank_local_gradients(tmp_path):
    mp.spawn(_train_worker, args=(str(tmp_path / "gloo_store"),), nprocs=2, join=True)
