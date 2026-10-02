"""CPU behavior checks for the opt-in Stage-II paper numeric schedule."""

from __future__ import annotations

import copy

import pytest
import torch

from agents.utils.ema import ExponentialMovingAverage
from trainers.base_trainer import BaseTrainer, make_linear_lr_scheduler, optimizer_updates_per_epoch


def test_power_ema_uses_optimizer_update_count_and_restores_schedule():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    ema = ExponentialMovingAverage([parameter], decay=1.0, device="cpu", power=0.75)

    with torch.no_grad():
        parameter.fill_(1.0)
    ema.update([parameter])
    first_decay = 1.0 - 2.0 ** -0.75
    assert ema.num_updates == 1
    assert ema.shadow_params[0].item() == pytest.approx(1.0 - first_decay)

    restored = ExponentialMovingAverage([parameter], decay=0.995, device="cpu")
    restored.load_state_dict(copy.deepcopy(ema.state_dict()))
    assert restored.power == 0.75
    with torch.no_grad():
        parameter.fill_(2.0)
    ema.update([parameter])
    restored.update([parameter])
    assert ema.num_updates == restored.num_updates == 2
    torch.testing.assert_close(ema.shadow_params[0], restored.shadow_params[0])

    legacy = copy.deepcopy(ema.state_dict())
    legacy.pop("power")
    restored.load_state_dict(legacy)
    assert restored.power is None


def test_ema_store_restores_only_parameters_that_copy_to_can_change():
    trainable = torch.nn.Parameter(torch.tensor(2.0))
    frozen = torch.nn.Parameter(torch.tensor(9.0), requires_grad=False)
    params = [trainable, frozen]
    ema = ExponentialMovingAverage(params, decay=0.9, device="cpu")
    ema.store(params)
    assert len(ema.collected_params) == 1
    ema.shadow_params[0].fill_(4.0)
    ema.copy_to(params)
    assert trainable.item() == 4.0 and frozen.item() == 9.0
    ema.restore(params)
    assert trainable.item() == 2.0 and frozen.item() == 9.0


def test_linear_lr_counts_full_accumulation_groups():
    assert optimizer_updates_per_epoch(10, 4, full_groups_only=True) == 2
    assert optimizer_updates_per_epoch(10, 4, full_groups_only=False) == 3
    with pytest.raises(ValueError, match="No full optimizer"):
        optimizer_updates_per_epoch(3, 4, full_groups_only=True)

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = make_linear_lr_scheduler(optimizer, total_updates=4, end_factor=0.1)
    actual = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        actual.append(optimizer.param_groups[0]["lr"])
    assert actual == pytest.approx([0.775, 0.55, 0.325, 0.1])

    with pytest.raises(ValueError, match="end_factor"):
        make_linear_lr_scheduler(optimizer, total_updates=4, end_factor=-0.1)


def test_amp_overflow_does_not_advance_lr_or_ema():
    class TinyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, obs_dict, action, alignment_context=None):
            return self.weight.square(), None

    class FakeScaler:
        def __init__(self):
            self.overflow = True
            self.scale_value = 2.0

        def scale(self, loss):
            return loss

        def get_scale(self):
            return self.scale_value

        def step(self, optimizer):
            if not self.overflow:
                optimizer.step()

        def update(self):
            if self.overflow:
                self.scale_value /= 2.0

    agent = TinyAgent()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
    trainer.scheduler = make_linear_lr_scheduler(trainer.optimizer, total_updates=2, end_factor=0.1)
    trainer.ema_helper = ExponentialMovingAverage(agent.parameters(), decay=1.0, device="cpu", power=0.75)
    trainer.use_amp = True
    trainer.use_lr_scheduler = True
    trainer.if_use_ema = True
    trainer.grad_scaler = FakeScaler()

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    initial_scheduler_epoch = trainer.scheduler.last_epoch
    trainer.train_one_step(agent, {}, None)
    assert agent.weight.item() == pytest.approx(1.0)
    assert trainer.optimizer.param_groups[0]["lr"] == initial_lr
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch
    assert trainer.ema_helper.num_updates == 0

    trainer.grad_scaler.overflow = False
    trainer.train_one_step(agent, {}, None)
    assert agent.weight.item() < 1.0
    assert trainer.optimizer.param_groups[0]["lr"] < initial_lr
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch + 1
    assert trainer.ema_helper.num_updates == 1
