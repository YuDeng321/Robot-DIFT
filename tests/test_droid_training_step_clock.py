"""Guard the optimizer-step clocks used by a resumable Stage-I run."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "droid_policy_learning"))
from robomimic.algo.diffusion_policy import (  # noqa: E402
    DiffusionPolicyUNet,
    alignment_weight_at_step,
)


def test_alignment_decay_uses_absolute_step_after_resume():
    kwargs = dict(
        base_weight=0.1,
        warmdown_steps=150_000,
        min_decay_factor=0.01,
        decay_power=1.0,
    )
    assert alignment_weight_at_step(**kwargs, epoch=1, first_active_epoch=1, origin="absolute") == pytest.approx(0.1)
    # A run resumed at update 100,000 must not restart the annealing ramp.
    assert alignment_weight_at_step(**kwargs, epoch=100_001, first_active_epoch=100_001, origin="absolute") == pytest.approx(0.034)
    assert alignment_weight_at_step(**kwargs, epoch=150_001, first_active_epoch=100_001, origin="absolute") == pytest.approx(0.001)
    assert alignment_weight_at_step(**kwargs, epoch=100_001, first_active_epoch=100_001, origin="first_active") == pytest.approx(0.1)


def test_scheduler_does_not_advance_when_amp_skips_all_updates():
    class CountingScheduler:
        count = 0

        def step(self):
            self.count += 1

    policy = object.__new__(DiffusionPolicyUNet)
    scheduler = CountingScheduler()
    policy.lr_schedulers = {"policy": scheduler}
    policy._optimizer_stepped_since_scheduler = False
    policy.on_epoch_end(1)
    assert scheduler.count == 0
    policy._optimizer_stepped_since_scheduler = True
    policy.on_epoch_end(2)
    assert scheduler.count == 1
    policy.on_epoch_end(3)
    assert scheduler.count == 1
