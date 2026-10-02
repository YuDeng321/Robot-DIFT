"""Check that DROID epochs count optimizer updates, not microbatches."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "droid_policy_learning"))
TrainUtils = pytest.importorskip("robomimic.utils.train_utils")


class ToyAlgo:
    def __init__(self, accumulation_steps):
        self.global_config = SimpleNamespace(
            train={"gradient_accumulation_steps": accumulation_steps}
        )
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.optimizer = torch.optim.SGD([self.weight], lr=0.01)
        self.calls = []
        self.optimizer_steps = 0

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def process_batch_for_training(self, batch):
        return batch

    def postprocess_batch_for_training(self, batch, obs_normalization_stats=None):
        return batch

    def train_on_batch(
        self, batch, epoch, validate=False, accumulation_step=0, accumulation_steps=1
    ):
        self.calls.append((epoch, accumulation_step, accumulation_steps, validate))
        loss = ((self.weight * batch["x"] - batch["y"]) ** 2).mean()
        update_complete = False
        if not validate:
            if accumulation_step == 0:
                self.optimizer.zero_grad(set_to_none=True)
            (loss / accumulation_steps).backward()
            if accumulation_step == accumulation_steps - 1:
                self.optimizer.step()
                self.optimizer_steps += 1
                update_complete = True
        return {"loss": loss.detach(), "optimizer_step": update_complete}

    def log_info(self, info):
        log = {"Loss": info["loss"].item()}
        if info["optimizer_step"]:
            log["Optimizer_Step"] = 1.0
        return log


def test_accumulation_matches_full_batch_and_preserves_step_count():
    microbatches = [
        {"x": torch.tensor([float(i)]), "y": torch.tensor([0.25 * i])}
        for i in range(1, 9)
    ]
    model = ToyAlgo(accumulation_steps=4)
    stats, remaining = TrainUtils.run_epoch(
        model=model,
        data_loader=microbatches,
        epoch=3,
        num_steps=2,
    )

    reference = ToyAlgo(accumulation_steps=1)
    for start in (0, 4):
        combined = {
            key: torch.cat([batch[key] for batch in microbatches[start:start + 4]])
            for key in ("x", "y")
        }
        reference.train_on_batch(combined, epoch=3)

    assert model.optimizer_steps == 2
    assert stats["Optimizer_Step"] == 1.0
    assert model.calls == [(3, micro, 4, False) for micro in range(4)] * 2
    assert torch.allclose(model.weight, reference.weight, atol=1e-7)
    with pytest.raises(StopIteration):
        next(remaining)


def test_validation_uses_one_batch_even_with_accumulation_configured():
    model = ToyAlgo(accumulation_steps=4)
    TrainUtils.run_epoch(
        model=model,
        data_loader=[{"x": torch.tensor([1.0]), "y": torch.tensor([0.0])}],
        epoch=3,
        num_steps=1,
        validate=True,
    )
    assert model.calls == [(3, 0, 1, True)]
    assert model.optimizer_steps == 0


def test_default_one_microbatch_keeps_one_update_per_batch():
    model = ToyAlgo(accumulation_steps=1)
    batches = [
        {"x": torch.tensor([1.0]), "y": torch.tensor([0.0])},
        {"x": torch.tensor([2.0]), "y": torch.tensor([0.0])},
    ]
    TrainUtils.run_epoch(model=model, data_loader=batches, epoch=3, num_steps=2)
    assert model.calls == [(3, 0, 1, False)] * 2
    assert model.optimizer_steps == 2
