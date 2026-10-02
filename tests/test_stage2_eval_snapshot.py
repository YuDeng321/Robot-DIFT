from pathlib import Path

import pytest
import torch
from torch import nn

from trainers.base_trainer import BaseTrainer


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def store_model_weights(self, folder, sv_name):
        torch.save(self.state_dict(), Path(folder) / f"{sv_name}.pth")

    def store_model_scaler(self, folder):
        (Path(folder) / "model_scaler.pkl").write_bytes(b"scaler")


class _EMA:
    def store(self, parameters):
        self.before = [p.detach().clone() for p in parameters]

    def copy_to(self, parameters):
        for p in parameters:
            p.data.add_(1)

    def restore(self, parameters):
        for p, previous in zip(parameters, self.before):
            p.data.copy_(previous)


class _CrashingSim:
    def test_agent(self, agent, step):
        assert step == 20
        assert float(agent.weight) == 2.0
        raise RuntimeError("simulator stopped")


def test_eval_snapshot_survives_simulator_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_DIFT_SAVE_SIM_EMA", "1")
    monkeypatch.setenv("ROBOT_DIFT_POLICY_OUTPUT_DIR", str(tmp_path))
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.is_dist = False
    trainer.rank = 0
    trainer.if_use_ema = True
    trainer.ema_helper = _EMA()
    trainer._ensure_simulation = lambda: _CrashingSim()
    model = _Model()

    with pytest.raises(RuntimeError, match="simulator stopped"):
        trainer._run_simulation(model, current_epoch=20)

    snapshot = tmp_path / "epoch_0020_eval" / "last_model.pth"
    assert float(torch.load(snapshot, weights_only=True)["weight"]) == 2.0
    assert float(model.weight) == 1.0
    assert model.training
