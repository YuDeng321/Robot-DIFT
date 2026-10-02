"""Rollout clipping uses physical actions after inverse normalization."""

import numpy as np
import torch

from agents.utils.scaler import ActionScaler, MinMaxScaler, Scaler


def test_action_scalers_clip_in_physical_units():
    actions = torch.tensor([[10.0, -4.0], [20.0, 6.0]])
    scalers = (
        ActionScaler(actions, True, "cpu"),
        MinMaxScaler(actions.numpy(), True, "cpu"),
        Scaler(np.zeros((2, 1), dtype=np.float32), actions.numpy(), True, "cpu"),
    )
    proposed = torch.tensor([[5.0, -9.0], [25.0, 11.0]])
    expected = torch.tensor([[9.0, -5.0], [21.0, 7.0]])
    for scaler in scalers:
        torch.testing.assert_close(scaler.clip_action(proposed), expected)
