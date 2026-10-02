# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py

from __future__ import division
from __future__ import unicode_literals

import torch


# copied from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
class ExponentialMovingAverage:
    """Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters, decay, device: str = 'cuda', use_num_updates=True, power=None):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        if power is not None and (power <= 0.0 or not use_num_updates):
            raise ValueError('EMA power requires a positive value and update counting')
        self.decay = decay
        self.power = power
        self._device = device
        self.num_updates = 0 if use_num_updates else None
        # Shadow params start on the same device as source params
        self.shadow_params = [p.clone().detach().to(p.device)
                              for p in parameters if p.requires_grad]
        self.collected_params = []
        self.steps = 0

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            if self.power is None:
                decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
            else:
                # EMAWarmup's inverse-decay schedule (inv_gamma=1), capped by
                # ``decay``. With decay=1 the power controls the full curve.
                decay = min(decay, 1.0 - (1.0 + self.num_updates) ** -self.power)
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
                if s_param.device != param.device:
                    self.shadow_params[i] = s_param.data.to(param.device)
                    s_param = self.shadow_params[i]
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if s_param.device != param.device:
                self.shadow_params[i] = s_param.data.to(param.device)
                s_param = self.shadow_params[i]
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        # copy_to only replaces trainable parameters. Cloning a frozen Student
        # and CLIP tower here would consume several GiB on every evaluation.
        self.collected_params = [param.clone() for param in parameters if param.requires_grad]

    def restore(self, parameters):
        parameters = [param for param in parameters if param.requires_grad]
        if len(self.collected_params) != len(parameters):
            raise ValueError("EMA restore parameter set differs from EMA store")
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params, power=self.power)

    def load_shadow_params(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if s_param.device != param.device:
                self.shadow_params[i] = s_param.data.to(param.device)
                s_param = self.shadow_params[i]
            if param.requires_grad:
                s_param.data.copy_(param.data)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']
        # Legacy checkpoints precede the optional power schedule.
        self.power = state_dict.get('power')


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1
