import math
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _orthogonalize_newton_schulz(
    update: torch.Tensor,
    ns_coefficients=DEFAULT_NS_COEFFS,
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    if update.ndim != 2:
        raise ValueError(f"Muon update must be 2D after flattening, got {tuple(update.shape)}")
    if ns_steps >= 100:
        raise ValueError("Muon ns_steps must be less than 100")

    a, b, c = ns_coefficients
    x = update.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    x = x / x.norm().clamp(min=eps)
    for _ in range(int(ns_steps)):
        gram = x @ x.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, gram_update, x, beta=a)

    if transposed:
        x = x.T
    return x.to(dtype=update.dtype)


def _muon_adjusted_lr(lr: float, adjust_lr_fn: Optional[str], matrix_shape: torch.Size) -> float:
    rows, cols = int(matrix_shape[0]), int(matrix_shape[1])
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, rows / max(1, cols)))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    if adjust_lr_fn == "none":
        return lr
    raise ValueError(f"Unsupported Muon adjust_lr_fn={adjust_lr_fn}")


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Muon only supports parameters with ndim >= 2, got {tuple(tensor.shape)}")
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _group_uses_muon(group_name: str, scope: str, explicit_value) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    scope = str(scope).lower()
    group_name = str(group_name).lower()
    if scope == "all":
        return True
    if scope == "backbone":
        return group_name == "backbone" or group_name.endswith("_backbone")
    if scope == "vision":
        return group_name in {"backbone", "head", "vision"} or "vision" in group_name
    if scope in {"none", "off", "false"}:
        return False
    raise ValueError(f"Unsupported muon_scope={scope}")


def split_muon_adamw_groups(param_groups: Iterable[dict], muon_scope: str = "all") -> list[dict]:
    split_groups = []
    for group in param_groups:
        group = dict(group)
        params = [p for p in list(group.pop("params")) if p.requires_grad]
        if not params:
            continue

        group_name = str(group.get("name", "unnamed"))
        explicit_use_muon = group.pop("use_muon", None)
        use_muon = _group_uses_muon(group_name, muon_scope, explicit_use_muon)

        muon_params = [p for p in params if use_muon and p.ndim >= 2]
        muon_param_ids = {id(p) for p in muon_params}
        adamw_params = [p for p in params if id(p) not in muon_param_ids]

        if muon_params:
            muon_group = dict(group)
            muon_group["params"] = muon_params
            muon_group["name"] = f"{group_name}_muon"
            muon_group["use_muon"] = True
            split_groups.append(muon_group)

        if adamw_params:
            adamw_group = dict(group)
            adamw_group["params"] = adamw_params
            adamw_group["name"] = f"{group_name}_adamw"
            adamw_group["use_muon"] = False
            split_groups.append(adamw_group)

    return split_groups


class MuonWithAdamW(Optimizer):
    """Hybrid optimizer: Muon for matrix/tensor hidden weights, AdamW fallback otherwise."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_ns_coefficients=DEFAULT_NS_COEFFS,
        muon_eps: float = 1e-7,
        muon_adjust_lr_fn: Optional[str] = "match_rms_adamw",
        muon_scope: str = "all",
    ):
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument should be an iterable, not a Tensor")
        param_groups = list(params)
        if param_groups and not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        split_groups = split_muon_adamw_groups(param_groups, muon_scope=muon_scope)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_ns_steps=muon_ns_steps,
            muon_ns_coefficients=tuple(muon_ns_coefficients),
            muon_eps=muon_eps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            use_muon=False,
        )
        super().__init__(split_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group):
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        momentum = float(group.get("muon_momentum", 0.95))
        nesterov = bool(group.get("muon_nesterov", True))
        ns_steps = int(group.get("muon_ns_steps", 5))
        ns_coefficients = tuple(group.get("muon_ns_coefficients", DEFAULT_NS_COEFFS))
        muon_eps = float(group.get("muon_eps", 1e-7))
        adjust_lr_fn = group.get("muon_adjust_lr_fn", "match_rms_adamw")

        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1.0 - momentum)
            update = grad.lerp(buf, momentum) if nesterov else buf

            update_matrix = _as_matrix(update)
            ortho_matrix = _orthogonalize_newton_schulz(
                update_matrix,
                ns_coefficients=ns_coefficients,
                ns_steps=ns_steps,
                eps=muon_eps,
            )
            adjusted_lr = _muon_adjusted_lr(lr, adjust_lr_fn, update_matrix.shape)
            p.mul_(1.0 - lr * weight_decay)
            p.add_(ortho_matrix.reshape_as(p), alpha=-adjusted_lr)

    def _step_adamw_group(self, group):
        lr = float(group["lr"])
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))

        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("AdamW fallback does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("AdamW fallback does not support sparse gradients")

            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = int(state["step"])

            p.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))
