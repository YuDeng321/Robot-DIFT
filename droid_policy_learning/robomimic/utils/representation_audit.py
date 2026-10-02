"""Optional, non-mutating task/alignment gradient and Student update audit.

Gradient measurements use the first microbatch on every rank, averaged before
computing norms. They are not the complete accumulated optimizer gradient.
Actual updates are measured after the full accumulated optimizer step. Only
small declared U-Net tensors are sampled; results do not describe every weight.
"""

import json
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist


STUDENT_SUFFIXES = (
    "conv_in.weight",
    "mid_block.resnets.0.norm1.weight",
    "up_blocks.0.resnets.0.norm1.weight",
    "up_blocks.1.resnets.0.norm1.weight",
    "up_blocks.2.resnets.0.norm1.weight",
    "up_blocks.3.resnets.0.norm1.weight",
)


def select_student_parameters(module):
    prefix = "unet_feature_extractor_cleandift."
    selected = [(name, p) for name, p in module.named_parameters()
                if p.requires_grad and prefix in name
                and name.split(prefix, 1)[1] in STUDENT_SUFFIXES]
    if not selected:
        raise ValueError("Representation audit found no declared Student tensors")
    return selected


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _mean_gradient(value, parameter):
    gradient = torch.zeros_like(parameter, dtype=torch.float32) if value is None else value.detach().float().clone()
    if _distributed():
        dist.all_reduce(gradient)
        gradient.div_(dist.get_world_size())
    return gradient


def input_digest(value):
    """Hash nested processed inputs without storing robot images or actions."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, ensure_ascii=True).encode())
        digest.update(b"\x00")
    visit(value)
    return digest.hexdigest()


class RepresentationAudit:
    def __init__(self, named_parameters, path=None):
        self.parameters = dict(named_parameters)
        if not self.parameters:
            raise ValueError("Audit needs at least one parameter")
        self.initial = {name: p.detach().float().clone() for name, p in self.parameters.items()}
        self.path = Path(path) if path else None
        self.pending = None

    def gradients(self, policy_loss, weighted_alignment_loss, *, step, microbatch_size, batch=None):
        params = list(self.parameters.values())
        def grads(loss):
            return (torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    if loss.requires_grad else (None,) * len(params))
        policy, alignment = grads(policy_loss), grads(weighted_alignment_loss)
        rows = {}
        for (name, p), gp, ga in zip(self.parameters.items(), policy, alignment):
            gp, ga = _mean_gradient(gp, p), _mean_gradient(ga, p)
            pn, an = float(gp.norm()), float(ga.norm())
            rows[name] = {
                "policy_gradient_norm": pn,
                "weighted_alignment_gradient_norm": an,
                "alignment_to_policy_norm_ratio": an / pn if pn > 0 else None,
                "cosine": float((gp.flatten() @ ga.flatten()) / (pn * an)) if pn * an > 0 else None,
                "relative_drift_from_run_start": float((p.detach().float() - self.initial[name]).norm()
                                                       / self.initial[name].norm().clamp_min(1e-12)),
            }
        self.pending = {
            "optimizer_step": int(step),
            "scope": "selected Student tensors; gradients of rank-averaged first microbatch, before clipping",
            "first_microbatch_global_size": int(microbatch_size) * (dist.get_world_size() if _distributed() else 1),
            "world_size": dist.get_world_size() if _distributed() else 1,
            "tensors": rows,
        }
        if batch is not None:
            digest = input_digest(batch)
            digests = [None] * (dist.get_world_size() if _distributed() else 1)
            if _distributed():
                dist.all_gather_object(digests, digest)
            else:
                digests[0] = digest
            self.pending["processed_first_microbatch_sha256_by_rank"] = digests
        return self.pending

    def before_update(self):
        return {name: p.detach().float().clone() for name, p in self.parameters.items()}

    def after_update(self, before, optimizer, *, step, optimizer_stepped):
        if self.pending is None or self.pending["optimizer_step"] != int(step):
            raise ValueError("Missing first-microbatch gradient audit for this optimizer step")
        lr_by_id = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
        for name, p in self.parameters.items():
            self.pending["tensors"][name].update({
                "learning_rate": float(lr_by_id[id(p)]),
                "relative_optimizer_update": float((p.detach().float() - before[name]).norm()
                                                   / before[name].norm().clamp_min(1e-12)),
            })
        self.pending["optimizer_stepped"] = bool(optimizer_stepped)
        result = self.pending
        if not _distributed() or dist.get_rank() == 0:
            line = json.dumps(result, sort_keys=True, allow_nan=False)
            print("[RepresentationAudit] " + line, flush=True)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        self.pending = None
        return result
