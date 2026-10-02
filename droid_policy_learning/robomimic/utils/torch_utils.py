"""
This file contains some PyTorch utilities.
"""
import math

import numpy as np
import scipy
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.optimizers.muon_adamw import MuonWithAdamW


def soft_update(source, target, tau):
    """
    Soft update from the parameters of a @source torch module to a @target torch module
    with strength @tau. The update follows target = target * (1 - tau) + source * tau.

    Args:
        source (torch.nn.Module): source network to push target network parameters towards
        target (torch.nn.Module): target network to update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(
            target_param * (1.0 - tau) + param * tau
        )


def hard_update(source, target):
    """
    Hard update @target parameters to match @source.

    Args:
        source (torch.nn.Module): source network to provide parameters
        target (torch.nn.Module): target network to update parameters for
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(param)


def get_torch_device(try_to_use_cuda):
    """
    Return torch device. If using cuda (GPU), will also set cudnn.benchmark to True
    to optimize CNNs.

    Args:
        try_to_use_cuda (bool): if True and cuda is available, will use GPU

    Returns:
        device (torch.Device): device to use for models
    """
    if try_to_use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def reparameterize(mu, logvar):
    """
    Reparameterize for the backpropagation of z instead of q.
    This makes it so that we can backpropagate through the sampling of z from
    our encoder when feeding the sampled variable to the decoder.

    (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)

    Args:
        mu (torch.Tensor): batch of means from the encoder distribution
        logvar (torch.Tensor): batch of log variances from the encoder distribution

    Returns:
        z (torch.Tensor): batch of sampled latents from the encoder distribution that
            support backpropagation
    """
    # logvar = \log(\sigma^2) = 2 * \log(\sigma)
    # \sigma = \exp(0.5 * logvar)

    # clamped for numerical stability
    logstd = (0.5 * logvar).clamp(-4, 15)
    std = torch.exp(logstd)

    # Sample \epsilon from normal distribution
    # use std to create a new tensor, so we don't have to care
    # about running on GPU or not
    eps = std.new(std.size()).normal_()

    # Then multiply with the standard deviation and add the mean
    z = eps.mul(std).add_(mu)

    return z


def optimizer_from_optim_params(net_optim_params, net):
    """
    Helper function to return a torch Optimizer from the optim_params
    section of the config for a particular network.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines the optimizer that is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    """
    optimizer_type = net_optim_params.get("optimizer_type", "adam")
    lr = net_optim_params["learning_rate"]["initial"]

    # 获取额外的优化器参数（如果有）
    betas = net_optim_params.get("betas", (0.9, 0.999))  # Adam/AdamW的默认betas
    eps = net_optim_params.get("eps", 1e-8)
    params = _parameter_groups_from_modules(net_optim_params, net, lr)

    if optimizer_type == "adam":
        return optim.Adam(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=net_optim_params["regularization"]["L2"],
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=net_optim_params["regularization"]["L2"],
        )
    elif optimizer_type in {"muon", "muon_adamw", "muonw"}:
        return MuonWithAdamW(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=net_optim_params["regularization"]["L2"],
            muon_momentum=float(net_optim_params.get("muon_momentum", 0.95)),
            muon_nesterov=bool(net_optim_params.get("muon_nesterov", True)),
            muon_ns_steps=int(net_optim_params.get("muon_ns_steps", 5)),
            muon_adjust_lr_fn=net_optim_params.get("muon_adjust_lr_fn", "match_rms_adamw"),
            muon_scope=net_optim_params.get("muon_scope", "all"),
        )
    else:
        raise ValueError(f"Unsupported optimizer_type={optimizer_type}")


STUDENT_GROUP_NAMES = frozenset({"student", "student_muon", "student_adamw"})


def student_lr_schedule(net_optim_params):
    """``(freeze_steps, warmup_steps)`` of the optional Student LR schedule; ``(0, 0)`` is off."""
    freeze_steps = int(net_optim_params.get("student_freeze_steps", 0) or 0)
    warmup_steps = int(net_optim_params.get("student_lr_warmup_steps", 0) or 0)
    if freeze_steps < 0 or warmup_steps < 0:
        raise ValueError("student_freeze_steps and student_lr_warmup_steps must be >= 0")
    return freeze_steps, warmup_steps


def student_lr_scale(step, freeze_steps, warmup_steps):
    """Student LR factor at zero-based optimizer step ``step``.

    The factor is 0 during the freeze, then rises linearly and reaches 1 on the
    last warmup step.
    """
    if step < freeze_steps:
        return 0.0
    if step < freeze_steps + warmup_steps:
        return float(step - freeze_steps + 1) / float(warmup_steps)
    return 1.0


def _parameter_groups_from_modules(net_optim_params, net, base_lr):
    """Use encoder-provided lr groups while keeping the rest of the policy trainable."""
    # The paper trains every module at the same Adam learning rate.
    backbone_lr_multiplier = float(net_optim_params.get("backbone_lr_multiplier", 1.0))
    head_lr_multiplier = float(net_optim_params.get("head_lr_multiplier", 1.0))
    backbone_weight_decay = net_optim_params.get("backbone_weight_decay", None)
    head_weight_decay = net_optim_params.get("head_weight_decay", None)
    if backbone_weight_decay is not None:
        backbone_weight_decay = float(backbone_weight_decay)
    if head_weight_decay is not None:
        head_weight_decay = float(head_weight_decay)
    grouped = []
    seen = set()

    for module in net.modules():
        if module is net:
            continue
        get_groups = getattr(module, "get_parameter_groups", None)
        if not callable(get_groups):
            continue
        trainable_ids = {id(param) for param in module.parameters() if param.requires_grad}
        if not trainable_ids or trainable_ids & seen:
            continue
        group_kwargs = dict(
            base_lr=float(base_lr),
            backbone_lr_multiplier=backbone_lr_multiplier,
            head_lr_multiplier=head_lr_multiplier,
        )
        if backbone_weight_decay is not None:
            group_kwargs["backbone_weight_decay"] = backbone_weight_decay
        if head_weight_decay is not None:
            group_kwargs["head_weight_decay"] = head_weight_decay
        student_lr_multiplier = net_optim_params.get("student_lr_multiplier", None)
        if student_lr_multiplier is None and any(student_lr_schedule(net_optim_params)):
            # The Student schedule scales a separate "student" group; its LR is unchanged.
            student_lr_multiplier = backbone_lr_multiplier
        if student_lr_multiplier is not None:
            group_kwargs["student_lr_multiplier"] = float(student_lr_multiplier)
        try:
            module_groups = get_groups(**group_kwargs)
        except TypeError:
            if student_lr_multiplier is not None:
                raise  # Never silently drop an explicitly requested Student LR.
            module_groups = get_groups(
                base_lr=float(base_lr),
                backbone_lr_multiplier=backbone_lr_multiplier,
                head_lr_multiplier=head_lr_multiplier,
            )

        for group in module_groups:
            params = [param for param in group.get("params", []) if param.requires_grad and id(param) not in seen]
            if not params:
                continue
            seen.update(id(param) for param in params)
            next_group = dict(group)
            next_group["params"] = params
            group_name = str(next_group.get("name", ""))
            if "weight_decay" not in next_group:
                if group_name == "backbone" and backbone_weight_decay is not None:
                    next_group["weight_decay"] = backbone_weight_decay
                elif group_name == "head" and head_weight_decay is not None:
                    next_group["weight_decay"] = head_weight_decay
            grouped.append(next_group)

    remaining = [param for param in net.parameters() if param.requires_grad and id(param) not in seen]
    if remaining:
        grouped.append({"params": remaining, "lr": float(base_lr), "name": "policy"})

    return grouped if grouped else net.parameters()


def lr_scheduler_from_optim_params(net_optim_params, net, optimizer):
    """
    Helper function to return a LRScheduler from the optim_params
    section of the config for a particular network. Returns None
    if a scheduler is not needed.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines whether a learning rate scheduler is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

        optimizer (torch.optim.Optimizer): optimizer for this net

    Returns:
        lr_scheduler (torch.optim.lr_scheduler or None): learning rate scheduler
    """
    lr_scheduler_type = net_optim_params["learning_rate"].get("scheduler_type", "multistep")
    epoch_schedule = net_optim_params["learning_rate"]["epoch_schedule"]

    # Warmup配置
    warmup_epochs = net_optim_params["learning_rate"].get("warmup_epochs", 0)
    warmup_type = net_optim_params["learning_rate"].get("warmup_type", "linear")  # 'linear' or 'constant'

    lr_scheduler = None

    if lr_scheduler_type == "cosine_warmup":
        total_epochs = int(net_optim_params["learning_rate"].get("total_epochs", 1000))
        warmup_epochs = int(warmup_epochs)
        min_lr_ratio = float(net_optim_params["learning_rate"].get("min_lr_ratio", 0.01))
        total_epochs = max(total_epochs, warmup_epochs + 1)

        if warmup_type == "linear":
            warmup_start_factor = float(
                net_optim_params["learning_rate"].get("warmup_start_factor", 0.01)
            )
        else:
            warmup_start_factor = float(
                net_optim_params["learning_rate"].get("warmup_start_factor", 0.1)
            )

        def cosine_warmup_lambda(epoch):
            epoch = max(0, int(epoch))
            if warmup_epochs > 0 and epoch < warmup_epochs:
                if warmup_type == "linear":
                    return warmup_start_factor + (1.0 - warmup_start_factor) * (
                        epoch / max(1, warmup_epochs)
                    )
                return warmup_start_factor

            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        # LambdaLR applies the same multiplier to each param group's own base_lr.
        # This preserves backbone/head/policy LR ratios across warmup and cosine decay.
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda)

    # === 原有的调度器（保持兼容性）===
    if len(epoch_schedule) > 0:
        if lr_scheduler_type == "linear":
            assert len(epoch_schedule) == 1
            end_epoch = int(epoch_schedule[0])
            end_factor = float(net_optim_params["learning_rate"]["decay_factor"])
            warmup_epochs = int(warmup_epochs)
            if warmup_epochs <= 0:
                return optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=end_factor,
                    total_iters=end_epoch,
                )
            # Linear warmup followed by the linear decay.
            warmup_start = float(net_optim_params["learning_rate"].get("warmup_start_factor", 0.01))
            decay_epochs = max(1, end_epoch - warmup_epochs)

            def linear_warmup_lambda(epoch):
                epoch = max(0, int(epoch))
                if epoch < warmup_epochs:
                    return warmup_start + (1.0 - warmup_start) * epoch / warmup_epochs
                progress = min(1.0, (epoch - warmup_epochs) / decay_epochs)
                return 1.0 + (end_factor - 1.0) * progress

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup_lambda)
        elif lr_scheduler_type == "multistep":
            # 可选：为 MultiStepLR 添加 Warmup
            multistep_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=epoch_schedule,
                gamma=net_optim_params["learning_rate"]["decay_factor"],
            )

            if warmup_epochs > 0:
                if warmup_type == "linear":
                    warmup_scheduler = optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=warmup_epochs
                    )
                else:
                    warmup_scheduler = optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor=0.1,
                        total_iters=warmup_epochs
                    )

                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, multistep_scheduler],
                    milestones=[warmup_epochs]
                )
                return lr_scheduler
            else:
                return multistep_scheduler
        else:
            raise ValueError("Invalid LR scheduler type: {}".format(lr_scheduler_type))

    return lr_scheduler


def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False):
    """
    Backpropagate loss and update parameters for network with
    name @name.

    Args:
        net (torch.nn.Module): network to update

        optim (torch.optim.Optimizer): optimizer to use

        loss (torch.Tensor): loss to use for backpropagation

        max_grad_norm (float): if provided, used to clip gradients

        retain_graph (bool): if True, graph is not freed after backward call

    Returns:
        grad_norms (float): average gradient norms from backpropagation
    """

    # backprop
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    # compute grad norms
    grad_norms = 0.
    for p in net.parameters():
        # only clip gradients for parameters for which requires_grad is True
        if p.grad is not None:
            grad_norms += p.grad.data.norm(2).pow(2).item()

    # step
    optim.step()

    return grad_norms


def rot_6d_to_axis_angle(rot_6d):
    """
    Converts tensor with rot_6d representation to axis-angle representation.
    """
    rot_mat = rotation_6d_to_matrix(rot_6d)
    rot = matrix_to_axis_angle(rot_mat)
    return rot

def rot_6d_to_euler_angles(rot_6d):
    """
    Converts tensor with rot_6d representation to euler representation.
    """
    rot_mat = rotation_6d_to_matrix(rot_6d)
    rot = scipy.spatial.transform.Rotation.from_matrix(rot_mat.numpy())
    rot = rot.as_euler("xyz")
    rot = torch.Tensor(rot)
    return rot

def axis_angle_to_rot_6d(axis_angle):
    """
    Converts tensor with rot_6d representation to axis-angle representation.
    """
    rot_mat = axis_angle_to_matrix(axis_angle)
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d


def euler_angles_to_rot_6d(euler_angles, convention="XYZ"):
    """
    Converts tensor with rot_6d representation to euler representation.
    """
    rot_mat = euler_angles_to_matrix(euler_angles, convention="XYZ")
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d


class dummy_context_mgr():
    """
    A dummy context manager - useful for having conditional scopes (such
    as @maybe_no_grad). Nothing happens in this scope.
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_no_grad(no_grad):
    """
    Args:
        no_grad (bool): if True, the returned context will be torch.no_grad(), otherwise
            it will be a dummy context
    """
    return torch.no_grad() if no_grad else dummy_context_mgr()


"""
The following utility functions were taken from PyTorch3D:
https://github.com/facebookresearch/pytorch3d/blob/d84f274a0822da969668d00e831870fd88327845/pytorch3d/transforms/rotation_conversions.py
"""
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
