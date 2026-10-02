import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | int, global_cond: Optional[torch.Tensor] = None):
        sample = sample.moveaxis(-1, -2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


class DroidDiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_seq_len: int,
        obs_tokens: int,
        latent_dim: int,
        goal_dim: int,
        use_goal_cond: bool = True,
        noise_samples: int = 1,
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 100,
        num_inference_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.obs_tokens = obs_tokens
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.use_goal_cond = bool(use_goal_cond)
        self.noise_samples = max(1, int(noise_samples))
        self.num_inference_timesteps = int(num_inference_timesteps)

        global_cond_dim = obs_tokens * latent_dim + (goal_dim if self.use_goal_cond else 0)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        scheduler_kwargs = {
            "num_train_timesteps": int(num_train_timesteps),
            "beta_schedule": beta_schedule,
            "clip_sample": clip_sample,
            "prediction_type": prediction_type,
        }
        if scheduler_type.lower() == "ddim":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            self.noise_scheduler = DDPMScheduler(**scheduler_kwargs)

    def _flatten_goal(self, latent_goal: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if latent_goal is None or not self.use_goal_cond:
            return None
        if latent_goal.dim() == 3:
            latent_goal = latent_goal.flatten(start_dim=1)
        return latent_goal

    def _global_condition(self, perceptual_emb: torch.Tensor, latent_goal: Optional[torch.Tensor]) -> torch.Tensor:
        if perceptual_emb.dim() == 3:
            obs_cond = perceptual_emb.flatten(start_dim=1)
        else:
            obs_cond = perceptual_emb
        goal_cond = self._flatten_goal(latent_goal)
        if goal_cond is None:
            return obs_cond
        return torch.cat([obs_cond, goal_cond], dim=-1)

    def _diffusion_loss(
        self,
        actions: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=actions.device
        ).long()

        noise = torch.randn(
            (self.noise_samples, batch_size) + actions.shape[1:], device=actions.device
        )
        actions_rep = actions.repeat(self.noise_samples, 1, 1)
        timesteps_rep = timesteps.repeat(self.noise_samples)
        noise_rep = noise.view(self.noise_samples * batch_size, *actions.shape[1:])

        noisy_actions = self.noise_scheduler.add_noise(actions_rep, noise_rep, timesteps_rep)
        global_cond_rep = global_cond.repeat(self.noise_samples, 1)

        noise_pred = self.noise_pred_net(noisy_actions, timesteps_rep, global_cond=global_cond_rep)
        loss = F.mse_loss(noise_pred, noise_rep)
        return loss

    @torch.no_grad()
    def _sample_actions(self, global_cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=device)
        sample = torch.randn(
            (global_cond.shape[0], self.action_seq_len, self.action_dim), device=device
        )

        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(sample, timestep, global_cond=global_cond)
            sample = self.noise_scheduler.step(noise_pred, timestep, sample).prev_sample

        return sample

    def forward(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        if_train: bool = False,
    ) -> torch.Tensor:
        global_cond = self._global_condition(perceptual_emb, latent_goal)
        if if_train:
            if action is None:
                raise ValueError("Training requires action tensor.")
            return self._diffusion_loss(action, global_cond)
        return self._sample_actions(global_cond, device=perceptual_emb.device)
