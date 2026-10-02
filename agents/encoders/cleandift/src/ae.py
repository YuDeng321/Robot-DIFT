import os

from .compat import patch_transformers_for_diffusers_compat
from .model_repo import resolve_sd_model_repo

patch_transformers_for_diffusers_compat()
import diffusers
import torch
from torch import nn

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        scale: float = 0.18215,
        shift: float = 0.0,
        repo="sd2-community/stable-diffusion-2-1-base",
        latent_mode: str = "sample",
    ):
        super().__init__()
        if latent_mode not in {"sample", "mode"}:
            raise ValueError("latent_mode must be 'sample' or 'mode'")
        repo = resolve_sd_model_repo(repo)
        compile_enabled = os.environ.get("TORCHDYNAMO_DISABLE", "").lower() not in {"1", "true"}
        self.scale = scale
        self.shift = shift
        self.latent_mode = latent_mode
        self.ae = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae")
        self.ae.eval()
        if compile_enabled:
            try:
                self.ae.compile()
            except Exception:
                pass
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img)

    @torch.no_grad()
    def encode(self, img):
        posterior = self.ae.encode(img, return_dict=False)[0]
        latent = posterior.sample() if self.latent_mode == "sample" else posterior.mode()
        return (latent - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, latent):
        rec = self.ae.decode(latent / self.scale + self.shift, return_dict=False)[0]
        return rec
