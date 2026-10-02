"""Frozen CLIP ViT-B/32 text tower shared by the Robot-DIFT readout and policy goal.

Stage I (DROID) and Stage II (RoboCasa/LIBERO) use the same text interface:
all 77 hidden tokens act as readout queries, and the end-of-text embedding
projected by CLIP conditions the diffusion policy as its language goal.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from agents.models.beso.models.networks.clip import load_clip, tokenize

CLIP_GOAL_DIM = 512


class FrozenCLIPTextTokens(nn.Module):
    """Return all 77 hidden tokens from a local, frozen CLIP ViT-B/32 model."""

    def __init__(self, model_path: str):
        super().__init__()
        if not model_path:
            raise ValueError("model_path is required for the frozen CLIP text encoder")
        clip, _ = load_clip(model_path, device="cpu", jit=False)
        # The image tower is unused by both the readout and the pooled policy
        # goal. Register only the text tower so it is not copied to GPU or saved
        # in every downstream policy checkpoint.
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.requires_grad_(False)
        self.eval()

    @property
    def goal_dim(self) -> int:
        return int(self.text_projection.shape[-1])

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def _encode(self, captions: Sequence[str]) -> tuple[Tensor, Tensor, Tensor]:
        token_ids = tokenize(list(captions), context_length=77, truncate=True).to(
            self.token_embedding.weight.device
        )
        mask = token_ids.ne(0)
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.positional_embedding.to(dtype=hidden.dtype)
        hidden = self.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
        hidden = self.ln_final(hidden).float()
        return hidden, mask, token_ids

    @torch.no_grad()
    def forward(self, captions: Sequence[str]) -> tuple[Tensor, Tensor]:
        hidden, mask, _ = self._encode(captions)
        return hidden, mask

    @torch.no_grad()
    def encode_goal(self, captions: Sequence[str]) -> Tensor:
        """Match CLIP's end-of-text pooled embedding for policy conditioning."""
        hidden, _, token_ids = self._encode(captions)
        eot = token_ids.argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.shape[0], device=eot.device), eot]
        return (pooled @ self.text_projection.float())[:, None, :]
