"""Paper Stage-I visual path: trainable Student, Teacher alignment, S2-FPN + CLIP readout.

Per observation frame, every camera view runs through one shared Student U-Net.
The S2-FPN/CLIP readout (:class:`RobotDIFTPaperReadout`, the same module the
Stage-II policies train) fuses the raw ``us3/us6/us8`` maps of all views into a
single policy token. The frozen CLIP ViT-B/32 text tower provides both the
readout queries and the policy's language goal.

During training the current frame's Student pass also feeds the Teacher
alignment loss, so the Student runs once per image. The alignment terms only
read that pass, so the result equals separate policy and alignment passes on
the same latents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from agents.encoders.frozen_clip_text import CLIP_GOAL_DIM, FrozenCLIPTextTokens
from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout

PAPER_FEATURE_KEYS = ("us3", "us6", "us8")


class RobotDIFTStage1Encoder(nn.Module):
    """Multi-view Stage-I encoder returning one readout token per frame.

    Args:
        rgb_keys: Camera observation keys. Order does not matter: views are
            max-pooled inside the readout and carry no camera embedding.
        student: A backbone-only :class:`CleanDIFTImgEncoder`, or the keyword
            arguments to build one (``readout='none'`` and ``feature_key`` are set here).
        clip_model_path: Local CLIP ViT-B/32 weight file.
        feature_keys: Coarse-to-fine Student maps read by the S2-FPN.
        text_encoder: Optional injected frozen text tower (tests).
    """

    def __init__(
        self,
        rgb_keys: Sequence[str],
        student,
        clip_model_path: str | None = None,
        *,
        feature_keys: Sequence[str] = PAPER_FEATURE_KEYS,
        output_dim: int = 512,
        fpn_dim: int = 256,
        model_dim: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 1,
        mlp_hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.0,
        token_pooling: str = "flatten",
        text_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.rgb_keys = tuple(sorted(str(key) for key in rgb_keys))
        if not self.rgb_keys:
            raise ValueError("RobotDIFTStage1Encoder requires at least one RGB view")
        self.feature_keys = tuple(str(key) for key in feature_keys)
        if isinstance(student, Mapping):
            from agents.encoders.cleandift_img_encoder import CleanDIFTImgEncoder

            kwargs = dict(student)
            kwargs.update(feature_key=list(self.feature_keys), readout="none")
            student = CleanDIFTImgEncoder(**kwargs)
        self.student = student
        self.text_encoder = text_encoder if text_encoder is not None else FrozenCLIPTextTokens(clip_model_path)
        channels = {key: int(self.student.feature_dims[key]) for key in self.feature_keys}
        self.readout = RobotDIFTPaperReadout(
            channels,
            feature_keys=self.feature_keys,
            fpn_dim=int(fpn_dim),
            model_dim=int(model_dim),
            output_dim=int(output_dim),
            num_heads=int(num_heads),
            transformer_layers=int(transformer_layers),
            mlp_hidden_dims=tuple(int(width) for width in mlp_hidden_dims),
            dropout=float(dropout),
            token_pooling=token_pooling,
        )
        self.output_dim = int(output_dim)
        self.goal_dim = int(getattr(self.text_encoder, "goal_dim", CLIP_GOAL_DIM))
        self._readout_config = {
            "feature_keys": list(self.feature_keys),
            "output_dim": self.output_dim,
            "fpn_dim": int(fpn_dim),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "transformer_layers": int(transformer_layers),
            "mlp_hidden_dims": [int(width) for width in mlp_hidden_dims],
            "dropout": float(dropout),
            "token_pooling": token_pooling,
            "goal_dim": self.goal_dim,
        }

    # ------------------------------------------------------------------
    # Interfaces used by the robomimic diffusion policy
    # ------------------------------------------------------------------

    def output_shape(self, input_shape=None):
        return [self.output_dim]

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    @staticmethod
    def _captions(lang_cond, batch: int) -> list[str]:
        if lang_cond is None:
            return [""] * batch
        if isinstance(lang_cond, str):
            return [lang_cond] * batch
        captions = [str(value) for value in lang_cond]
        if len(captions) != batch:
            raise ValueError(f"Expected {batch} language instructions, got {len(captions)}")
        return captions

    def forward(self, obs: Mapping[str, Tensor], lang_cond=None, **_unused) -> Tensor:
        """Encode one frame: ``obs[key]`` is ``[B, 3, H, W]``; returns ``[B, output_dim]``."""
        features, _, _ = self.encode_sequence({key: obs[key][:, None] for key in self.rgb_keys}, lang_cond)
        return features[:, 0]

    def encode_sequence(
        self,
        obs: Mapping[str, Tensor],
        lang_cond=None,
        alignment: bool = False,
        return_raw_cosine: bool = False,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """Encode ``obs[key]`` of shape ``[B, T, 3, H, W]`` (images in [-1, 1]).

        Returns ``(features [B, T, output_dim], alignment_loss, metrics)``.
        With ``alignment=True`` the loss uses the current frame (``T-1``) of
        every view, i.e. one Teacher timestep per view, as in the paper.
        """
        missing = [key for key in self.rgb_keys if key not in obs]
        if missing:
            raise KeyError(f"Missing RGB views: {missing}")
        images = torch.stack([obs[key] for key in self.rgb_keys], dim=1)  # [B, V, T, 3, H, W]
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError(f"RGB views must have shape [B, T, 3, H, W], got {tuple(images.shape[:1] + images.shape[2:])}")
        batch, views, frames = images.shape[:3]
        image_shape = images.shape[3:]
        captions = self._captions(lang_cond, batch)

        def per_image(repeats: int) -> list[str]:
            return [caption for caption in captions for _ in range(repeats)]

        loss = None
        metrics: dict[str, Tensor] = {}
        if alignment:
            current, loss, metrics = self.student.encode_student(
                images[:, :, -1].reshape(batch * views, *image_shape),
                per_image(views),
                list(self.feature_keys),
                alignment=True,
                return_raw_cosine=return_raw_cosine,
            )
            maps = {key: value.reshape(batch, views, 1, *value.shape[1:]) for key, value in current.items()}
            if frames > 1:
                history, _, _ = self.student.encode_student(
                    images[:, :, :-1].reshape(batch * views * (frames - 1), *image_shape),
                    per_image(views * (frames - 1)),
                    list(self.feature_keys),
                )
                maps = {
                    key: torch.cat(
                        (history[key].reshape(batch, views, frames - 1, *history[key].shape[1:]), maps[key]),
                        dim=2,
                    )
                    for key in self.feature_keys
                }
        else:
            flat, _, _ = self.student.encode_student(
                images.reshape(batch * views * frames, *image_shape),
                per_image(views * frames),
                list(self.feature_keys),
            )
            maps = {key: value.reshape(batch, views, frames, *value.shape[1:]) for key, value in flat.items()}

        view_maps = [
            {key: maps[key][:, view].reshape(batch * frames, *maps[key].shape[3:]) for key in self.feature_keys}
            for view in range(views)
        ]
        text_tokens, text_mask = self.text_encoder(captions)
        text_tokens = text_tokens.repeat_interleave(frames, dim=0)
        text_mask = text_mask.repeat_interleave(frames, dim=0)
        features = self.readout(view_maps, text_tokens, text_mask)
        return features.reshape(batch, frames, -1), loss, metrics

    @torch.no_grad()
    def encode_language_goal(self, lang_cond, batch: int | None = None) -> Tensor:
        """Frozen CLIP end-of-text embedding ``[B, goal_dim]`` for the policy goal."""
        if batch is None:
            batch = 1 if isinstance(lang_cond, str) or lang_cond is None else len(lang_cond)
        goal = self.text_encoder.encode_goal(self._captions(lang_cond, batch))
        return goal.reshape(goal.shape[0], -1).float()

    def get_parameter_groups(
        self,
        base_lr: float,
        backbone_lr_multiplier: float = 1.0,
        head_lr_multiplier: float = 1.0,
        backbone_weight_decay: float | None = None,
        head_weight_decay: float | None = None,
        student_lr_multiplier: float | None = None,
        **_unused,
    ) -> list[dict]:
        """Student/adapters (``backbone``) and S2-FPN/CLIP readout (``head``) groups."""
        groups = self.student.get_parameter_groups(
            base_lr=base_lr,
            backbone_lr_multiplier=backbone_lr_multiplier,
            head_lr_multiplier=head_lr_multiplier,
            backbone_weight_decay=backbone_weight_decay,
            head_weight_decay=head_weight_decay,
            student_lr_multiplier=student_lr_multiplier,
        )
        readout = [parameter for parameter in self.readout.parameters() if parameter.requires_grad]
        if readout:
            groups.append({
                "params": readout,
                "lr": base_lr * head_lr_multiplier,
                "name": "head",
                **({"weight_decay": head_weight_decay} if head_weight_decay is not None else {}),
            })
        return groups

    def readout_state(self) -> dict[str, Tensor]:
        """Detached CPU copy of the S2-FPN/CLIP readout for the deploy head."""
        return {
            f"readout.{key}": value.detach().cpu().contiguous()
            for key, value in self.readout.state_dict().items()
        }

    def readout_config(self) -> dict:
        """Constructor settings needed to rebuild the readout downstream."""
        return dict(self._readout_config)
