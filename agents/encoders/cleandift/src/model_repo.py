"""Resolve the one Stable Diffusion snapshot shared by every Robot-DIFT component."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REPO_ENV_VARS = ("ROBOT_DIFT_CLEANDIFT_MODEL_REPO", "ROBOT_DIFT_MODEL_DIR")


def _normalize(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def resolve_sd_model_repo(default_repo: str) -> str:
    """Return the configured SD2.1 snapshot, or ``default_repo`` when none is set.

    The VAE, text encoder, Teacher, and Student must come from one snapshot.
    Both environment variables are accepted, but they may not disagree.
    """
    configured = {name: os.environ[name] for name in MODEL_REPO_ENV_VARS if os.environ.get(name)}
    if len({_normalize(value) for value in configured.values()}) > 1:
        raise ValueError(
            "ROBOT_DIFT_CLEANDIFT_MODEL_REPO and ROBOT_DIFT_MODEL_DIR point to different "
            f"Stable Diffusion snapshots: {configured}"
        )
    return next(iter(configured.values()), default_repo)
