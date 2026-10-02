"""Export Robot-DIFT Stage-I Student snapshots (DIFT component format + deploy head).

Kept free of TensorFlow/Octo imports so exports can be tested and reused
outside the DROID training loop.
"""

import gc
import json
import os
import subprocess
from collections import OrderedDict

import torch
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_parallel(module):
    if module is None:
        return None
    return module.module if hasattr(module, "module") else module


def find_cleandift_encoder(root_encoder):
    """Return ``(student_encoder, source)`` for the paper or legacy visual encoder."""
    root_encoder = unwrap_parallel(root_encoder)

    paper_student = unwrap_parallel(getattr(root_encoder, "student", None))
    if paper_student is not None and hasattr(paper_student, "model") and hasattr(root_encoder, "readout"):
        return paper_student, "paper_stage1"

    direct_candidate = unwrap_parallel(getattr(root_encoder, "encoder", None))
    if direct_candidate is not None and hasattr(direct_candidate, "model"):
        return direct_candidate, "direct"

    if hasattr(root_encoder, "nets"):
        for group_name, group_encoder in root_encoder.nets.items():
            group_encoder = unwrap_parallel(group_encoder)
            if not hasattr(group_encoder, "obs_nets"):
                continue
            for modality_name, modality_encoder in group_encoder.obs_nets.items():
                modality_encoder = unwrap_parallel(modality_encoder)

                candidate = unwrap_parallel(getattr(modality_encoder, "encoder", None))
                if candidate is not None and hasattr(candidate, "model"):
                    return candidate, f"{group_name}/{modality_name}"

                backbone = unwrap_parallel(getattr(modality_encoder, "backbone", None))
                if backbone is None:
                    continue

                candidate = unwrap_parallel(getattr(backbone, "encoder", None))
                if candidate is not None and hasattr(candidate, "model"):
                    return candidate, f"{group_name}/{modality_name}"

    return None, None


def git_revision():
    """Return ``(commit, dirty)`` of the source tree, or ``(None, None)``."""
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, dirty


def save_robot_dift_snapshot(
    epoch_dir,
    *,
    student,
    readout_owner,
    source,
    config,
    epoch,
    ema,
    save_full_state=False,
    save_full_encoder=False,
):
    """Write one Stage-I Student snapshot (DIFT component format + deploy head)."""
    from agents.encoders.robot_dift_deploy_head import DEPLOY_HEAD_FILE, collect_deploy_head_state

    feature_aligner = student.model
    os.makedirs(os.path.join(epoch_dir, "unet"), exist_ok=True)
    torch.save(
        feature_aligner.unet_feature_extractor_cleandift.state_dict(),
        os.path.join(epoch_dir, "unet", "diffusion_pytorch_model.bin"),
    )
    components = {
        "student_unet": "unet/",
        "timestep": "timestep.bin",
        "deploy_head": DEPLOY_HEAD_FILE,
        "adapters": None,
        "mapping_network": None,
        "full_state_dict": None,
        "robot_dift_encoder_state": None,
    }
    # Training-only alignment modules are saved for resuming/inspection; the
    # release export drops them.
    if getattr(feature_aligner, "adapters", None) is not None:
        torch.save(feature_aligner.adapters.state_dict(), os.path.join(epoch_dir, "adapters.bin"))
        components["adapters"] = "adapters.bin"
    if getattr(feature_aligner, "mapping", None) is not None:
        torch.save(feature_aligner.mapping.state_dict(), os.path.join(epoch_dir, "mapping_network.bin"))
        torch.save(feature_aligner.time_emb.state_dict(), os.path.join(epoch_dir, "time_emb.bin"))
        torch.save(feature_aligner.time_in_proj.state_dict(), os.path.join(epoch_dir, "time_in_proj.bin"))
        components["mapping_network"] = "mapping_network.bin"
    torch.save({"timestep": feature_aligner.timestep.data}, os.path.join(epoch_dir, "timestep.bin"))
    save_file(collect_deploy_head_state(readout_owner), os.path.join(epoch_dir, DEPLOY_HEAD_FILE))

    def cpu_float_state(state):
        return OrderedDict(
            (key, value.detach().cpu().float() if value.dtype in (torch.float16, torch.bfloat16) else value.detach().cpu())
            for key, value in state.items()
        )

    if save_full_state:
        state = OrderedDict((key, value.contiguous()) for key, value in cpu_float_state(feature_aligner.state_dict()).items())
        save_file(state, os.path.join(epoch_dir, "cleandift_full_state.safetensors"))
        components["full_state_dict"] = "cleandift_full_state.safetensors"
        del state
    if save_full_encoder:
        torch.save(
            {
                "state_dict": cpu_float_state(readout_owner.state_dict()),
                "epoch": epoch,
                "ema": bool(ema),
                "model_type": "robot_dift_encoder",
                "source_modality": source,
            },
            os.path.join(epoch_dir, "robot_dift_encoder_state.pt"),
        )
        components["robot_dift_encoder_state"] = "robot_dift_encoder_state.pt"
    gc.collect()

    paper_readout = callable(getattr(readout_owner, "readout_config", None))
    revision, dirty = git_revision()
    metadata = {
        "epoch": epoch,
        "ema": bool(ema),
        "model_type": "cleandift",
        "source_modality": source,
        "sd_version": getattr(student, "sd_version", "sd21"),
        "sd_model_repo": getattr(feature_aligner, "repo", None),
        "feature_key": list(getattr(student, "feature_keys", [])),
        "alignment_feature_keys": list(getattr(student, "alignment_feature_keys", [])),
        "apply_feature_adapter": getattr(student, "apply_feature_adapter", None),
        "alignment_apply_feature_adapter": getattr(student, "alignment_apply_feature_adapter", None),
        "alignment_layer_reduction": getattr(student, "alignment_layer_reduction", "mean"),
        "student_init": getattr(student, "student_init", None),
        "readout": "paper" if paper_readout else "legacy",
        "readout_config": readout_owner.readout_config() if paper_readout else None,
        "fusion_mode": "paper" if paper_readout else getattr(student, "fusion_mode", "s2fpn"),
        "output_mode": None if paper_readout else getattr(student, "output_mode", None),
        "map_out_dim": None if paper_readout else getattr(student, "map_out_dim", None),
        "freeze_backbone": getattr(student, "freeze_backbone", True),
        "use_text_condition": getattr(student, "use_text_condition", False),
        "vae_latent_mode": getattr(student, "vae_latent_mode", "sample"),
        "feature_dims": getattr(student, "feature_dims", {}),
        "student_timestep": float(feature_aligner.timestep.detach().float().cpu()),
        "source_git_revision": revision,
        "source_worktree_dirty": dirty,
        "training_config": config.dump(),
        "components": components,
    }
    with open(os.path.join(epoch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def export_robot_dift_encoder(model, config, epoch, save_dir, *, save_ema, trace=lambda phase: None):
    """Export the raw and (optionally) EMA Stage-I Student. Returns saved directories."""
    nets = model.nets.module if isinstance(model.nets, DDP) else model.nets
    obs_encoder = unwrap_parallel(nets["policy"]["obs_encoder"])
    student, source = find_cleandift_encoder(obs_encoder)
    if student is None:
        raise RuntimeError("Robot-DIFT encoder export could not locate a CleanDIFT Student")
    readout_owner = obs_encoder if source == "paper_stage1" else student
    save_full_encoder = bool(getattr(config.experiment, "save_robot_dift_full_encoder", False))
    if save_full_encoder and source == "paper_stage1":
        # The deploy head already holds the paper readout; a full-encoder file
        # of the Stage-I wrapper would not load into CleanDIFTImgEncoder.
        raise ValueError("save_robot_dift_full_encoder applies to the legacy readout only")
    save_kwargs = dict(
        student=student,
        readout_owner=readout_owner,
        source=source,
        config=config,
        epoch=epoch,
        save_full_state=bool(getattr(config.experiment, "save_cleandift_full_state", False)),
        save_full_encoder=save_full_encoder,
    )
    saved = []
    raw_dir = os.path.join(save_dir, f"checkpoint-{epoch}")
    save_robot_dift_snapshot(raw_dir, ema=False, **save_kwargs)
    saved.append(raw_dir)
    trace("after_raw_encoder_save")

    if save_ema and getattr(model, "ema", None) is not None:
        parameters = model.ema_parameters()
        model.ema.store(parameters)
        model.ema.copy_to(parameters)
        try:
            ema_dir = os.path.join(save_dir, f"checkpoint-{epoch}-ema")
            save_robot_dift_snapshot(ema_dir, ema=True, **save_kwargs)
            saved.append(ema_dir)
            trace("after_ema_encoder_save")
        finally:
            model.ema.restore(parameters)
            trace("after_ema_restore")
    return saved
