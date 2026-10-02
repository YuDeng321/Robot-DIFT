#!/usr/bin/env python3
"""Compare a real Stage-II adapter against its full training save on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402

from release.stage2_adapter import load_stage2_adapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", choices=("robot_dift_paper_candidate", "robot_dift_compact_candidate"), required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--check-cache", action="store_true", help="compare repeated cached and uncached Student features")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Runtime adapter parity requires one GPU")
    os.environ["ROBOT_DIFT_STAGE1_ENCODER"] = str(args.stage1_checkpoint.resolve())
    os.environ["ROBOT_DIFT_STAGE1_FUSION_CHECKPOINT"] = str(args.stage1_checkpoint.resolve())
    os.environ["ROBOT_DIFT_CLIP_MODEL"] = str(args.clip_model.resolve())
    os.environ["ROBOT_DIFT_MODEL_DIR"] = str(args.model_repo.resolve())
    os.environ.setdefault("WORK", str(Path.cwd()))

    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = hydra.compose(
            config_name="robocasa_config",
            overrides=[
                "agents=droid_diffusion_agent",
                "agents/model=droid/droid_diffusion_unet_stage2",
                f"agents/obs_encoders={args.readout}",
                f"agents.language_encoders.model_name={args.clip_model.resolve()}",
                "obs_seq_len=2",
                "pred_seq_len=16",
                "act_seq_len=8",
                "obs_tokens=2",
            ],
        )
    agent = hydra.utils.instantiate(cfg.agents)
    manifest = load_stage2_adapter(
        agent, args.adapter,
        stage1_checkpoint=args.stage1_checkpoint,
        clip_model=args.clip_model,
    )
    if any(parameter.requires_grad for parameter in agent.img_encoder.rgb_model.parameters()):
        raise AssertionError("Stage-II Student is not frozen")
    if not any(parameter.requires_grad for parameter in agent.img_encoder.readout.parameters()):
        raise AssertionError("Stage-II readout is not trainable")
    agent.eval()
    torch.manual_seed(919)
    image = torch.rand(1, 2, 3, 128, 128, device="cuda")

    def predict(*, frame_ids: bool = False):
        obs = {f"{name}_image": image.clone() for name in cfg.camera_names}
        obs["lang"] = ["press the coffee machine button"]
        if frame_ids:
            obs["_robot_dift_frame_ids"] = torch.tensor([[[123, 0, 0], [123, 0, 1]]])
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)
        with torch.inference_mode():
            return agent(obs).detach().float().cpu()

    adapter_action = predict()
    agent.load_pretrained_model(str(args.full_checkpoint))
    agent.eval()
    full_action = predict()
    torch.testing.assert_close(adapter_action, full_action, rtol=0, atol=0)
    if not torch.isfinite(adapter_action).all():
        raise AssertionError("Nonfinite Stage-II policy action")
    cache_items = None
    eviction_parity = None
    if args.check_cache:
        agent.img_encoder.feature_cache_max_bytes = 1 << 30
        agent.img_encoder._feature_cache.clear()
        agent.img_encoder._feature_cache_bytes = 0
        cached_first = predict(frame_ids=True)
        cache_items = len(agent.img_encoder._feature_cache)
        cached_second = predict(frame_ids=True)
        if len(agent.img_encoder._feature_cache) != cache_items or cache_items != 6:
            raise AssertionError("Frozen Student feature cache did not retain six camera/frame entries")
        uncached = predict()
        torch.testing.assert_close(cached_first, cached_second, rtol=0, atol=0)
        torch.testing.assert_close(cached_first, uncached, rtol=0, atol=0)
        first_item = next(iter(agent.img_encoder._feature_cache.values()))
        item_bytes = sum(value.numel() * value.element_size() for value in first_item.values())
        agent.img_encoder.feature_cache_max_bytes = 2 * item_bytes
        agent.img_encoder._feature_cache.clear()
        agent.img_encoder._feature_cache_bytes = 0
        evicted = predict(frame_ids=True)
        torch.testing.assert_close(evicted, uncached, rtol=0, atol=0)
        eviction_parity = True
    print(json.dumps({
        "readout": args.readout,
        "adapter_manifest_pooling": manifest["token_pooling"],
        "action_shape": list(adapter_action.shape),
        "max_abs_diff": float((adapter_action - full_action).abs().max()),
        "student_frozen": True,
        "readout_trainable": True,
        "cache_items": cache_items,
        "eviction_parity": eviction_parity,
    }, indent=2))


if __name__ == "__main__":
    main()
