#!/usr/bin/env python3
"""Verify that stopping after us8 preserves real checkpoint features bitwise."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor


def timed(call):
    start = time.monotonic()
    result = call()
    return result, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = RobotDIFTStudentFeatureExtractor(
        str(args.checkpoint), str(args.model_repo), device=args.device, use_fp32=True
    )
    unet = model.student_unet
    if not getattr(unet, "supports_requested_feature_keys", False):
        raise RuntimeError("Loaded Student lacks feature-pruning support")
    device = model.student_timestep.device
    dtype = next(unet.parameters()).dtype
    latent = torch.linspace(-0.5, 0.5, 4 * 32 * 32, device=device, dtype=dtype).reshape(1, 4, 32, 32)
    caption = model._prompt_embeds(["press the button"], device, dtype)
    kwargs = {"encoder_hidden_states": caption, "added_cond_kwargs": {}}
    timestep = model.student_timestep.expand(1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.no_grad():
        full, full_s = timed(lambda: unet(latent, timestep, **kwargs))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fast, fast_s = timed(lambda: unet(latent, timestep, requested_feature_keys=("us3", "us6", "us8"), **kwargs))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if "us10" not in full or "us10" in fast:
        raise RuntimeError("Full and fast paths returned the wrong feature taps")
    taps = ("us3", "us6", "us8")
    for key in taps:
        torch.testing.assert_close(fast[key], full[key], rtol=0, atol=0)
    last_block_calls = []
    hook = unet.up_blocks[3].register_forward_hook(
        lambda *_args: last_block_calls.append(True)
    )
    try:
        with torch.no_grad():
            released_maps = model._encode_backbone(
                torch.full((1, 3, 256, 256), 0.25, device=device, dtype=dtype),
                ["press the button"],
            )
    finally:
        hook.remove()
    if last_block_calls or tuple(released_maps) != taps:
        raise RuntimeError("Public encoder did not use the three-tap fast path")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": hashlib.sha256((args.checkpoint / "metadata.json").read_bytes()).hexdigest(),
        "device": str(device),
        "exact_us3_us6_us8_parity": True,
        "public_encoder_skips_last_up_block": True,
        "full_forward_seconds_first_call": full_s,
        "pruned_forward_seconds_first_call": fast_s,
        "timing_scope": "single first-call inference each, not a stable throughput benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
