#!/usr/bin/env python3
"""Extract raw Robot-DIFT maps from one or more RGB camera images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Exported Student directory")
    parser.add_argument("--model-repo", required=True, type=Path, help="Local SD2.1 Diffusers snapshot")
    parser.add_argument("--images", required=True, nargs="+", type=Path, help="Camera images, in output batch order")
    parser.add_argument("--prompt", default="", help="Instruction shared by all images; defaults to empty text")
    parser.add_argument("--image-size", type=int, default=256, help="Square RGB input size (default: 256)")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda or cpu")
    parser.add_argument("--output", type=Path, help="Optional .pt file for CPU feature maps and input settings")
    args = parser.parse_args(argv)
    if args.image_size < 64 or args.image_size % 64:
        parser.error("--image-size must be a positive multiple of 64")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    return args


def load_images(paths, image_size):
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    images = []
    for path in paths:
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            resized = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True)).permute(2, 0, 1))
    return torch.stack(images)


def main(argv=None):
    args = parse_args(argv)
    # Keep --help usable without installing model or training dependencies.
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.encoders.robot_dift_student_feature_extractor import RobotDIFTStudentFeatureExtractor

    images = load_images(args.images, args.image_size)
    encoder = RobotDIFTStudentFeatureExtractor(
        checkpoint_dir=str(args.checkpoint), model_repo=str(args.model_repo), device=args.device,
    )
    maps = encoder(images, args.prompt, input_range="uint8")
    print(f"One shared Student; VAE latent mode: {encoder.vae_latent_mode}")
    for key, value in maps.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "features": {key: value.cpu() for key, value in maps.items()},
            "image_names": [path.name for path in args.images],
            "prompt": args.prompt,
            "image_size": args.image_size,
            "vae_latent_mode": encoder.vae_latent_mode,
        }, args.output)
        print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
