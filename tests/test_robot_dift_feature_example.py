"""Exercise the portable image example without downloading model weights."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "extract_features.py"
SPEC = importlib.util.spec_from_file_location("robot_dift_example", EXAMPLE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_help_needs_no_model_or_training_dependencies():
    result = subprocess.run([sys.executable, "-S", str(EXAMPLE), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--checkpoint" in result.stdout
    assert "--images" in result.stdout


def test_image_loading_preserves_rgb_and_camera_batch_order(tmp_path):
    first, second = tmp_path / "front.png", tmp_path / "wrist.png"
    Image.fromarray(np.full((20, 30, 3), [255, 0, 0], dtype=np.uint8)).save(first)
    Image.fromarray(np.full((20, 30, 3), [0, 0, 255], dtype=np.uint8)).save(second)
    images = example.load_images([first, second], 64)
    assert images.shape == (2, 3, 64, 64)
    assert images.dtype == torch.uint8
    assert images[0, :, 0, 0].tolist() == [255, 0, 0]
    assert images[1, :, 0, 0].tolist() == [0, 0, 255]


def test_cli_rejects_existing_output_and_invalid_size(tmp_path):
    image = tmp_path / "camera.png"
    Image.new("RGB", (64, 64)).save(image)
    output = tmp_path / "features.pt"
    output.write_bytes(b"existing")
    base = ["--checkpoint", "checkpoint", "--model-repo", "sd21", "--images", str(image)]
    with pytest.raises(SystemExit):
        example.parse_args(base + ["--output", str(output)])
    assert output.read_bytes() == b"existing"
    with pytest.raises(SystemExit):
        example.parse_args(base + ["--image-size", "65"])
