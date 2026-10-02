"""The public weight package must not expose training-machine paths."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_export_preserves_settings_without_local_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint-1000-ema"
    (checkpoint / "unet").mkdir(parents=True)
    (checkpoint / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "timestep.bin").write_bytes(b"timestep")
    (checkpoint / "deploy_head.safetensors").write_bytes(b"head")
    model_repo = tmp_path / "sd21"
    model_repo.mkdir()
    (model_repo / "model_index.json").write_text("{}", encoding="utf-8")
    private_root = "/home/private_user/experiment"
    config = {
        "train": {"data_path": f"{private_root}/droid", "batch_size": 8},
        "experiment": {"save_cleandift_dir": f"{private_root}/encoder"},
        "algo": {"robot_dift_paper_readout": {"clip_model_path": f"{private_root}/clip.pt"}},
    }
    metadata = {
        "model_type": "cleandift",
        "sd_version": "sd21",
        "sd_model_repo": str(model_repo),
        "feature_key": ["us3", "us6", "us8"],
        "components": {"student_unet": "unet/", "adapters": "adapters.bin"},
        "training_config": json.dumps(config),
        "source_git_revision": "abc123",
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    package = tmp_path / "public_encoder"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/release/export_encoder_release.py"),
         "--checkpoint", str(checkpoint), "--model-repo", str(model_repo), "--output", str(package)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/release/verify_encoder_release.py"),
         "--package", str(package), "--model-repo", str(model_repo)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    for name in ("metadata.json", "training_metadata.json", "release_manifest.json"):
        public_text = (package / name).read_text(encoding="utf-8")
        assert private_root not in public_text
        assert str(tmp_path) not in public_text
    public = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    public_config = json.loads(public["training_config"])
    assert public_config["train"]["batch_size"] == 8
    assert public_config["train"]["data_path"] == "${ROBOT_DIFT_DATA_ROOT}"
    assert public["components"]["adapters"] is None
