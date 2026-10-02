"""Student weight interpolation (WiSE-FT) writes loadable Stage-I checkpoints."""

import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from agents.encoders.robot_dift_deploy_head import validate_deploy_head
from agents.encoders.robot_dift_student_feature_extractor import (
    _read_stage1_metadata,
    _read_student_state,
    _read_student_timestep,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("interpolate_student", ROOT / "scripts/release/interpolate_student.py")
interpolate_student = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(interpolate_student)

FEATURE_DIMS = {"us3": 1280, "us6": 1280, "us8": 640}


def _unet_state(seed):
    generator = torch.Generator().manual_seed(seed)
    return OrderedDict(
        (key, torch.randn(*shape, generator=generator))
        for key, shape in (("conv_in.weight", (4, 4, 3, 3)), ("conv_in.bias", (4,)), ("norm.weight", (4,)))
    )


def _student_checkpoint(path, unet_state, timestep, student_init="sd_teacher"):
    (path / "unet").mkdir(parents=True)
    torch.save(unet_state, path / "unet" / "diffusion_pytorch_model.bin")
    torch.save({"timestep": torch.tensor(timestep)}, path / "timestep.bin")
    torch.save({"scale": torch.ones(2)}, path / "adapters.bin")
    save_file({"readout.lateral.us3.0.weight": torch.ones(2, 2)}, str(path / "deploy_head.safetensors"))
    metadata = {
        "model_type": "cleandift", "sd_version": "sd21", "use_text_condition": True, "ema": True,
        "student_init": student_init, "readout": "paper", "feature_key": list(FEATURE_DIMS),
        "feature_dims": FEATURE_DIMS, "vae_latent_mode": "mode", "student_timestep": timestep,
        "components": {"student_unet": "unet/", "timestep": "timestep.bin",
                       "deploy_head": "deploy_head.safetensors", "adapters": "adapters.bin"},
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


@pytest.fixture
def trained(tmp_path):
    repo = tmp_path / "sd21"
    (repo / "unet").mkdir(parents=True)
    init = _unet_state(0)
    save_file(dict(init), str(repo / "unet" / "diffusion_pytorch_model.safetensors"))
    student = OrderedDict((key, value + 0.1 * torch.randn_like(value)) for key, value in init.items())
    checkpoint = _student_checkpoint(tmp_path / "checkpoint-300-ema", student, 301.0)
    return repo, checkpoint, init, student


def test_endpoints_are_exact_and_outputs_load_like_stage1_checkpoints(trained, tmp_path):
    repo, checkpoint, init, student = trained
    out = tmp_path / "interpolated"
    assert interpolate_student.main([
        "--checkpoint", str(checkpoint), "--model-repo", str(repo), "--alpha", "0", "0.5", "1",
        "--output-root", str(out),
    ]) == 0
    t0 = interpolate_student.sd_teacher_initial_timestep()
    assert t0 == 261.0
    for alpha, expected_t in ((0.0, t0), (0.5, (t0 + 301.0) / 2), (1.0, 301.0)):
        path = out / f"checkpoint-300-ema-alpha{alpha:.2f}"
        state = _read_student_state(path)
        for key in student:
            expected = init[key] * (1 - alpha) + student[key] * alpha
            assert torch.equal(state[key], expected) if alpha in (0.0, 1.0) else torch.allclose(state[key], expected)
        assert _read_student_timestep(path).item() == pytest.approx(expected_t)
        assert _read_stage1_metadata(path, FEATURE_DIMS) == "mode"
        assert validate_deploy_head(path)
        assert (path / "deploy_head.safetensors").read_bytes() == (checkpoint / "deploy_head.safetensors").read_bytes()
        assert not (path / "adapters.bin").exists()
        metadata = json.loads((path / "metadata.json").read_text())
        assert metadata["components"]["adapters"] is None
        assert metadata["weight_interpolation"]["alpha"] == alpha
        assert metadata["weight_interpolation"]["init"] == f"sd21:{repo.resolve()}"
        assert metadata["student_timestep"] == pytest.approx(expected_t)
    assert torch.equal(_read_student_state(out / "checkpoint-300-ema-alpha0.00")["conv_in.weight"],
                       init["conv_in.weight"])


def test_non_sd21_initialization_needs_its_checkpoint(tmp_path):
    init = _unet_state(1)
    student = OrderedDict((key, value * 1.5) for key, value in init.items())
    checkpoint = _student_checkpoint(tmp_path / "trained", student, 250.0, student_init="cleandift")
    with pytest.raises(ValueError, match="--init-checkpoint"):
        interpolate_student.main(["--checkpoint", str(checkpoint), "--alpha", "0.5",
                                  "--output-root", str(tmp_path / "a")])
    init_checkpoint = _student_checkpoint(tmp_path / "cleandift", init, 261.0)
    interpolate_student.main(["--checkpoint", str(checkpoint), "--init-checkpoint", str(init_checkpoint),
                              "--alpha", "0.5", "--output-root", str(tmp_path / "b")])
    path = tmp_path / "b" / "trained-alpha0.50"
    assert torch.allclose(_read_student_state(path)["conv_in.bias"], init["conv_in.bias"] * 1.25)
    assert _read_student_timestep(path).item() == pytest.approx(255.5)


def test_mismatched_initialization_and_bad_requests_fail(trained, tmp_path):
    repo, checkpoint, _, _ = trained
    other = tmp_path / "other"
    (other / "unet").mkdir(parents=True)
    save_file({"conv_in.weight": torch.zeros(4, 4, 3, 3)}, str(other / "unet" / "diffusion_pytorch_model.safetensors"))
    with pytest.raises(ValueError, match="keys differ"):
        interpolate_student.main(["--checkpoint", str(checkpoint), "--model-repo", str(other),
                                  "--alpha", "0.5", "--output-root", str(tmp_path / "c")])
    with pytest.raises(SystemExit):
        interpolate_student.main(["--checkpoint", str(checkpoint), "--model-repo", str(repo),
                                  "--alpha", "1.5", "--output-root", str(tmp_path / "d")])
    interpolate_student.main(["--checkpoint", str(checkpoint), "--model-repo", str(repo),
                              "--alpha", "0.5", "--output-root", str(tmp_path / "e")])
    with pytest.raises(FileExistsError):
        interpolate_student.main(["--checkpoint", str(checkpoint), "--model-repo", str(repo),
                                  "--alpha", "0.5", "--output-root", str(tmp_path / "e")])
    assert not list((tmp_path / "e").glob(".*partial*"))


def test_initial_timestep_follows_the_aligner_default():
    import inspect

    from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner

    default = inspect.signature(StableFeatureAligner.__init__).parameters["t_init"].default
    assert interpolate_student.sd_teacher_initial_timestep() == float(default)
