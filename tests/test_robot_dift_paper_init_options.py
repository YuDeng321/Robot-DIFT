"""Exercise Robot-DIFT initialization choices without model downloads or CUDA."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import warnings

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "agents/encoders/cleandift_img_encoder.py"
WRAPPER = ROOT / "droid_policy_learning/robomimic/models/cleandift_backbone.py"


def _method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _constructor_option_guard():
    """Execute the constructor's early option checks without building a model."""
    init = _method(ENCODER, "CleanDIFTImgEncoder", "__init__")
    index = next(
        i
        for i, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "student_init"
    )
    student_guards = [
        node for node in init.body[index + 1 :]
        if isinstance(node, ast.If) and "self.student_init" in ast.unparse(node.test)
    ][:2]
    guard = ast.FunctionDef(
        name="check_options",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="student_init"), ast.arg(arg="custom_checkpoint")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[init.body[index], *student_guards],
        decorator_list=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(ENCODER), "exec"), namespace)
    return namespace["check_options"]


def _run_init_backbone(student_init, alignment_feature_keys, vae_latent_mode="sample",
                       device="cpu", freeze_backbone=False):
    calls = {"hub": [], "load_file": [], "aligner_kwargs": None, "state_loads": 0, "vae_mode": None,
             "dtypes": []}

    class FakeAutoencoder:
        def __init__(self, repo, latent_mode="sample"):
            self.repo = repo
            calls["vae_mode"] = latent_mode

        def to(self, _device):
            return self

    class FakeAligner:
        repo = "test/sd21"

        def __init__(self, **kwargs):
            calls["aligner_kwargs"] = kwargs
            self.weight = torch.nn.Parameter(torch.ones(()))

        def to(self, *_args, **kwargs):
            if "dtype" in kwargs:
                calls["dtypes"].append(kwargs["dtype"])
            return self

        def parameters(self):
            return iter([self.weight])

        def load_state_dict(self, _state, strict=False):
            assert strict is False
            calls["state_loads"] += 1
            return [], []

    def fake_hub_download(**kwargs):
        calls["hub"].append(kwargs)
        return "/tmp/fake-cleandift.safetensors"

    def fake_load_file(path):
        calls["load_file"].append(path)
        return {"weight": torch.ones(())}

    config = {
        "model": {
            "feature_dims": {"mid": 4, "us3": 4, "us6": 4, "us8": 4},
            "ae": {"repo": "test/sd21"},
            "mapping": {"depth": 1, "width": 4, "d_ff": 4, "dropout": 0.0},
            "adapter_layer_class": "test.Adapter",
            "feature_extractor_cls": "test.Extractor",
        }
    }
    namespace = {
        "__file__": str(ENCODER),
        "os": os,
        "warnings": warnings,
        "torch": torch,
        "Optional": Optional,
        "OmegaConf": SimpleNamespace(load=lambda _path: config),
        "AutoencoderKL": FakeAutoencoder,
        "MappingSpec": lambda **kwargs: SimpleNamespace(**kwargs),
        "StableFeatureAligner": FakeAligner,
        "locate": lambda _path: object(),
        "hf_hub_download": fake_hub_download,
        "load_file": fake_load_file,
    }
    node = _method(ENCODER, "CleanDIFTImgEncoder", "_init_backbone")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENCODER), "exec"), namespace)
    encoder = SimpleNamespace(
        feature_keys=["us3", "us6", "us8"],
        alignment_feature_keys=alignment_feature_keys,
        student_init=student_init,
        vae_latent_mode=vae_latent_mode,
        use_text_condition=False,
        freeze_backbone=freeze_backbone,
        _force_backbone_fp32=False,
    )
    namespace["_init_backbone"](encoder, "sd21", None, device, None)
    return encoder, calls


def test_trainable_student_keeps_float32_weights_and_bf16_autocast(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("ROBOT_DIFT_DROID_AMP_DTYPE", "bfloat16")
    encoder, calls = _run_init_backbone("sd_teacher", ["us3"], device="cuda", freeze_backbone=False)
    assert calls["dtypes"] == [torch.float32]
    assert encoder._amp_autocast_kwargs == {"device_type": "cuda", "dtype": torch.bfloat16}
    frozen, calls = _run_init_backbone("sd_teacher", ["us3"], device="cuda", freeze_backbone=True)
    assert calls["dtypes"] == [torch.bfloat16]
    assert frozen._amp_enabled is True


def test_sd_teacher_init_uses_all_alignment_keys_without_cleandift_download():
    encoder, calls = _run_init_backbone("sd_teacher", ["mid", "us3", "us6", "us8"])

    assert list(calls["aligner_kwargs"]["feature_dims"]) == ["mid", "us3", "us6", "us8"]
    assert calls["hub"] == []
    assert calls["load_file"] == []
    assert calls["state_loads"] == 0
    assert encoder._checkpoint_source == "test/sd21:sd_teacher_weight_copy"


def test_legacy_init_keeps_three_alignment_keys_and_public_checkpoint():
    encoder, calls = _run_init_backbone("cleandift", ["us3", "us6", "us8"])

    assert list(calls["aligner_kwargs"]["feature_dims"]) == ["us3", "us6", "us8"]
    assert calls["hub"] == [
        {"repo_id": "CompVis/cleandift", "filename": "cleandift_sd21_full.safetensors"}
    ]
    assert calls["state_loads"] == 1
    assert encoder._checkpoint_source == "CompVis/cleandift:fake-cleandift.safetensors"


def test_stage1_vae_mode_reaches_shared_autoencoder_without_checkpoint_download():
    _, calls = _run_init_backbone("sd_teacher", ["us3", "us6", "us8"], "mode")
    assert calls["vae_mode"] == "mode"
    assert calls["hub"] == []


def test_robomimic_wrapper_forwards_opt_in_parameters():
    node = _method(WRAPPER, "CleanDIFTConv", "__init__")
    call = next(
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "CleanDIFTImgEncoder"
    )
    kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
    for name in ("alignment_feature_keys", "student_init", "vae_latent_mode"):
        assert isinstance(kwargs[name], ast.Name)
        assert kwargs[name].id == name


def test_constructor_defaults_preserve_legacy_behavior_and_guard_conflicts():
    init = _method(ENCODER, "CleanDIFTImgEncoder", "__init__")
    default_names = [arg.arg for arg in init.args.args[-len(init.args.defaults) :]]
    defaults = dict(zip(default_names, init.args.defaults))
    assert ast.literal_eval(defaults["alignment_feature_keys"]) is None
    assert ast.literal_eval(defaults["student_init"]) == "cleandift"

    check_options = _constructor_option_guard()
    with pytest.raises(ValueError, match="Unsupported student_init"):
        check_options(SimpleNamespace(), "unknown", None)
    with pytest.raises(ValueError, match="cannot be combined with custom_checkpoint"):
        check_options(SimpleNamespace(), "sd_teacher", "/tmp/weights")
    check_options(SimpleNamespace(), "sd_teacher", None)
