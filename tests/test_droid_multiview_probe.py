"""DROID multi-camera probe: frame selection, grouped ridge probe, paired statistics, camera use."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from agents.encoders.robot_dift_paper_readout import RobotDIFTPaperReadout

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("droid_multiview_probe", ROOT / "scripts/probes/droid_multiview_probe.py")
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

CAMERAS = ("wrist_image_left", "exterior_image_1_left")
CHANNELS = {"us3": 4, "us6": 4, "us8": 4}
TINY_READOUT = {"fpn_dim": 8, "model_dim": 16, "output_dim": 12, "num_heads": 2, "transformer_layers": 1,
                "mlp_hidden_dims": [16], "token_pooling": "flatten", "feature_keys": ["us3", "us6", "us8"]}


def test_frame_selection_leaves_room_for_the_horizon():
    frames = probe.select_frames(20, 6, 8)
    assert len(frames) == 6 and frames[0] == 0 and frames[-1] == 11
    assert probe.select_frames(12, 20, 8) == [0, 1, 2, 3]
    assert probe.select_frames(8, 3, 8) == []


def test_grouped_folds_keep_each_episode_on_one_side():
    groups = np.repeat(np.arange(10), 3)
    folds = probe.grouped_folds(groups, 5, seed=0)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(30))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    with pytest.raises(ValueError, match="episodes"):
        probe.grouped_folds(np.arange(3), 5, seed=0)


def test_ridge_probe_recovers_a_latent_signal_and_not_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 4)
    latent = rng.normal(size=(160, 3))
    features = latent @ rng.normal(size=(3, 300)) + 0.05 * rng.normal(size=(160, 300))
    targets = np.stack([latent @ np.array([1.0, -2.0, 0.5]), rng.normal(size=160)], axis=1)
    predictions = probe.ridge_oof_predictions(features.astype(np.float16), targets, groups, folds=5,
                                              inner_folds=3, seed=0)
    assert probe.r2(targets[:, :1], predictions[:, :1]) > 0.95
    assert probe.r2(targets[:, 1:], predictions[:, 1:]) < 0.1


def test_ridge_path_clips_negative_roundoff_eigenvalues():
    # A Gram matrix assembled at lower precision can acquire a small negative
    # eigenvalue. Its magnitude must not cancel a valid positive ridge penalty.
    kernel = np.array([[1.0, 1.000001], [1.000001, 1.0]])
    path = probe._ridge_path(kernel, np.array([[1.0], [-1.0]]), kernel, [1e-6])
    assert np.isfinite(path).all()
    assert np.max(np.abs(path)) < 3.0


def test_paired_difference_detects_the_better_predictions():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 5)
    targets = rng.normal(size=(150, 2))
    worse = targets + rng.normal(scale=1.0, size=targets.shape)
    better = targets + rng.normal(scale=0.3, size=targets.shape)
    result = probe.paired_r2_difference(targets, worse, better, groups, draws=500, seed=0)
    assert result["delta_r2"] > 0.3 and result["ci95"][0] > 0 and result["p_positive"] == 1.0


class _StubReadout:
    """Max-pools a fixed per-camera tensor, like the paper readout's view pooling."""

    def __init__(self, per_view):
        self.per_view = per_view

    def encode_views(self, views, tokens, mask):
        return self.per_view[:, [view["index"] for view in views]]

    def __call__(self, views, tokens, mask):
        return self.encode_views(views, tokens, mask).amax(dim=1).flatten(1)


def test_camera_win_share_counts_active_query_channels():
    per_view = torch.zeros(1, 2, 3, 4)
    per_view[0, 0, :, :3] = 1.0   # camera 0 wins three of four channels
    per_view[0, 1, :, 3] = 1.0    # camera 1 wins the last one
    per_view[0, 1, 2, :] = 5.0    # a padded query that camera 1 would win
    mask = torch.tensor([[True, True, False]])
    stats = probe.camera_statistics(_StubReadout(per_view), [{"index": 0}, {"index": 1}], None, mask)
    assert stats["win_share"].tolist() == [[0.75, 0.25]]
    assert stats["drop_distance"][0, 0] > 0 and stats["drop_distance"][0, 1] > 0


def _tiny_readout():
    config = {key: value for key, value in TINY_READOUT.items() if key != "feature_keys"}
    config["mlp_hidden_dims"] = tuple(config["mlp_hidden_dims"])
    return RobotDIFTPaperReadout(CHANNELS, feature_keys=("us3", "us6", "us8"), **config)


def test_identical_cameras_are_interchangeable_in_the_paper_readout():
    torch.manual_seed(0)
    readout = _tiny_readout().eval()
    maps = {"us3": torch.randn(2, 4, 2, 2), "us6": torch.randn(2, 4, 4, 4), "us8": torch.randn(2, 4, 4, 4)}
    mask = torch.zeros(2, 77, dtype=torch.bool)
    mask[:, :5] = True
    stats = probe.camera_statistics(readout, [maps, {k: v.clone() for k, v in maps.items()}],
                                    torch.randn(2, 77, 512), mask)
    torch.testing.assert_close(stats["drop_distance"], torch.zeros(2, 2), atol=1e-5, rtol=0)
    torch.testing.assert_close(stats["single_distance"], torch.zeros(2, 2), atol=1e-5, rtol=0)
    torch.testing.assert_close(stats["win_share"].sum(dim=1), torch.ones(2))


def _jpeg(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _episodes(count, steps=24, seed=0):
    """Synthetic episodes whose images encode the end-effector position."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        position = np.cumsum(rng.normal(scale=0.02, size=(steps, 6)), axis=0)
        gripper = (np.arange(steps) > steps // 2).astype(float)[:, None] * rng.uniform(0.6, 1.0)
        images = {}
        for camera_index, camera in enumerate(CAMERAS):
            frames = []
            for step in range(steps):
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                row = int(np.clip(16 + 60 * position[step, 0], 0, 31))
                col = int(np.clip(16 + 60 * position[step, 1] * (1 if camera_index == 0 else -1), 0, 31))
                image[max(row - 3, 0):row + 3, max(col - 3, 0):col + 3] = 255
                image[..., 2] = np.uint8(200 * gripper[step, 0])
                frames.append(_jpeg(image))
            images[camera] = frames
        yield {"episode_id": f"/droid/failure/episode_{index}", "instruction": "put the cup in the sink",
               "images": images, "cartesian_position": position, "gripper_position": gripper}


def test_samples_from_episodes_targets_and_cache_roundtrip(tmp_path):
    samples = probe.samples_from_episodes(_episodes(3), cameras=CAMERAS, per_episode=4, horizon=8)
    assert len(samples) == 12 and samples.images[CAMERAS[0]].shape == (12, 32, 32, 3)
    episode = next(_episodes(1))
    frame = samples.frames[1]
    expected_displacement = episode["cartesian_position"][frame + 8, :3] - episode["cartesian_position"][frame, :3]
    np.testing.assert_allclose(samples.targets[1, 4:7], expected_displacement)
    np.testing.assert_allclose(samples.targets[1, :3], episode["cartesian_position"][frame, :3])
    path = tmp_path / "frames.npz"
    samples.save(path)
    loaded = probe.Samples.load(path)
    assert loaded.instructions == samples.instructions and loaded.episode_names == samples.episode_names
    np.testing.assert_array_equal(loaded.images[CAMERAS[1]], samples.images[CAMERAS[1]])
    np.testing.assert_array_equal(loaded.targets, samples.targets)


class _Extractor:
    """Stand-in Student: informative maps pool the image; uninformative ones are image-independent noise."""

    feature_dims = CHANNELS

    def __init__(self, informative):
        self.informative = informative
        self.generator = torch.Generator().manual_seed(123)

    def _encode_backbone(self, images, captions):
        assert len(captions) == images.shape[0]
        batch = images.shape[0]
        if self.informative:
            base = torch.cat([images, images.mean(dim=1, keepdim=True)], dim=1)
            return {"us3": F.adaptive_avg_pool2d(base, 8), "us6": F.adaptive_avg_pool2d(base, 16),
                    "us8": F.adaptive_avg_pool2d(base, 16)}
        return {key: torch.randn(batch, 4, size, size, generator=self.generator)
                for key, size in (("us3", 8), ("us6", 16), ("us8", 16))}


class _Text(nn.Module):
    def forward(self, captions):
        mask = torch.zeros(len(captions), 77, dtype=torch.bool)
        mask[:, :6] = True
        return torch.ones(len(captions), 77, 512), mask


def _checkpoint(path, readout):
    path.mkdir(parents=True)
    metadata = {"readout": "legacy", "feature_key": ["us3", "us6", "us8"],
                "components": {"deploy_head": "deploy_head.safetensors"}}
    if readout is not None:
        metadata.update(readout="paper", readout_config=TINY_READOUT)
        save_file({f"readout.{key}": value.contiguous() for key, value in readout.state_dict().items()},
                  str(path / "deploy_head.safetensors"))
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def test_probe_end_to_end_prefers_the_informative_student(tmp_path):
    torch.manual_seed(0)
    samples = probe.samples_from_episodes(_episodes(24), cameras=CAMERAS, per_episode=4, horizon=8)
    cache = tmp_path / "frames.npz"
    samples.save(cache)
    noise = _checkpoint(tmp_path / "sd21", None)
    informative = _checkpoint(tmp_path / "robot_dift", _tiny_readout())
    extractors = {noise: _Extractor(False), informative: _Extractor(True)}
    assert probe.main(
        ["--checkpoint", f"sd21_dift={noise}", "--checkpoint", f"robot_dift={informative}",
         "--samples", str(cache), "--cameras", *CAMERAS, "--clip-model", "stand-in", "--device", "cpu",
         "--image-size", "256", "--grid", "2", "--folds", "4", "--inner-folds", "2", "--bootstrap", "200",
         "--batch-size", "32", "--output-dir", str(tmp_path / "report")],
        extractor_factory=lambda path: extractors[path], text_encoder_factory=_Text,
    ) == 0
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["reference"] == "sd21_dift"
    assert report["readout_analysis"] == {"sd21_dift": "checkpoint has no paper readout", "robot_dift": "analyzed"}
    delta = report["paired_vs_reference"]["robot_dift"]["all_cameras"]["ee_position"]
    assert delta["delta_r2"] > 0.2 and delta["ci95"][0] > 0
    assert report["raw_maps"]["robot_dift"]["all_cameras"]["gripper"]["r2"] > 0.5
    readout = report["readout"]["robot_dift"]
    assert set(readout["win_share"]) == set(CAMERAS)
    assert sum(readout["win_share"].values()) == pytest.approx(1.0)
    assert set(readout["token_change_without_camera"]) == set(CAMERAS)
    assert all(value > 0 for value in readout["token_change_without_camera"].values())
    assert "ee_position" in readout["policy_token_probe"]
    summary = (tmp_path / "report" / "summary.md").read_text()
    assert "robot_dift minus sd21_dift" in summary and "wrist_image_left" in summary
