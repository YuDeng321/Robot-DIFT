"""The RoboCasa loader must return the observation horizon used by Stage II."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from environments.dataset.robocasa_dataset_memory import RobocasaDataset


def _write_demo(dataset_root):
    date_dir = dataset_root / "kitchen_coffee" / "CoffeePressButton" / "2024-04-25"
    date_dir.mkdir(parents=True)
    path = date_dir / "demo_gentex_im128_randcams.hdf5"
    frames = 18
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["num_samples"] = frames
        demo.attrs["ep_meta"] = json.dumps({"lang": "press the coffee button"})
        actions = np.repeat(np.arange(frames, dtype=np.float32)[:, None], 7, axis=1)
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        pixels = np.broadcast_to(
            np.arange(frames, dtype=np.uint8)[:, None, None, None], (frames, 4, 4, 3)
        ).copy()
        obs.create_dataset("robot0_agentview_left_image", data=pixels)
        obs.create_dataset("robot0_gripper_qpos", data=np.arange(frames, dtype=np.float32)[:, None])
        obs.create_dataset("robot0_joint_pos", data=np.zeros((frames, 7), dtype=np.float32))
        obs.create_dataset("robot0_eef_pos", data=np.zeros((frames, 3), dtype=np.float32))
        obs.create_dataset("robot0_eef_quat", data=np.zeros((frames, 4), dtype=np.float32))


def _dataset(dataset_root, **kwargs):
    return RobocasaDataset(
        cam_names=["robot0_agentview_left"],
        env_name=["CoffeePressButton"],
        data_directory=str(dataset_root),
        action_dim=7,
        window_size=17,
        **kwargs,
    )


def test_two_frame_observations_with_17_action_window(tmp_path):
    _write_demo(tmp_path)
    dataset = _dataset(tmp_path, obs_seq_len=2)
    assert len(dataset) == 2

    first_obs, first_actions, first_mask = dataset[0]
    second_obs, second_actions, second_mask = dataset[1]
    key = "robot0_agentview_left_image"
    assert first_obs[key].shape == (2, 3, 4, 4)
    assert first_obs["robot_states"].shape == (2, 30)
    assert first_obs["lang"] == "press the coffee button"
    torch.testing.assert_close(first_obs[key][:, 0, 0, 0], torch.tensor([0.0, 1 / 255]))
    torch.testing.assert_close(second_obs[key][:, 0, 0, 0], torch.tensor([1 / 255, 2 / 255]))
    torch.testing.assert_close(first_obs["robot_states"][:, 0], torch.tensor([-1.0, -1 + 2 / 17]))
    assert first_actions.shape == second_actions.shape == (17, 7)
    assert first_mask.shape == second_mask.shape == (17,)
    # The trainer drops obs_seq_len-1 actions, leaving the 16 policy targets.
    assert first_actions[1:].shape == (16, 7)
    torch.testing.assert_close(first_actions[:, 0], torch.arange(17, dtype=torch.float32))
    torch.testing.assert_close(second_actions[:, 0], torch.arange(1, 18, dtype=torch.float32))


def test_default_keeps_single_frame_legacy_behavior(tmp_path):
    _write_demo(tmp_path)
    obs, actions, mask = _dataset(tmp_path)[0]
    assert obs["robot0_agentview_left_image"].shape[0] == 1
    assert obs["robot_states"].shape[0] == 1
    assert actions.shape == (17, 7)
    assert mask.shape == (17,)


def test_observation_horizon_must_fit_window(tmp_path):
    with pytest.raises(ValueError, match="obs_seq_len"):
        _dataset(tmp_path, obs_seq_len=18)


def test_opt_in_frame_ids_track_overlapping_windows(tmp_path):
    _write_demo(tmp_path)
    dataset = _dataset(tmp_path, obs_seq_len=2, return_frame_ids=True)
    first = dataset[0][0]["_robot_dift_frame_ids"]
    second = dataset[1][0]["_robot_dift_frame_ids"]
    assert first.shape == second.shape == (2, 3)
    torch.testing.assert_close(first[1], second[0])
    assert not torch.equal(first[0], first[1])
    other = _dataset(tmp_path, obs_seq_len=2, return_frame_ids=True)
    assert dataset[0][0]["_robot_dift_frame_ids"][0, 0] != other[0][0]["_robot_dift_frame_ids"][0, 0]
