from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.dataset.libero_dataset import LiberoDataset


def _write_minimal_libero_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("joint_states", data=np.zeros((2, 7), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((2, 2), dtype=np.float32))


def test_libero_dataset_without_task_embeddings_when_disabled(tmp_path):
    dataset_dir = tmp_path / "libero_object"
    dataset_dir.mkdir()
    _write_minimal_libero_hdf5(dataset_dir / "open_drawer_demo.hdf5")

    dataset = LiberoDataset(
        data_directory=str(dataset_dir),
        max_len_data=4,
        window_size=1,
        traj_per_task=1,
        use_task_emb=False,
    )

    obs, action, mask = dataset[0]

    assert "lang_emb" not in obs
    assert obs["lang"] == "open drawer"
    assert action.shape == (1, 7)
    assert mask.shape == (1,)


def test_libero_two_frame_observation_and_sixteen_step_action_window(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "libero_10"
    dataset_dir.mkdir()
    path = dataset_dir / "open_drawer_demo.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["num_samples"] = 17
        demo.create_dataset("actions", data=np.arange(17 * 7, dtype=np.float32).reshape(17, 7))
        obs = demo.create_group("obs")
        frames = np.arange(17, dtype=np.uint8).reshape(17, 1, 1, 1) * np.ones((1, 8, 8, 3), dtype=np.uint8)
        obs.create_dataset("agentview_rgb", data=frames)
        obs.create_dataset("eye_in_hand_rgb", data=frames)
        obs.create_dataset("joint_states", data=np.arange(17 * 7, dtype=np.float32).reshape(17, 7))
        obs.create_dataset("gripper_states", data=np.zeros((17, 2), dtype=np.float32))

    libero_source = tmp_path / "source"
    bddl_dir = libero_source / "libero" / "libero" / "bddl_files" / "libero_10"
    bddl_dir.mkdir(parents=True)
    (bddl_dir / "open_drawer.bddl").write_text("(:language open the drawer carefully)")
    monkeypatch.setenv("ROBOT_DIFT_LIBERO_SOURCE", str(libero_source))

    dataset = LiberoDataset(
        data_directory=str(dataset_dir),
        max_len_data=17,
        window_size=17,
        obs_seq_len=2,
        traj_per_task=1,
        use_task_emb=False,
    )
    obs, action, mask = dataset[0]
    assert obs["agentview_image"].shape == (2, 3, 8, 8)
    assert obs["eye_in_hand_image"].shape == (2, 3, 8, 8)
    assert obs["robot_states"].shape == (2, 9)
    assert obs["agentview_image"][1, 0, 0, 0].item() == pytest.approx(1 / 255)
    assert obs["lang"] == "open the drawer carefully"
    assert action[1:].shape == (16, 7)
    assert mask[1:].shape == (16,)
