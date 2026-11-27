import json
import pytest
import yaml
import numpy as np
import pandas as pd

from PIL import Image

from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader


@pytest.fixture
def loader(tmp_path):

    session_dir = tmp_path / "session_001"
    episode_dir = session_dir / "episode_001"
    observation_dir = episode_dir / "observations"

    observation_dir.mkdir(parents=True)

    manifest = {
        "execution": {
            "session_id": "001",
            "episodes": [
                {
                    "episode_id": "001",
                    "directory": "episode_001",
                    "status": "success"
                }
            ]
        }
    }

    with (session_dir / "session.yaml").open("w") as file:
        yaml.safe_dump(manifest, file)

    metadata = {
        "episode_id": "001",
        "status": "success",
        "seed": 42
    }

    stats = {
        "status": "stopped",
        "stream_counts": {
            "observations": 2,
            "actions": 2,
            "states": 2
        }
    }

    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata)
    )

    (episode_dir / "recording_stats.json").write_text(
        json.dumps(stats)
    )

    actions = pd.DataFrame({
        "timestamp": [1.01, 1.02],
        "event_id": [0, 1],
        "delta_x": [10, -5],
        "delta_y": [20, 15],
        "button_event": ["none", "press"]
    })

    actions.to_csv(
        episode_dir / "actions.csv",
        index=False
    )

    states = pd.DataFrame({
        "timestamp": [1.0, 1.03],
        "frame_id": [0, 1],
        "cursor_x": [100, 105],
        "cursor_y": [100, 135],
        "target_x": [150, 150],
        "target_y": [150, 150],
        "target_radius": [20, 20],
        "button_down": [False, True]
    })

    states.to_csv(
        episode_dir / "states.csv",
        index=False
    )

    index = pd.DataFrame({
        "timestamp": [1.0, 1.03],
        "frame_id": [0, 1],
        "filename": [
            "frame_000000.png",
            "frame_000001.png"
        ]
    })

    index.to_csv(
        observation_dir / "index.csv",
        index=False
    )

    for i in range(2):

        image = np.full(
            (480, 640, 3),
            i * 100,
            dtype=np.uint8
        )

        Image.fromarray(image).save(
            observation_dir / f"frame_{i:06d}.png"
        )

    catalog = DataCatalog(tmp_path)

    return EpisodeLoader(
        catalog, "001", "001"
    )


def test_load_metadata(loader):

    metadata = loader.get_metadata()

    assert metadata["episode_id"] == "001"
    assert metadata["status"] == "success"
    assert metadata["seed"] == 42


def test_load_recording_stats(loader):

    stats = loader.get_recording_stats()

    assert stats["status"] == "stopped"
    assert stats["stream_counts"]["observations"] == 2


def test_load_actions(loader):

    actions = loader.load_actions()

    assert len(actions) == 2
    assert actions["event_id"].tolist() == [0, 1]
    assert actions["delta_x"].tolist() == [10, -5]
    assert actions["button_event"].tolist() == [
        "none", "press"
    ]


def test_load_states(loader):

    states = loader.load_states()

    assert len(states) == 2
    assert states["cursor_x"].tolist() == [100, 105]
    assert states["button_down"].tolist() == [
        False, True
    ]


def test_load_observation_index(loader):

    index = loader.load_observation_index()

    assert index["frame_id"].tolist() == [0, 1]
    assert index["timestamp"].tolist() == [1.0, 1.03]


def test_load_observation(loader):

    image = loader.load_observation(1)

    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8
    assert np.all(image == 100)


def test_unknown_episode(loader):

    with pytest.raises(KeyError):

        EpisodeLoader(
            loader.catalog, "001", "unknown"
        )


def test_unknown_frame(loader):

    with pytest.raises(KeyError):
        loader.load_observation(999)


def test_missing_stream(loader):

    path = loader._episode_dir / "actions.csv"
    path.unlink()

    with pytest.raises(FileNotFoundError):
        loader.load_actions()


def test_missing_image(loader):

    path = (
        loader._episode_dir
        / "observations/frame_000000.png"
    )

    path.unlink()

    with pytest.raises(FileNotFoundError):
        loader.load_observation(0)


def test_duplicate_frame_id(loader):

    path = (
        loader._episode_dir
        / "observations/index.csv"
    )

    index = loader.load_observation_index()

    index = pd.concat(
        [index, index.iloc[[0]]],
        ignore_index=True
    )

    index.to_csv(path, index=False)

    with pytest.raises(ValueError):
        loader.load_observation(0)


def test_read_only_access(loader):

    paths = [
        p for p in loader._episode_dir.rglob("*")
        if p.is_file()
    ]

    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in paths
    }

    loader.get_metadata()
    loader.get_recording_stats()
    loader.load_actions()
    loader.load_states()
    loader.load_observation_index()
    loader.load_observation(0)

    after = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in paths
    }

    assert before == after