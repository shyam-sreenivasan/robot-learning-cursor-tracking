import csv
import json
from PIL import Image

import pytest
import numpy as np

from src.data_acquisition.collection.recorder import Recorder
from src.data_acquisition.collection.recorder_status import RecorderStatus

from src.data_acquisition.collection.data_contracts import (
    ObservationRecord,
    ActionRecord,
    StateRecord
)

from src.data_acquisition.storage.stream_writers import (
    CSVWriter,
    ImageWriter
)


@pytest.fixture
def recorder(tmp_path):

    recorder = Recorder(tmp_path)

    recorder.register_stream(
        "observations",
        ObservationRecord,
        ImageWriter()
    )

    recorder.register_stream(
        "actions",
        ActionRecord,
        CSVWriter("actions.csv", ActionRecord)
    )

    recorder.register_stream(
        "states",
        StateRecord,
        CSVWriter("states.csv", StateRecord)
    )

    yield recorder

    if recorder.get_status() == RecorderStatus.RECORDING:
        recorder.stop()

def test_start(recorder):
    recorder.start("test_001")

    assert recorder.get_status() == RecorderStatus.RECORDING
    assert recorder.episode_dir.exists()

    assert (recorder.episode_dir / "actions.csv").exists()
    assert (recorder.episode_dir / "states.csv").exists()
    assert (recorder.episode_dir / "observations").exists()


def test_record_observation(recorder):
    recorder.start("test_001")

    image = np.zeros((480, 640, 3), dtype=np.uint8)

    recorder.record(
        "observations",
        ObservationRecord(1.0, 0, image)
    )

    recorder.stop()

    path = (
        recorder.episode_dir
        / "observations/frame_000000.png"
    )

    assert path.exists()

    with Image.open(path) as saved:
        assert np.array_equal(np.array(saved), image)

    with (
        recorder.episode_dir / "observations/index.csv"
    ).open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["frame_id"] == "0"


def test_record_action(recorder):
    recorder.start("test_001")

    recorder.record(
        "actions",
        ActionRecord(1.0, 0, 10, 20, "press")
    )

    recorder.stop()

    with (recorder.episode_dir / "actions.csv").open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert float(rows[0]["delta_x"]) == 10
    assert float(rows[0]["delta_y"]) == 20
    assert rows[0]["button_event"] == "press"


def test_record_state(recorder):
    recorder.start("test_001")

    recorder.record(
        "states",
        StateRecord(1.0, 0, 100, 200, 150, 150, 30, False)
    )

    recorder.stop()

    with (recorder.episode_dir / "states.csv").open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert float(rows[0]["cursor_x"]) == 100
    assert float(rows[0]["cursor_y"]) == 200
    assert int(rows[0]["target_radius"]) == 30


def test_invalid_record_type(recorder):
    recorder.start("test_001")

    with pytest.raises(TypeError):
        recorder.record("actions", "invalid record")


def test_invalid_record_values(recorder):
    recorder.start("test_001")

    record = ActionRecord(1.0, 0, 10, 20, "invalid")

    with pytest.raises(ValueError):
        recorder.record("actions", record)


def test_unknown_stream(recorder):
    recorder.start("test_001")

    with pytest.raises(ValueError):
        recorder.record("unknown", None)


def test_stop(recorder):
    recorder.start("test_001", {"seed": 42})

    recorder.stop({"status": "success"})

    assert recorder.get_status() == RecorderStatus.STOPPED

    with (recorder.episode_dir / "metadata.json").open() as file:
        metadata = json.load(file)

    assert metadata["seed"] == 42
    assert metadata["status"] == "success"


def test_record_before_start(recorder):

    with pytest.raises(RuntimeError):
        recorder.record("actions", None)


def test_record_after_stop(recorder):
    recorder.start("test_001")
    recorder.stop()

    with pytest.raises(RuntimeError):
        recorder.record("actions", None)


def test_duplicate_episode_id(recorder):
    recorder.start("test_001")
    recorder.stop()

    another = Recorder(recorder.output_dir)

    another.register_stream(
        "actions",
        ActionRecord,
        CSVWriter("actions.csv", ActionRecord)
    )

    with pytest.raises(FileExistsError):
        another.start("test_001")


def test_recording_statistics(recorder):
    recorder.start("test_001")

    for i in range(5):
        recorder.record(
            "actions",
            ActionRecord(float(i), i, 10, 20, "none")
        )

    recorder.stop()

    stats = recorder.get_stats()

    assert stats["stream_counts"]["actions"] == 5
    assert stats["stream_counts"]["observations"] == 0
    assert stats["status"] == "stopped"