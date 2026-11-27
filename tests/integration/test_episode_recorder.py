import csv
import numpy as np

from src.core.config.config_manager import ConfigManager
from src.core.environments.environment import Environment
from src.core.environments.action import Action, ButtonEvent
from src.core.tasks.click_target_task import ClickTargetTask

from src.data_acquisition.collection.episode import Episode
from src.data_acquisition.collection.recorder import Recorder
from src.data_acquisition.collection.recorder_status import RecorderStatus

from src.data_acquisition.collection.data_contracts import (
    ObservationRecord,
    ActionRecord
)

from src.data_acquisition.storage.stream_writers import (
    CSVWriter,
    ImageWriter
)


def test_terminal_data_preserved(tmp_path):

    config = ConfigManager("configs/experiment_001.yaml")

    env = Environment(config)
    task = ClickTargetTask(config)

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

    episode = Episode(
        env, task, config,
        recorder=recorder
    )

    state = episode.initialize(seed=42)
    episode.start()

    assert episode.is_running()
    assert recorder.get_status() == RecorderStatus.RECORDING

    # Move directly to target and click
    action = Action(
        delta_x=state.target_pos[0] - state.cursor_pos[0],
        delta_y=state.target_pos[1] - state.cursor_pos[1],
        button_event=ButtonEvent.PRESS
    )

    interaction = episode.step(action)

    assert episode.is_terminal()
    assert interaction.task_result.success

    # Recorder must still be accepting data
    assert recorder.get_status() == RecorderStatus.RECORDING

    recorder.record(
        "actions",
        ActionRecord(
            interaction.timestamp,
            0,
            action.delta_x,
            action.delta_y,
            action.button_event.value
        )
    )

    # Simulate terminal observation
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    recorder.record(
        "observations",
        ObservationRecord(
            episode.clock(),
            0,
            image
        )
    )

    episode.finalize()

    assert episode.is_finalized()
    assert recorder.get_status() == RecorderStatus.STOPPED

    # Verify terminal observation was persisted
    assert (
        recorder.episode_dir
        / "observations/frame_000000.png"
    ).exists()

    # Verify terminal click was persisted
    with (recorder.episode_dir / "actions.csv").open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["button_event"] == "press"