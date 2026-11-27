import random

from src.config.config_manager import ConfigManager
from src.environments.environment import Environment
from src.tasks.click_target_task import ClickTargetTask

from src.collection.episode import Episode
from src.collection.recorder import Recorder

from src.collection.data_contracts import (
    ObservationRecord,
    ActionRecord,
    StateRecord
)

from src.storage.stream_writers import (
    ImageWriter,
    CSVWriter
)

from src.interfaces.pygame_application import PygameApplication


def main():

    config = ConfigManager("configs/experiment_001.yaml")

    env = Environment(config)
    task = ClickTargetTask(config)

    recorder = Recorder(
        config.resolve_path(
            config.get("storage.raw_data_dir")
        )
    )

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

    episode = Episode(
        env,
        task,
        config,
        recorder=recorder
    )

    app = PygameApplication(
        config,
        episode,
        recorder
    )

    seed = config.get("collection.random_seed")

    if seed is None:
        seed = random.randrange(2**32)

    metadata = app.run(seed)

    print("Episode metadata:")
    print(metadata)

    print("\nRecording statistics:")
    print(recorder.get_stats())

    print("\nDataset location:")
    print(recorder.episode_dir)


if __name__ == "__main__":
    main()