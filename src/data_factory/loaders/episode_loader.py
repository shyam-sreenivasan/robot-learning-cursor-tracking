import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image


class EpisodeLoader:

    def __init__(self, catalog, session_id, episode_id):

        self.catalog = catalog
        self.session_id = session_id
        self.episode_id = episode_id

        episode = catalog.get_episode(
            session_id,
            episode_id
        )

        directory = episode["directory"]

        if directory != f"episode_{episode_id}":
            raise ValueError(
                "Invalid episode directory reference."
            )

        self._episode_dir = (
            catalog.root_dir
            / f"session_{session_id}"
            / directory
        )

        if not self._episode_dir.is_dir():
            raise FileNotFoundError(self._episode_dir)

    # Metadata

    def get_metadata(self):

        return self._read_json("metadata.json")

    def get_recording_stats(self):

        return self._read_json("recording_stats.json")

    # Recorded streams

    def load_actions(self):

        return self._read_csv("actions.csv")

    def load_states(self):

        return self._read_csv("states.csv")

    def load_observation_index(self):

        return self._read_csv(
            "observations/index.csv"
        )

    # Individual observations

    def load_observation(self, frame_id):

        index = self.load_observation_index()

        matches = index[
            index["frame_id"] == frame_id
        ]

        if matches.empty:
            raise KeyError(
                f"Frame not found: {frame_id}"
            )

        if len(matches) != 1:
            raise ValueError(
                f"Duplicate frame ID: {frame_id}"
            )

        filename = matches.iloc[0]["filename"]

        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            raise ValueError("Invalid frame filename.")

        path = (
            self._episode_dir
            / "observations"
            / filename
        )

        with Image.open(path) as image:

            return np.array(image)

    # Internal storage access

    def _read_json(self, filename):

        path = self._episode_dir / filename

        with path.open("r") as file:
            return json.load(file)

    def _read_csv(self, filename):

        path = self._episode_dir / filename

        return pd.read_csv(
            path,
            float_precision="round_trip",
            keep_default_na=False
        )