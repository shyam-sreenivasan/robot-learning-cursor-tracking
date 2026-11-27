import numpy as np
import pandas as pd

from .contracts import ValidationResult, ValidationStatus


class EpisodeIntegrityValidator:

    REQUIRED_COLUMNS = {
        "actions": [
            "timestamp", "event_id",
            "delta_x", "delta_y", "button_event"
        ],
        "states": [
            "timestamp", "frame_id",
            "cursor_x", "cursor_y",
            "target_x", "target_y",
            "target_radius", "button_down"
        ],
        "observations": [
            "timestamp", "frame_id", "filename"
        ]
    }

    def __init__(self, episode_loader):

        self.loader = episode_loader

    def validate(self):

        checks = [
            self.validate_metadata,
            self.validate_schema,
            self.validate_values,
            self.validate_record_counts,
            self.validate_record_ids,
            self.validate_timestamps,
            self.validate_stream_alignment,
            self.validate_temporal_boundaries,
            self.validate_images
        ]

        results = []

        for check in checks:

            try:
                results.append(check())

            except Exception as error:

                results.append(
                    ValidationResult(
                        check_name=check.__name__,
                        status=ValidationStatus.FAIL,
                        message=str(error)
                    )
                )

        return results

    def _load_streams(self):

        return {
            "actions": self.loader.load_actions(),
            "states": self.loader.load_states(),
            "observations":
                self.loader.load_observation_index()
        }

    @staticmethod
    def _result(name, passed, expected=None, observed=None):

        return ValidationResult(
            check_name=name,
            status=(
                ValidationStatus.PASS
                if passed
                else ValidationStatus.FAIL
            ),
            message=(
                "Check passed"
                if passed
                else "Check failed"
            ),
            expected=expected,
            observed=observed
        )

    @staticmethod
    def _skipped(name, reason):

        return ValidationResult(
            check_name=name,
            status=ValidationStatus.SKIPPED,
            message=reason
        )

    def validate_metadata(self):

        metadata = self.loader.get_metadata()
        stats = self.loader.get_recording_stats()

        registered = self.loader.catalog.get_episode(
            self.loader.session_id,
            self.loader.episode_id
        )

        expected = {
            "episode_id": self.loader.episode_id,
            "episode_status": registered["status"],
            "recorder_status": "stopped",
            "recorder_episode_id": self.loader.episode_id
        }

        observed = {
            "episode_id": metadata["episode_id"],
            "episode_status": metadata["status"],
            "recorder_status": stats["status"],
            "recorder_episode_id": stats["episode_id"]
        }

        return self._result(
            "metadata",
            expected == observed
            and metadata["status"] in {
                "success", "failed", "aborted"
            },
            expected,
            observed
        )

    def validate_schema(self):

        streams = self._load_streams()

        missing = {}

        for name, required in self.REQUIRED_COLUMNS.items():

            columns = streams[name].columns

            absent = [
                col for col in required
                if col not in columns
            ]

            if absent:
                missing[name] = absent

        return self._result(
            "schema",
            not missing,
            {},
            missing
        )

    def validate_values(self):

        streams = self._load_streams()

        numeric_columns = {
            "actions": [
                "timestamp", "event_id",
                "delta_x", "delta_y"
            ],
            "states": [
                "timestamp", "frame_id",
                "cursor_x", "cursor_y",
                "target_x", "target_y",
                "target_radius"
            ],
            "observations": [
                "timestamp", "frame_id"
            ]
        }

        errors = []

        for name, columns in numeric_columns.items():

            for column in columns:

                if column not in streams[name]:
                    return self._skipped(
                        "values", "Incomplete schema"
                    )

                values = streams[name][column]

                if not pd.api.types.is_numeric_dtype(values):

                    if len(values) > 0:
                        errors.append(f"{name}.{column}")

                elif not np.isfinite(values).all():

                    errors.append(f"{name}.{column}")

        actions = streams["actions"]
        states = streams["states"]
        index = streams["observations"]

        required = [
            (actions, "button_event"),
            (states, "button_down"),
            (index, "filename")
        ]

        if any(col not in df for df, col in required):
            return self._skipped(
                "values", "Incomplete schema"
            )

        if not actions["button_event"].isin(
            ["none", "press", "release"]
        ).all():
            errors.append("actions.button_event")

        if not states["button_down"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            errors.append("states.button_down")

        if not index["filename"].map(
            lambda value: isinstance(value, str)
            and bool(value.strip())
        ).all():
            errors.append("observations.filename")

        return self._result(
            "values",
            not errors,
            [],
            errors
        )

    def validate_record_counts(self):

        streams = self._load_streams()

        expected = self.loader.get_recording_stats()[
            "stream_counts"
        ]

        observed = {
            name: len(stream)
            for name, stream in streams.items()
        }

        return self._result(
            "record_counts",
            expected == observed,
            expected,
            observed
        )
    
    def validate_record_ids(self):

        streams = self._load_streams()

        id_columns = {
            "actions": "event_id",
            "states": "frame_id",
            "observations": "frame_id"
        }

        results = {}

        for name, column in id_columns.items():

            stream = streams[name]

            if column not in stream:
                return self._skipped(
                    "record_ids", "Missing ID column"
                )

            observed = stream[column].to_numpy()
            expected = np.arange(len(stream))

            results[name] = bool(
                np.array_equal(expected, observed)
            )

        return self._result(
            "record_ids",
            all(results.values()),
            "Sequential IDs starting at zero",
            results
        )

    def validate_timestamps(self):

        results = {}

        for name, stream in self._load_streams().items():

            if "timestamp" not in stream:
                return self._skipped(
                    "timestamps", "Missing timestamp column"
                )

            timestamps = pd.to_numeric(
                stream["timestamp"],
                errors="coerce"
            ).to_numpy(dtype=float)

            results[name] = bool(
                np.isfinite(timestamps).all()
                and np.all(np.diff(timestamps) >= 0)
            )

        return self._result(
            "timestamps",
            all(results.values()),
            "Finite, non-decreasing timestamps",
            results
        )

    def validate_stream_alignment(self):

        states = self.loader.load_states()
        index = self.loader.load_observation_index()

        columns = ["frame_id", "timestamp"]

        if any(
            column not in stream
            for stream in [states, index]
            for column in columns
        ):
            return self._skipped(
                "stream_alignment", "Incomplete schema"
            )

        same_ids = np.array_equal(
            states["frame_id"].to_numpy(),
            index["frame_id"].to_numpy()
        )

        same_timestamps = np.array_equal(
            states["timestamp"].to_numpy(),
            index["timestamp"].to_numpy()
        )

        passed = (
            len(index) > 0
            and same_ids
            and same_timestamps
        )

        return self._result(
            "stream_alignment",
            passed,
            "Matching frame IDs and timestamps",
            {
                "state_count": len(states),
                "observation_count": len(index),
                "matching_ids": same_ids,
                "matching_timestamps": same_timestamps
            }
        )

    def validate_temporal_boundaries(self):

        actions = self.loader.load_actions()
        index = self.loader.load_observation_index()

        if (
            "timestamp" not in actions
            or "timestamp" not in index
        ):
            return self._skipped(
                "temporal_boundaries",
                "Missing timestamps"
            )

        if index.empty:

            return self._result(
                "temporal_boundaries",
                False,
                "At least one observation",
                0
            )

        start = index["timestamp"].iloc[0]
        end = index["timestamp"].iloc[-1]

        passed = bool(
            (actions["timestamp"] >= start).all()
            and (actions["timestamp"] <= end).all()
        )

        return self._result(
            "temporal_boundaries",
            passed,
            {"start": start, "end": end},
            {
                "first_action": (
                    actions["timestamp"].min()
                    if not actions.empty else None
                ),
                "last_action": (
                    actions["timestamp"].max()
                    if not actions.empty else None
                )
            }
        )

    def validate_images(self):

        index = self.loader.load_observation_index()

        manifest = self.loader.catalog.get_session(
            self.loader.session_id
        )

        environment = manifest["environment"]

        expected_shape = (
            environment["height"],
            environment["width"],
            3
        )

        if "frame_id" not in index:
            return self._skipped(
                "images", "Missing frame IDs"
            )

        errors = []

        for frame_id in index["frame_id"]:

            try:

                image = self.loader.load_observation(
                    frame_id
                )

                if (
                    image.shape != expected_shape
                    or image.dtype != np.uint8
                ):

                    errors.append({
                        "frame_id": frame_id,
                        "shape": image.shape,
                        "dtype": str(image.dtype)
                    })

            except Exception as error:

                errors.append({
                    "frame_id": frame_id,
                    "error": str(error)
                })

        return self._result(
            "images",
            not errors and not index.empty,
            {
                "shape": expected_shape,
                "dtype": "uint8"
            },
            {
                "total_frames": len(index),
                "invalid_frames": errors
            }
        )