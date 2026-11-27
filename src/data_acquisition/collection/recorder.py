import json
import time

from pathlib import Path
from .recorder_status import RecorderStatus


class Recorder:

    def __init__(self, output_dir, clock=time.monotonic):

        self.output_dir = Path(output_dir)
        self.clock = clock

        self.status = RecorderStatus.IDLE

        self.episode_id = None
        self.episode_dir = None

        self.stream_contracts = {}
        self.writers = {}
        self.stream_counts = {}

        self.started_at = None
        self.stopped_at = None

        self.metadata = {}

    def register_stream(self, name, contract, writer):

        if self.status != RecorderStatus.IDLE:
            raise RuntimeError(
                "Cannot register streams after starting."
            )

        if name in self.stream_contracts:
            raise ValueError("Stream already registered.")

        self.stream_contracts[name] = contract
        self.writers[name] = writer
        self.stream_counts[name] = 0

    def start(self, episode_id, metadata=None):

        if self.status != RecorderStatus.IDLE:
            raise RuntimeError("Recorder is not idle.")

        if not self.writers:
            raise RuntimeError("No registered streams.")

        if not episode_id or Path(episode_id).name != episode_id:
            raise ValueError("Invalid episode ID.")

        self.episode_id = episode_id

        self.episode_dir = (
            self.output_dir / f"episode_{episode_id}"
        )

        self.metadata = dict(metadata or {})
        self.metadata["episode_id"] = episode_id

        try:
            self.episode_dir.mkdir(
                parents=True,
                exist_ok=False
            )

            for writer in self.writers.values():
                writer.open(self.episode_dir)

            self.started_at = self.clock()
            self.status = RecorderStatus.RECORDING

        except Exception:
            self.status = RecorderStatus.ERROR
            self._close_writers()
            raise

    def record(self, stream_name, record):

        if self.status != RecorderStatus.RECORDING:
            raise RuntimeError("Recorder is not recording.")

        if stream_name not in self.stream_contracts:
            raise ValueError("Unknown stream.")

        contract = self.stream_contracts[stream_name]

        if not isinstance(record, contract):
            raise TypeError("Invalid record type.")

        record.validate()

        try:
            self.writers[stream_name].write(record)
            self.stream_counts[stream_name] += 1

        except Exception:
            self.status = RecorderStatus.ERROR
            self._close_writers()
            raise

    def stop(self, metadata=None):

        if self.status != RecorderStatus.RECORDING:
            raise RuntimeError("Recorder is not recording.")

        self.stopped_at = self.clock()

        self.metadata.update(metadata or {})
        self.metadata["episode_id"] = self.episode_id

        try:
            self._close_writers()

            with (
                self.episode_dir / "metadata.json"
            ).open("x") as file:
                json.dump(self.metadata, file, indent=2)

            self.status = RecorderStatus.STOPPED

            with (
                self.episode_dir / "recording_stats.json"
            ).open("x") as file:
                json.dump(self.get_stats(), file, indent=2)

        except Exception:
            self.status = RecorderStatus.ERROR
            raise

    def get_status(self):
        return self.status

    def get_stats(self):

        return {
            "episode_id": self.episode_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "stream_counts": dict(self.stream_counts)
        }

    def _close_writers(self):

        for writer in self.writers.values():
            writer.close()