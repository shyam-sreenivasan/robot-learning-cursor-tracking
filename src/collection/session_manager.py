import uuid
import random
import hashlib
import json

from datetime import datetime, timezone

from .session_status import SessionStatus
from .episode import Episode
from .recorder import Recorder

from .data_contracts import (
    ObservationRecord,
    ActionRecord,
    StateRecord
)

from src.environments.environment import Environment
from src.tasks.click_target_task import ClickTargetTask

from src.storage.stream_writers import (
    ImageWriter,
    CSVWriter
)
from copy import deepcopy
import yaml
from .episode_status import EpisodeStatus


class SessionManager:

    def __init__(self, config):

        self.config = config.clone()

        self.session_id = str(uuid.uuid4())
        self.status = SessionStatus.CREATED

        self.current_episode = None
        self.episode_history = []

        self.batch_size = self.config.get(
            "collection.episodes_per_batch"
        )

        self.batch_number = 1

        self.started_at = None
        self.ended_at = None

        self.session_dir = self.config.resolve_path(
            self.config.get("storage.raw_data_dir")
        ) / f"session_{self.session_id}"

        self.config_hash = self._compute_config_hash()

        self.random_seed = self.config.get(
            "collection.random_seed"
        )

        if self.random_seed is None:
            self.random_seed = random.randrange(2**32)

        self.rng = random.Random(self.random_seed)

    def start(self):

        if self.status != SessionStatus.CREATED:
            raise RuntimeError("Session already started.")

        self.session_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        self.started_at = self._now()
        self.status = SessionStatus.ACTIVE
        self._persist_metadata()

    def create_episode(self):

        if self.status != SessionStatus.ACTIVE:
            raise RuntimeError("Session is not active.")

        if self.current_episode is not None:
            raise RuntimeError(
                "Previous episode has not been registered."
            )

        environment = Environment(self.config)
        task = ClickTargetTask(self.config)

        recorder = self._create_recorder()

        episode = Episode(
            environment,
            task,
            self.config,
            recorder=recorder
        )

        seed = self.rng.randrange(2**32)

        episode.initialize(seed)

        self.current_episode = episode

        return episode

    def get_status(self):
        return self.status

    def _create_recorder(self):

        recorder = Recorder(self.session_dir)

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

        return recorder

    def _compute_config_hash(self):

        sections = [
            "schema_version",
            "experiment",
            "environment",
            "collection"
        ]

        config = {
            section: self.config.get(section)
            for section in sections
        }

        config["collection"].pop("random_seed", None)

        config["operator_instructions"] = self.config.get(
            "operator.instructions"
        )

        serialized = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":")
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()


    def complete_episode(self):

        if self.current_episode is None:
            raise RuntimeError("No active episode.")

        if not self.current_episode.is_finalized():
            raise RuntimeError("Episode has not been finalized.")

        metadata = self.current_episode.get_metadata()

        metadata["batch_number"] = self.batch_number

        metadata["directory"] = (
            f"episode_{self.current_episode.episode_id}"
        )

        self.episode_history.append(metadata)

        self.current_episode = None

        if self.is_batch_complete():
            self.status = SessionStatus.BATCH_COMPLETE

        self._persist_metadata()
        return metadata

    def is_batch_complete(self):

        return self.get_batch_statistics()["evaluable"] >= self.batch_size

    def start_next_batch(self):

        if self.status != SessionStatus.BATCH_COMPLETE:
            raise RuntimeError("Current batch is not complete.")

        self.batch_number += 1

        self.status = SessionStatus.ACTIVE
        self._persist_metadata()

    def end(self):

        if self.status not in (
            SessionStatus.ACTIVE,
            SessionStatus.BATCH_COMPLETE
        ):
            raise RuntimeError("Session cannot be ended.")

        if self.current_episode is not None:
            raise RuntimeError("An unfinished episode exists.")

        self.ended_at = self._now()

        self.status = SessionStatus.COMPLETED
        self._persist_metadata()

    @staticmethod
    def _calculate_statistics(episodes):

        successful = sum(
            ep["status"] == "success"
            for ep in episodes
        )

        failed = sum(
            ep["status"] == "failed"
            for ep in episodes
        )

        aborted = sum(
            ep["status"] == "aborted"
            for ep in episodes
        )

        evaluable = successful + failed

        return {
            "total": len(episodes),
            "successful": successful,
            "failed": failed,
            "aborted": aborted,
            "evaluable": evaluable,
            "success_rate": (
                successful / evaluable
                if evaluable > 0 else None
            )
        }


    def get_statistics(self):

        return self._calculate_statistics(
            self.episode_history
        )


    def get_batch_statistics(self):

        episodes = [
            ep for ep in self.episode_history
            if ep["batch_number"] == self.batch_number
        ]

        return self._calculate_statistics(episodes)

    def get_metadata(self):

        return {
            "session_id": self.session_id,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "batch_number": self.batch_number,
            "statistics": self.get_statistics(),
            "episodes": deepcopy(self.episode_history)
        }


    def _persist_metadata(self):

        manifest = self.config.to_dict()

        manifest["execution"] = self.get_metadata()

        path = self.session_dir / "session.yaml"
        temp_path = self.session_dir / "session.yaml.tmp"

        with temp_path.open("w") as file:
            yaml.safe_dump(
                manifest,
                file,
                sort_keys=False
            )

        temp_path.replace(path)

    def discard_episode(self):

        if self.current_episode is None:
            raise RuntimeError("No episode to discard.")

        if self.current_episode.get_status() != EpisodeStatus.READY:
            raise RuntimeError(
                "Only unstarted episodes can be discarded."
            )

        self.current_episode = None