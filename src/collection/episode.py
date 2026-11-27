import time
import uuid

from .episode_status import EpisodeStatus
from .interaction import Interaction


class Episode:

    def __init__(
        self,
        environment,
        task,
        config,
        clock=time.monotonic,
        recorder=None
    ):
        self.environment = environment
        self.task = task
        self.clock = clock

        self.episode_id = str(uuid.uuid4())
        self.timeout = config.get(
            "collection.episode_timeout_sec"
        )
        self.recorder = recorder
        self.finalized = False

        self.seed = None
        self.status = EpisodeStatus.CREATED

        self.started_at = None
        self.ended_at = None
        self.termination_reason = None

    def initialize(self, seed):

        if self.status != EpisodeStatus.CREATED:
            raise RuntimeError("Invalid episode state.")

        self.seed = seed

        state = self.environment.reset(seed)
        self.task.reset()

        self.status = EpisodeStatus.READY

        return state

    def start(self):

        if self.status != EpisodeStatus.READY:
            raise RuntimeError("Episode is not ready.")

        if self.recorder is not None:
            self.recorder.start(
                episode_id=self.episode_id,
                metadata={
                    "seed": self.seed,
                    "timeout_sec": self.timeout
                }
            )

        self.started_at = self.clock()
        self.status = EpisodeStatus.RUNNING

    def step(self, action):

        if self.status != EpisodeStatus.RUNNING:
            raise RuntimeError("Episode is not running.")

        timestamp = self.clock()

        if self.check_timeout(timestamp):
            return None

        previous_state = self.environment.get_state()

        current_state = self.environment.step(action)

        result = self.task.evaluate(
            previous_state,
            action,
            current_state
        )

        interaction = Interaction(
            timestamp=timestamp,
            previous_state=previous_state,
            action=action,
            current_state=current_state,
            task_result=result
        )

        if result.success:
            self._finish(
                EpisodeStatus.SUCCESS,
                "task_completed"
            )

        return interaction

    def check_timeout(self, timestamp=None):

        if self.status != EpisodeStatus.RUNNING:
            return False

        if timestamp is None:
            timestamp = self.clock()

        elapsed = timestamp - self.started_at

        if elapsed >= self.timeout:

            self._finish(
                EpisodeStatus.FAILED,
                "timeout"
            )

            return True

        return False

    def abort(self):

        if self.status != EpisodeStatus.RUNNING:
            raise RuntimeError("Episode is not running.")

        self._finish(
            EpisodeStatus.ABORTED,
            "user_abort"
        )

    def _finish(self, status, reason):

        self.ended_at = self.clock()
        self.status = status
        self.termination_reason = reason

    def get_duration(self):

        if self.started_at is None:
            return 0.0

        end = self.ended_at

        if end is None:
            end = self.clock()

        return end - self.started_at

    def get_status(self):
        return self.status

    def get_metadata(self):

        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "status": self.status.value,
            "duration_sec": self.get_duration(),
            "termination_reason": self.termination_reason
        }

    def is_running(self):
        return self.status == EpisodeStatus.RUNNING


    def is_terminal(self):
        return self.status in (
            EpisodeStatus.SUCCESS,
            EpisodeStatus.FAILED,
            EpisodeStatus.ABORTED
        )


    def is_finalized(self):
        return self.finalized

    def finalize(self):

        if not self.is_terminal():
            raise RuntimeError("Episode has not terminated.")

        if self.finalized:
            raise RuntimeError("Episode already finalized.")

        if self.recorder is not None:
            self.recorder.stop(
                metadata=self.get_metadata()
            )

        self.finalized = True