from enum import Enum


class EpisodeStatus(Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"