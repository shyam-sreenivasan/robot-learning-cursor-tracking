from enum import Enum


class RecorderStatus(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"
    ERROR = "error"