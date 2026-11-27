from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservationRecord:
    timestamp: float
    frame_id: int
    image: np.ndarray

    def validate(self):
        if not isinstance(self.image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")

        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Expected RGB image.")

        if self.image.dtype != np.uint8:
            raise ValueError("Expected uint8 image.")

        if self.frame_id < 0:
            raise ValueError("Invalid frame ID.")


@dataclass(frozen=True)
class ActionRecord:
    timestamp: float
    event_id: int
    delta_x: float
    delta_y: float
    button_event: str

    def validate(self):
        if self.event_id < 0:
            raise ValueError("Invalid event ID.")

        if self.button_event not in (
            "none", "press", "release"
        ):
            raise ValueError("Invalid button event.")


@dataclass(frozen=True)
class StateRecord:
    timestamp: float
    frame_id: int
    cursor_x: float
    cursor_y: float
    target_x: float
    target_y: float
    target_radius: int
    button_down: bool

    def validate(self):
        if self.frame_id < 0:
            raise ValueError("Invalid frame ID.")

        if self.target_radius <= 0:
            raise ValueError("Invalid target radius.")