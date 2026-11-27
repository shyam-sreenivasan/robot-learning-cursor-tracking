from dataclasses import dataclass
from enum import Enum


class ButtonEvent(Enum):
    NONE = "none"
    PRESS = "press"
    RELEASE = "release"


@dataclass(frozen=True)
class Action:
    delta_x: float = 0.0
    delta_y: float = 0.0

    button_event: ButtonEvent = ButtonEvent.NONE