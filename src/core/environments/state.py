from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentState:
    cursor_pos: tuple[float, float]
    target_pos: tuple[int, int]
    target_radius: int
    button_down: bool = False