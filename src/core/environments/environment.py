import random
from dataclasses import replace

from .state import EnvironmentState
from .action import Action, ButtonEvent


class Environment:

    def __init__(self, config):

        self.width = config.get("environment.width")
        self.height = config.get("environment.height")

        self.min_radius = config.get(
            "environment.target.min_radius_px"
        )

        self.max_radius = config.get(
            "environment.target.max_radius_px"
        )

        self._state = None
        self._sampled_radius = None

    def reset(self, seed=None):

        rng = random.Random(seed)

        self._sampled_radius = rng.uniform(
            self.min_radius,
            self.max_radius
        )

        radius = max(1, round(self._sampled_radius))

        target_pos = (
            rng.randint(radius, self.width - radius - 1),
            rng.randint(radius, self.height - radius - 1)
        )

        cursor_pos = (
            rng.randint(0, self.width - 1),
            rng.randint(0, self.height - 1)
        )

        self._state = EnvironmentState(
            cursor_pos=cursor_pos,
            target_pos=target_pos,
            target_radius=radius,
            button_down=False
        )

        return self._state

    def step(self, action: Action):

        state = self.get_state()

        x, y = state.cursor_pos

        x = self._clamp(
            x + action.delta_x,
            0,
            self.width - 1
        )

        y = self._clamp(
            y + action.delta_y,
            0,
            self.height - 1
        )

        button_down = state.button_down

        if action.button_event == ButtonEvent.PRESS:
            button_down = True

        elif action.button_event == ButtonEvent.RELEASE:
            button_down = False

        self._state = replace(
            state,
            cursor_pos=(x, y),
            button_down=button_down
        )

        return self._state

    def get_state(self):

        if self._state is None:
            raise RuntimeError(
                "Environment has not been initialized."
            )

        return self._state

    def get_initialization_metadata(self):

        return {
            "sampled_target_radius": self._sampled_radius
        }

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))