import math

from src.environments.action import ButtonEvent
from .task_result import TaskResult


class ClickTargetTask:

    def __init__(self, config):
        self.instruction = config.get(
            "operator.instructions"
        )

    def get_instruction(self):
        return self.instruction

    def reset(self):
        pass

    def evaluate(self, previous_state, action, current_state):

        if action.button_event != ButtonEvent.PRESS:
            return TaskResult()

        if previous_state.button_down:
            return TaskResult()

        distance = math.dist(
            current_state.cursor_pos,
            current_state.target_pos
        )

        radius = current_state.target_radius

        return TaskResult(
            attempted=True,
            success=distance <= radius,
            metrics={
                "click_error_px": distance,
                "normalized_click_error": distance / radius
            }
        )