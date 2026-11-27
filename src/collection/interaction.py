from dataclasses import dataclass

from src.environments.state import EnvironmentState
from src.environments.action import Action
from src.tasks.task_result import TaskResult


@dataclass(frozen=True)
class Interaction:
    timestamp: float
    previous_state: EnvironmentState
    action: Action
    current_state: EnvironmentState
    task_result: TaskResult