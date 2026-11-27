from dataclasses import dataclass

from src.core.environments.state import EnvironmentState
from src.core.environments.action import Action
from src.core.tasks.task_result import TaskResult


@dataclass(frozen=True)
class Interaction:
    timestamp: float
    previous_state: EnvironmentState
    action: Action
    current_state: EnvironmentState
    task_result: TaskResult