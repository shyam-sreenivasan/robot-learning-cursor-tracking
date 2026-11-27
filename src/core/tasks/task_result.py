from dataclasses import dataclass


@dataclass(frozen=True)
class TaskResult:
    attempted: bool = False
    success: bool = False
    metrics: dict | None = None