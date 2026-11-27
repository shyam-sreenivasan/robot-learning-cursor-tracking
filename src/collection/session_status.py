from enum import Enum


class SessionStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    BATCH_COMPLETE = "batch_complete"
    COMPLETED = "completed"