from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValidationStatus(str, Enum):

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ValidationResult:

    check_name: str
    status: ValidationStatus
    message: str
    expected: Any = None
    observed: Any = None