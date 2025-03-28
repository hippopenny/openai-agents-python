from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Task:
    """Represents a single task for the coder."""

    id: int
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Any | None = None
    error_message: str | None = None


@dataclass
class CoderContext:
    """Shared context for the planner and coder agents."""

    initial_request: str
    tasks: list[Task] = field(default_factory=list)
    current_task_id: int | None = None
    # Add any other relevant state, e.g., file paths, project structure
