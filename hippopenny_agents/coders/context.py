from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """Represents a single task for the coder."""

    id: int
    description: str
    status: str = "pending"  # e.g., pending, in_progress, completed, failed
    result: Any | None = None


@dataclass
class CoderContext:
    """Shared context for the planner and coder agents."""

    initial_request: str
    tasks: list[Task] = field(default_factory=list)
    current_task_id: int | None = None
    # Add any other relevant state, e.g., file paths, project structure
