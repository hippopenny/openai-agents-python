from typing import List, Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "in_progress", "done", "failed"]


class Task(BaseModel):
    id: int
    description: str
    status: TaskStatus = "pending"


class ProjectContext(BaseModel):
    project_goal: str
    tasks: List[Task] = Field(default_factory=list)
    coder_error: str | None = None # To store feedback if coder fails

    def get_next_pending_task(self) -> Task | None:
        for task in self.tasks:
            if task.status == "pending":
                return task
        return None

    def are_all_tasks_done(self) -> bool:
        if not self.tasks:
            return False # No tasks defined yet
        return all(task.status == "done" for task in self.tasks)

    def update_task_status(self, task_id: int, status: TaskStatus, error: str | None = None):
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                self.coder_error = error if status == "failed" else None
                return
        raise ValueError(f"Task with id {task_id} not found.")

    def add_tasks(self, task_descriptions: List[str]):
        start_id = len(self.tasks)
        for i, desc in enumerate(task_descriptions):
            self.tasks.append(Task(id=start_id + i, description=desc, status="pending"))

# Keep CoderOutput here for easy import in tests and tool function
class CoderOutput(BaseModel):
    status: Literal["completed", "failed", "needs_clarification"]
    summary: str = Field(description="Summary of the work done or explanation of failure/clarification needed.")
    code_changes: str | None = Field(default=None, description="Actual code changes or implementation details.")

