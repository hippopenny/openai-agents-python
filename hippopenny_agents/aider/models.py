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
    _next_task_id: int = 0 # Internal counter for unique IDs

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
        found = False
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                # Store error only if the new status is failed, clear otherwise
                self.coder_error = error if status == "failed" else None
                found = True
                break
        if not found:
            raise ValueError(f"Task with id {task_id} not found for status update.")

    def add_tasks(self, task_descriptions: List[str]):
        """Adds multiple tasks to the end of the list."""
        for desc in task_descriptions:
            self._add_single_task(desc)

    def _add_single_task(self, description: str, insert_before_id: int | None = None) -> Task:
        """Adds a single task, optionally inserting it."""
        new_task = Task(id=self._next_task_id, description=description, status="pending")
        self._next_task_id += 1

        if insert_before_id is not None:
            insert_idx = -1
            for i, task in enumerate(self.tasks):
                if task.id == insert_before_id:
                    insert_idx = i
                    break
            if insert_idx != -1:
                self.tasks.insert(insert_idx, new_task)
            else:
                # If insert_before_id not found, append to the end
                self.tasks.append(new_task)
        else:
            self.tasks.append(new_task)
        return new_task

    def add_new_task(self, description: str, insert_before_id: int | None = None) -> Task:
        """Adds a single new task, exposed for the tool."""
        return self._add_single_task(description, insert_before_id)


    def modify_task(self, task_id: int, new_description: str | None = None, new_status: TaskStatus | None = None) -> bool:
        """Modifies the description and/or status of an existing task."""
        for task in self.tasks:
            if task.id == task_id:
                if new_description is not None:
                    task.description = new_description
                if new_status is not None:
                    # Use update_task_status to handle coder_error logic if status changes
                    self.update_task_status(task_id, new_status, self.coder_error) # Preserve existing error if status isn't 'failed'
                return True
        return False # Task not found


# Keep CoderOutput here for easy import in tests and tool function
class CoderOutput(BaseModel):
    status: Literal["completed", "failed", "needs_clarification"]
    summary: str = Field(description="Summary of the work done or explanation of failure/clarification needed.")
    code_changes: str | None = Field(default=None, description="Actual code changes or implementation details.")

