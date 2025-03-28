from agents import Agent, function_tool
from agents.run_context import RunContextWrapper

from .context import CoderContext, Task


# Placeholder instructions - refine these based on desired planner behavior
PLANNER_INSTRUCTIONS = """
You are a planning agent responsible for breaking down a user's coding request into a series of actionable tasks for a coder agent named 'aider'.

Your responsibilities:
1. Analyze the user's request.
2. Create a list of specific, sequential coding tasks.
3. Number the tasks starting from 1.
4. Keep track of the status of each task.
5. Update the task list based on the results provided by the aider agent.
6. If aider fails a task, you may need to revise the plan or ask for clarification.
7. Once all tasks are complete, summarize the results.

Available tools:
- `update_task_status`: Use this to mark tasks as 'in_progress', 'completed', or 'failed'.
- `add_task`: Use this to add new tasks if needed during the process.
- `get_tasks`: Use this to retrieve the current list of tasks and their statuses.

Interact with the user or the aider agent as needed to clarify requirements or report progress/completion.
Start by creating the initial task list based on the user's request in the context.
"""


@function_tool
def update_task_status(
    context: RunContextWrapper[CoderContext], task_id: int, status: str, result: str | None = None
) -> str:
    """
    Updates the status and result of a specific task.

    Args:
        task_id: The ID of the task to update.
        status: The new status (e.g., 'in_progress', 'completed', 'failed').
        result: Any output or result message from the coder for this task.
    """
    task_found = False
    for task in context.actual_context.tasks:
        if task.id == task_id:
            task.status = status
            task.result = result
            task_found = True
            break
    if not task_found:
        return f"Error: Task with ID {task_id} not found."
    return f"Task {task_id} status updated to {status}."


@function_tool
def add_task(context: RunContextWrapper[CoderContext], description: str) -> str:
    """
    Adds a new task to the end of the task list.

    Args:
        description: The description of the new task.
    """
    new_task_id = (
        max(task.id for task in context.actual_context.tasks) + 1
        if context.actual_context.tasks
        else 1
    )
    new_task = Task(id=new_task_id, description=description, status="pending")
    context.actual_context.tasks.append(new_task)
    return f"Task {new_task_id} added: {description}"


@function_tool
def get_tasks(context: RunContextWrapper[CoderContext]) -> str:
    """
    Retrieves the current list of tasks and their statuses.
    """
    if not context.actual_context.tasks:
        return "No tasks defined yet."
    task_list_str = "\n".join(
        [f"Task {task.id}: {task.description} (Status: {task.status})" for task in context.actual_context.tasks]
    )
    return f"Current Tasks:\n{task_list_str}"


def create_planner_agent() -> Agent[CoderContext]:
    """Creates the planner agent."""
    planner_agent = Agent[CoderContext](
        instructions=PLANNER_INSTRUCTIONS,
        tools=[update_task_status, add_task, get_tasks],
        # Add model configuration if needed
    )
    return planner_agent
