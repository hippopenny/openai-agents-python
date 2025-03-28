import asyncio
import os # Import os for environment variables if needed directly, or rely on aider module

from agents import Agent, function_tool
# Import the provider and config constants
from agents.models.hippopenny_aider_provider import HippoPennyAiderModelProvider
from agents.run import Runner
from agents.run_context import RunContextWrapper

from .aider import (
    AIDER_MODEL_NAME,
    AIDER_PROXY_API_KEY,
    AIDER_PROXY_BASE_URL,
    create_aider_agent,
)
from .context import CoderContext, Task


# Refined instructions
PLANNER_INSTRUCTIONS = """
You are a planning agent responsible for breaking down a user's coding request into a series of actionable tasks for a coder agent named 'aider'.

Your primary goal is to create an initial plan.

1.  **Analyze Request:** Understand the user's coding request provided in the `initial_request` field of the context.
2.  **Create Tasks:** Generate a sequence of specific, actionable coding tasks. Use the `add_task` tool *repeatedly* to add each task to the plan. Ensure tasks are well-defined and can be executed independently by the 'aider' agent.
3.  **Task Granularity:** Break down complex requests into smaller, manageable steps.
4.  **Replanning (If Necessary):** If you are invoked later and the context contains tasks marked as 'failed', analyze the `error_message` for the failed task(s). You may need to use `add_task` to create new corrective tasks or potentially use `update_task_status` to modify an existing task's description and reset its status if a simple retry with adjustment is feasible.

**Available Tools:**
*   `add_task(description: str)`: Adds a new task to the end of the list. Use this for *all* initial task creation.
*   `get_tasks()`: Retrieves the current task list and statuses. Useful for context during replanning.
*   `update_task_status(task_id: int, status: str, result: str | None = None)`: Use *sparingly* during replanning to modify an existing task's status or result. The main control loop handles standard `in_progress`, `completed`, `failed` updates.

**Handoffs:**
*   You can theoretically hand off to `aider_agent` if complex interaction is needed, but the standard flow involves the orchestrator calling aider for each task.

Start by creating the initial task list based on the user's request. Do not execute the tasks yourself.
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
    # Define aider agent here primarily so it can be listed in handoffs,
    # Configure the HippoPennyAiderModelProvider for the planner
    # It takes no arguments, presumably reads config from environment or defaults
    planner_model_provider = HippoPennyAiderModelProvider()

    # Get the actual model instance from the provider
    # Assuming a .get_model() method exists. Adjust if the method name is different.
    planner_model = planner_model_provider.get_model("aider")

    planner_agent = Agent[CoderContext](
        name="PlannerAgent",
        model=planner_model, # Pass the actual model instance
        instructions=PLANNER_INSTRUCTIONS,
        tools=[update_task_status, add_task, get_tasks],
        # Refer to the handoff agent by its name
        handoffs=["AiderAgent"],
    )
    return planner_agent

async def main():
    await Runner.run(planner_agent, "Create a function to calculate Fibonacci numbers.")  # Example usage


if __name__ == "__main__":
    # This is just for testing the agent creation
    planner_agent = create_planner_agent()
    print(f"Created {planner_agent.name} with instructions: {planner_agent.instructions}")
    # You can add more tests or run the agent in a loop as needed
    # Note: The actual agent execution and interaction would typically be handled
    # by a higher-level orchestration script or framework.
    asyncio.run(main())  # Run the main function to test the planner agent
