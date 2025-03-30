import asyncio
import os # Import os for environment variables if needed directly, or rely on aider module

from agents import Agent, function_tool
# Import the provider and config constants
from agents.models.hippopenny_aider_provider import HippoPennyAiderModelProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from agents.run import Runner
from agents.run_context import RunContextWrapper

from .aider import (
    AIDER_MODEL_NAME,
    AIDER_PROXY_API_KEY,
    AIDER_PROXY_BASE_URL,
)
from .context import CoderContext, Task
from .prompts import PlannerPrompt


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
    for task in context.tasks:
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
        max(task.id for task in context.tasks) + 1
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


@function_tool
def cmd_web(context: RunContextWrapper[CoderContext], 
            args="https://www.hippopenny.com/games/snake/") -> str:
    """
    An aider tool that is executed on aider server before llm runs.
    These tools won't be included in messages to llm.

    Args:
        args: the url to be scraped.
    """
    return f"Unexpected: received aider tool call cmd_. These tools should be run on server. No action taken here."


@function_tool
def cmd_clear(context: RunContextWrapper[CoderContext], 
            args="") -> str:
    """
    An aider tool that is executed on aider server before llm runs.
    These tools won't be included in messages to llm.

    Args:
        args: it's required to have a value.
    """
    return f"Unexpected: received aider tool call cmd_. These tools should be run on server. No action taken here."


def create_planner_agent() -> Agent[CoderContext]:
    """Creates the planner agent."""
   
    planner_model_provider = HippoPennyAiderModelProvider()
    planner_model = planner_model_provider.get_model("aider")
    planner_model_provider = OpenAIProvider()
    planner_model = planner_model_provider.get_model("gpt-4o-mini")
    prompt = PlannerPrompt()  # Create an instance of the PlannerPrompt to get the system message
    planner_agent = Agent[CoderContext](
        name="PlannerAgent",
        model=planner_model, # Pass the actual model instance
        instructions=prompt.get_system_message(),  # Use the PlannerPrompt to get the system message
        # tools=[update_task_status, add_task, get_tasks], # cmd_web, cmd_clear], 
    )
    return planner_agent

async def main(prompt: str):
    res = await Runner.run(planner_agent, prompt) 
    print(res.final_output)


if __name__ == "__main__":
    # This is just for testing the agent creation
    prompt = input("Enter your coding request: ")
    print(f"the prompt is: {prompt}")
    planner_agent = create_planner_agent()
    asyncio.run(main(prompt))  # Run the main function to test the planner agent
