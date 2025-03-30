import json
from typing import List, Optional

from agents import (
    Agent,
    FunctionTool,
    Runner,
    RunContextWrapper,
    function_tool,
    ModelBehaviorError, # To handle coder issues
)

from hippopenny_agents.aider.models import ProjectContext, TaskStatus, CoderOutput, Task
from .prompts import PlannerPrompt
from .coder import CoderAgent # Import CoderAgent for the implement_task tool


# --- Planner Tools ---

@function_tool
async def plan_initial_tasks(
    context: RunContextWrapper[ProjectContext],
    task_descriptions: List[str],
) -> str:
    """
    Adds the initial list of tasks to the project plan based on the project goal.
    Should only be called when the task list is empty.

    Args:
        task_descriptions: A list of strings, where each string is a description for a new task.
    """
    project_ctx = context.context
    if project_ctx.tasks:
        return "Error: Initial tasks have already been planned."
    if not task_descriptions:
        return "Error: No task descriptions provided."

    project_ctx.add_tasks(task_descriptions)
    return f"Successfully planned {len(task_descriptions)} initial tasks."

@function_tool
async def add_task(
    context: RunContextWrapper[ProjectContext],
    description: str,
    insert_before_id: int | None = None,
) -> str:
    """
    Adds a new task to the project plan.

    Args:
        description: The description of the new task.
        insert_before_id: Optional ID of an existing task before which the new task should be inserted. If None or not found, adds to the end.
    """
    project_ctx = context.context
    new_task = project_ctx.add_new_task(description, insert_before_id)
    return f"Successfully added new task '{new_task.description}' with ID {new_task.id}."

@function_tool
async def modify_task(
    context: RunContextWrapper[ProjectContext],
    task_id: int,
    new_description: str | None = None,
    new_status: TaskStatus | None = None,
) -> str:
    """
    Modifies the description and/or status of an existing task.

    Args:
        task_id: The ID of the task to modify.
        new_description: The new description for the task (optional).
        new_status: The new status for the task ('pending', 'in_progress', 'done', 'failed') (optional).
    """
    project_ctx = context.context
    try:
        # Find the task first to provide better feedback
        task_to_modify = None
        for task in project_ctx.tasks:
            if task.id == task_id:
                task_to_modify = task
                break
        if not task_to_modify:
             return f"Error: Task with ID {task_id} not found."

        original_desc = task_to_modify.description
        modified = project_ctx.modify_task(task_id, new_description, new_status)

        if modified:
            updates = []
            if new_description:
                updates.append(f"description updated to '{new_description}'")
            if new_status:
                updates.append(f"status updated to '{new_status}'")
            return f"Successfully modified task {task_id} ('{original_desc}'): {', '.join(updates)}."
        else:
            # This case should ideally be caught above, but added for safety
            return f"Error: Failed to modify task {task_id}. Task not found."
    except ValueError as e:
        return f"Error modifying task {task_id}: {e}"
    except Exception as e:
        # Catch unexpected errors during modification
        return f"Unexpected error modifying task {task_id}: {e}"


@function_tool
async def implement_task(
    context: RunContextWrapper[ProjectContext],
    task_id: int,
) -> str:
    """
    Assigns the specified task (by ID) to the Coder agent for implementation.
    Updates the task status to 'in_progress', runs the coder, and then updates
    to 'done' or 'failed' based on the coder's output. Stores coder errors.

    Args:
        task_id: The ID of the task to be implemented.
    """
    project_ctx = context.context
    task_to_implement: Task | None = None
    for task in project_ctx.tasks:
        if task.id == task_id:
            task_to_implement = task
            break

    if task_to_implement is None:
        return f"Error: Task with ID {task_id} not found."

    if task_to_implement.status != "pending":
        return f"Error: Task {task_id} ('{task_to_implement.description}') cannot be implemented because its status is '{task_to_implement.status}' (must be 'pending')."

    # --- Mark as in_progress ---
    print(f"[Planner Tool] Setting task {task_id} ('{task_to_implement.description}') to 'in_progress'")
    project_ctx.update_task_status(task_id, "in_progress") # Clear previous error if any

    coder_output: CoderOutput | None = None
    coder_error_detail: str | None = None
    final_status: TaskStatus = "failed" # Default to failed unless coder succeeds
    final_summary = "Coder execution failed or produced no output." # Default summary

    try:
        # --- Run the Coder Agent ---
        print(f"[Planner Tool] Running CoderAgent for task {task_id}: '{task_to_implement.description}'")
        # Note: We pass the *project context* to the runner, but the CoderAgent's prompt
        # currently only expects the task description as input. The context is available
        # if the CoderAgent or its potential future tools need it.
        coder_result = await Runner.run(
            CoderAgent, # The actual CoderAgent instance
            input=task_to_implement.description,
            context=project_ctx, # Pass the main context
            max_turns=3 # Prevent coder loops
        )

        if isinstance(coder_result.final_output, CoderOutput):
            coder_output = coder_result.final_output
            final_summary = coder_output.summary # Use coder's summary
            print(f"[Planner Tool] CoderAgent finished task {task_id}. Status: {coder_output.status}, Summary: {coder_output.summary}")
            if coder_output.status == "completed":
                final_status = "done"
                coder_error_detail = None # Clear error on success
            elif coder_output.status == "failed":
                final_status = "failed"
                coder_error_detail = coder_output.summary # Store failure reason
            elif coder_output.status == "needs_clarification":
                 final_status = "failed" # Treat clarification needed as failure for planner
                 coder_error_detail = f"Needs Clarification: {coder_output.summary}"
                 final_summary = coder_error_detail # Update summary for planner feedback
            else:
                 # Should not happen with Literal types, but defensively handle
                 final_status = "failed"
                 coder_error_detail = f"Coder returned unknown status: {coder_output.status}"
                 final_summary = coder_error_detail

        else:
            # Coder failed to return the expected Pydantic model
            error_message = f"CoderAgent returned unexpected output type: {type(coder_result.final_output)}. Raw output: {coder_result.final_output}"
            print(f"[Planner Tool Error] {error_message}")
            final_status = "failed"
            coder_error_detail = error_message
            final_summary = error_message
            # Raise ModelBehaviorError so the Planner knows the tool failed internally?
            # For now, just report failure via status and summary.
            # raise ModelBehaviorError(error_message)

    except Exception as e:
        # Error during the Runner.run call itself
        error_message = f"Error running CoderAgent for task {task_id}: {e}"
        print(f"[Planner Tool Error] {error_message}")
        final_status = "failed"
        coder_error_detail = error_message
        final_summary = error_message
        # raise # Option: Re-raise to potentially halt the main loop

    # --- Update task status based on coder outcome ---
    print(f"[Planner Tool] Updating task {task_id} final status to '{final_status}'")
    project_ctx.update_task_status(task_id, final_status, error=coder_error_detail)

    # Return the summary from the coder (or error message) to the Planner
    return final_summary


# --- Planner Agent Definition ---

planner_prompt_obj = PlannerPrompt() # Create instance to access method
PlannerAgent = Agent[ProjectContext](
    name="PlannerAgent",
    instructions=planner_prompt_obj.get_system_message(),
    tools=[
        plan_initial_tasks,
        add_task,
        modify_task,
        implement_task,
    ],
    # Expecting text output (completion message) or tool calls.
    # output_type=None # Default is fine
)
