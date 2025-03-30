import json
from typing import List, Optional

from agents import (
    Agent,
    FunctionTool,
    Runner,
    RunContextWrapper,
    function_tool,
)

from hippopenny_agents.aider.models import ProjectContext, TaskStatus, CoderOutput


# --- Tool Definitions ---

@function_tool(name_override="plan_initial_tasks")
async def plan_initial_tasks(
    context: RunContextWrapper[ProjectContext],
    task_descriptions: List[str],
) -> str:
    """
    Adds the initial list of planned tasks to the project context.
    This should only be called once at the beginning of the project if the task list is empty.

    Args:
        task_descriptions: A list of strings, where each string is a description of a task.
    """
    project_ctx = context.context
    if project_ctx.tasks:
        return "Error: Tasks already exist. Cannot re-plan."
    if not task_descriptions:
        return "Error: No task descriptions provided."

    project_ctx.add_tasks(task_descriptions)
    return f"Successfully planned {len(task_descriptions)} tasks."


@function_tool(name_override="add_task")
async def add_task(
    context: RunContextWrapper[ProjectContext],
    description: str,
    insert_before_task_id: Optional[int] = None,
) -> str:
    """
    Adds a new task to the project plan. Useful if a prerequisite step was missed
    or a task needs to be broken down further.

    Args:
        description: The description of the new task.
        insert_before_task_id: Optional. If provided, insert the new task immediately before the task with this ID. If omitted or ID not found, appends to the end.
    """
    project_ctx = context.context
    try:
        new_task = project_ctx.add_new_task(description, insert_before_task_id)
        return f"Successfully added new task {new_task.id}: '{description}'."
    except Exception as e:
        return f"Error adding task: {e}"


@function_tool(name_override="modify_task")
async def modify_task(
    context: RunContextWrapper[ProjectContext],
    task_id: int,
    new_description: Optional[str] = None,
) -> str:
    """
    Modifies the description of an existing task. Useful for clarifying instructions
    or refining a task based on feedback. Cannot change task status directly (status is updated via code_task).

    Args:
        task_id: The ID of the task to modify.
        new_description: The new description for the task.
    """
    project_ctx = context.context
    if new_description is None:
        return "Error: No changes specified. Please provide a new_description."

    try:
        success = project_ctx.modify_task(task_id=task_id, new_description=new_description)
        if success:
            return f"Successfully modified task {task_id}."
        else:
            return f"Error: Task with ID {task_id} not found for modification."
    except Exception as e:
        return f"Error modifying task {task_id}: {e}"


def create_code_task_tool(coder_agent: Agent) -> FunctionTool:
    """
    Factory function to create the 'code_task' tool.
    This tool allows the PlannerAgent to delegate a task to the CoderAgent.
    It handles running the CoderAgent and updating the shared ProjectContext.
    """

    @function_tool(name_override="code_task")
    async def code_task(
        context: RunContextWrapper[ProjectContext],
        task_id: int,
        task_description: str, # Include description for clarity and potential retry logic
    ) -> str:
        """
        Assigns the specified task (by ID) to the CoderAgent for implementation.
        Updates the task status in the project context based on the CoderAgent's result.
        Returns a summary of the outcome (success, failure, needs_clarification).

        Args:
            task_id: The unique ID of the task to be coded.
            task_description: The description of the task (for CoderAgent input).
        """
        project_ctx = context.context
        task_to_run = next((task for task in project_ctx.tasks if task.id == task_id), None)

        if not task_to_run:
            return f"Error: Task with ID {task_id} not found."

        if task_to_run.status not in ["pending", "failed"]:
             return f"Error: Task {task_id} is not in 'pending' or 'failed' state (current state: {task_to_run.status}). Cannot execute."

        # Mark task as in progress
        original_status = task_to_run.status
        project_ctx.update_task_status(task_to_run.id, "in_progress")

        try:
            # Run the Coder Agent
            print(f"[Aider Debug] Running CoderAgent for Task {task_id}: {task_description}")
            coder_result = await Runner.run(
                coder_agent,
                input=task_description,
                context=project_ctx, # Pass context for potential future use by coder/tools
                max_turns=3 # Prevent coder from looping indefinitely
            )
            print(f"[Aider Debug] CoderAgent for Task {task_id} finished.")


            if not isinstance(coder_result.final_output, CoderOutput):
                 # If coder failed to produce structured output, mark task as failed
                 error_message = f"CoderAgent returned unexpected output type: {type(coder_result.final_output)}. Raw output: {coder_result.final_output}"
                 print(f"[Aider Error] {error_message}")
                 project_ctx.update_task_status(task_to_run.id, "failed", error=error_message)
                 # Return a clear failure summary
                 return f"Coder agent failed to produce valid output for task {task_id}. Status: failed. Summary: {error_message}"


            coder_output: CoderOutput = coder_result.final_output
            print(f"[Aider Debug] CoderAgent output status: {coder_output.status}, summary: {coder_output.summary}")


            # Update task status based on coder output
            if coder_output.status == "completed":
                project_ctx.update_task_status(task_to_run.id, "done", error=None)
                return f"Task {task_id} completed successfully. Status: completed. Summary: {coder_output.summary}"
            elif coder_output.status == "failed":
                project_ctx.update_task_status(task_to_run.id, "failed", error=coder_output.summary)
                return f"Task {task_id} failed. Status: failed. Summary: {coder_output.summary}"
            else: # needs_clarification
                # Revert status to original (pending or failed) and store error
                project_ctx.update_task_status(task_to_run.id, original_status, error=f"Clarification needed: {coder_output.summary}")
                # Return summary indicating clarification needed
                return f"Task {task_id} needs clarification. Status: needs_clarification. Summary: {coder_output.summary}"

        except Exception as e:
            # Mark task as failed if the runner itself fails
            error_message = f"Error running CoderAgent for task {task_id}: {e}"
            print(f"[Aider Error] {error_message}")
            project_ctx.update_task_status(task_to_run.id, "failed", error=error_message)
            return f"Error running CoderAgent for task {task_id}. Status: failed. Summary: {error_message}"

    # Return the created FunctionTool instance
    return code_task # type: ignore


# --- Planner Agent Definition ---

# This agent manages the project plan based on the context.
PlannerAgent = Agent[ProjectContext](
    name="PlannerAgent",
    instructions=(
        "You are the project manager. Your goal is to orchestrate the completion of a project defined by `project_goal` in the context.\n"
        "You have access to the current project state (tasks and their statuses) via the shared `ProjectContext`.\n"
        "You have tools: `plan_initial_tasks`, `code_task`, `add_task`, `modify_task`.\n\n"
        "**Your Workflow:**\n"
        "1.  **Check Task List:** Examine `context.tasks`.\n"
        "2.  **Initial Planning:** If `context.tasks` is EMPTY:\n"
        "    a. Analyze the `context.project_goal`.\n"
        "    b. Create a list of small, specific, sequential task descriptions needed to achieve the goal.\n"
        "    c. Call the `plan_initial_tasks` tool ONCE with the list of task descriptions.\n"
        "3.  **Task Execution & Dynamic Planning:** If `context.tasks` is NOT empty:\n"
        "    a. Check if all tasks are 'done' using `context.are_all_tasks_done()`.\n"
        "    b. If ALL tasks are 'done', your job is finished. Respond with a final confirmation message (e.g., 'Project completed successfully.') and DO NOT call any more tools.\n"
        "    c. **Analyze Last Action:** Review the result from the previous tool call (if any). \n"
        "       - If `code_task` reported 'failed' or 'needs_clarification': Analyze the summary and `context.coder_error`. Decide the best course:\n"
        "           i. **Modify:** If the task description was unclear, call `modify_task` with the `task_id` and a `new_description`. Then, likely call `code_task` again for the *same* task ID in the *next* turn.\n"
        "           ii. **Add Prerequisite:** If a step was missing, call `add_task` with the description of the missing step, potentially using `insert_before_task_id` to place it correctly. Then proceed to execute the *new* task in the next turn.\n"
        "           iii. **Retry:** If it was a transient error, you might decide to call `code_task` again for the same `task_id` without modification.\n"
        "           iv. **Block:** If the failure is unrecoverable, mark the project as blocked and explain why (without calling tools).\n"
        "       - If `code_task` reported 'completed': Proceed to find the next pending task.\n"
        "       - If `add_task` or `modify_task` was just called: Proceed to find the next pending task (which might be the one just added/modified if it's now pending).\n"
        "    d. **Find Next Task:** Find the *next* task with status 'pending' using `context.get_next_pending_task()`.\n"
        "    e. **Delegate:** If a 'pending' task is found, call the `code_task` tool with its `task_id` and `task_description`.\n"
        "    f. **Handle Blockage:** If NO 'pending' tasks are found, but not all tasks are 'done' (e.g., all remaining are 'failed' and you decided not to retry), report the project status (e.g., blocked) and stop.\n\n"
        "**IMPORTANT RULES:**\n"
        "-   Only call `plan_initial_tasks` if the task list is empty.\n"
        "-   Only call `code_task` for tasks that are 'pending' or 'failed'.\n"
        "-   Use `add_task` or `modify_task` *reactively* based on coder feedback or identified plan gaps.\n"
        "-   Always check `context.are_all_tasks_done()` before deciding the next step.\n"
        "-   If all tasks are done, provide a final message and STOP.\n"
        "-   Base your decisions *strictly* on the information in the `ProjectContext` and the results from your tools."
    ),
    # Tools will be added dynamically in main.py
    tools=[],
    # Expecting text output, or tool calls.
    # output_type=None # Default text output is fine
)
