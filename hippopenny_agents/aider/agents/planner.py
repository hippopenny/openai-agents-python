import json
from typing import List

from agents import (
    Agent,
    FunctionTool,
    Runner,
    RunContextWrapper,
    function_tool,
)

from hippopenny_agents.aider.models import ProjectContext, TaskStatus, CoderOutput


# --- Planner Agent Definition ---

# This agent manages the project plan based on the context.
PlannerAgent = Agent[ProjectContext](
    name="PlannerAgent",
    instructions=(
        "You are the project manager. Your goal is to break down the user's project_goal into a list of actionable tasks and manage their execution. "
        "You have access to the current project state (tasks and their statuses) via the context. "
        "You also have a 'code_task' tool to delegate implementation of a task to a CoderAgent.\n\n"
        "Your workflow is as follows:\n"
        "1. **Initial Planning:** If the task list is empty, analyze the project_goal and create a list of small, specific tasks. Add these tasks to the context (this happens implicitly based on your response asking to add tasks - the orchestrator will handle it if you output a list). Then, identify the first task to be coded.\n"
        "2. **Task Assignment:** Review the current tasks in the context. Find the *next* task with status 'pending'.\n"
        "3. **Delegation:** If a pending task is found, call the `code_task` tool with the exact description of that task.\n"
        "4. **Handle Coder Failure:** If the previous coder attempt failed (check context.coder_error), analyze the error and decide whether to retry the task (potentially with modifications) by calling `code_task` again, or mark the project as blocked.\n"
        "5. **Completion Check:** If there are no pending tasks, check if all tasks are 'done'.\n"
        "6. **Final Report:** If all tasks are 'done', state that the project is complete. Do NOT call any tools in this case. If tasks remain but none are pending (e.g., some failed and you decided not to retry), report the project status (e.g., blocked).\n\n"
        "**IMPORTANT:** Only call the `code_task` tool when you want the CoderAgent to work on a specific task. If the project is finished or blocked, just provide a final status message."
    ),
    # The tool will be added dynamically in main.py after the CoderAgent is available
    tools=[],
    # Expecting the planner to output text, potentially including a task list initially
    # Or just instructions to call the tool / final status.
    # output_type=None # Default text output is fine
)


# --- Tool Definition ---

def create_code_task_tool(coder_agent: Agent) -> FunctionTool:
    """
    Factory function to create the 'code_task' tool.
    This tool allows the PlannerAgent to delegate a task to the CoderAgent.
    It handles running the CoderAgent and updating the shared ProjectContext.
    """

    @function_tool(name_override="code_task")
    async def code_task(
        context: RunContextWrapper[ProjectContext],
        task_description: str,
    ) -> str:
        """
        Assigns the specified task to the CoderAgent for implementation.
        Updates the task status in the project context based on the CoderAgent's result.

        Args:
            task_description: The exact description of the task to be coded.
        """
        project_ctx = context.context
        task_to_run = next((task for task in project_ctx.tasks if task.description == task_description and task.status == "pending"), None)

        if not task_to_run:
             # Maybe the planner hallucinated or the task was already done/in_progress
             # Or maybe it's retrying a failed task
             task_to_run = next((task for task in project_ctx.tasks if task.description == task_description and task.status == "failed"), None)
             if not task_to_run:
                return f"Error: Task '{task_description}' not found in pending or failed state."

        # Mark task as in progress
        project_ctx.update_task_status(task_to_run.id, "in_progress")

        try:
            # Run the Coder Agent
            # Pass the *same context* object. CoderAgent won't modify it, but Runner needs it.
            # The CoderAgent's input is just the task description string.
            coder_result = await Runner.run(
                coder_agent,
                input=task_description,
                context=project_ctx,
                # Prevent coder from looping indefinitely if it fails internally
                max_turns=3
            )

            if not isinstance(coder_result.final_output, CoderOutput):
                 raise TypeError(f"CoderAgent returned unexpected type: {type(coder_result.final_output)}")

            coder_output: CoderOutput = coder_result.final_output

            # Update task status based on coder output
            new_status: TaskStatus
            if coder_output.status == "completed":
                new_status = "done"
            elif coder_output.status == "failed":
                new_status = "failed"
            else: # needs_clarification treated as failed for now
                new_status = "failed" # Or could introduce a 'clarification_needed' status

            project_ctx.update_task_status(task_to_run.id, new_status, error=coder_output.summary if new_status == "failed" else None)

            # Return the summary to the Planner
            return coder_output.summary

        except Exception as e:
            # Mark task as failed if the runner itself fails
            error_message = f"Error running CoderAgent: {e}"
            project_ctx.update_task_status(task_to_run.id, "failed", error=error_message)
            return error_message

    # Return the created FunctionTool instance
    # The type hint helps, but function_tool returns the correct type
    return code_task # type: ignore

