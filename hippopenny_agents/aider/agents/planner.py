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
from .prompts import PlannerPrompt




def create_code_task_tool(coder_agent: Agent) -> FunctionTool:
    """
    Factory function to create the 'code_task' tool.
    This tool allows the PlannerAgent to delegate a task to the CoderAgent.
    It handles running the CoderAgent and updating the shared ProjectContext.
    """

    @function_tool(name_override="implement")
    async def aider_implement(
        context: RunContextWrapper[ProjectContext],
        task_description: str, # Include description for clarity and potential retry logic
    ) -> str:
        """
        Implement the tasks to build the software project.
        Returns a summary of the outcome (success, failure, needs_clarification).

        Args:
            task_description: The description of the tasks for the software engineer to implement.
        """

        try:
            # Run the Coder Agent
            coder_result = await Runner.run(
                coder_agent,
                input=task_description,
                context=context, # Pass context for potential future use by coder/tools
                max_turns=3 # Prevent coder from looping indefinitely
            )

            if not isinstance(coder_result.final_output, CoderOutput):
                 # If coder failed to produce structured output, mark task as failed
                 error_message = f"CoderAgent returned unexpected output type: {type(coder_result.final_output)}. Raw output: {coder_result.final_output}"
                 print(f"[Aider Error] {error_message}")
                 # Return a clear failure summary
                 return f"Coder agent failed to produce valid output for tasks. Status: failed. Summary: {error_message}"


            coder_output: CoderOutput = coder_result.final_output
            print(f"[Aider Debug] CoderAgent output status: {coder_output.status}, summary: {coder_output.summary}")
            
        except Exception as e:
            # Mark task as failed if the runner itself fails
            error_message = f"Error running CoderAgent for tasks: {e}"
            print(f"[Aider Error] {error_message}")
            return f"Error running CoderAgent for tasks. Status: failed. Summary: {error_message}"

    # Return the created FunctionTool instance
    return aider_implement # type: ignore


# --- Planner Agent Definition ---

# This agent manages the project plan based on the context.
planner_prompt = PlannerPrompt()
PlannerAgent = Agent[ProjectContext](
    name="PlannerAgent",
    instructions=planner_prompt.get_system_message(),
    # Tools will be added dynamically in main.py
    tools=[],
    # Expecting text output, or tool calls.
    # output_type=None # Default text output is fine
)
