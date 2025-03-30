import asyncio
import json
import os
from typing import Tuple

# Ensure agents SDK is discoverable in the path
# This might be needed if running directly from the hippopenny_agents directory
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from agents import Runner, Agent, RunResult, ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem, MaxTurnsExceeded

from hippopenny_agents.aider.models import ProjectContext
# Import the new tools as well
from hippopenny_agents.aider.agents import PlannerAgent, CoderAgent, create_code_task_tool, plan_initial_tasks, add_task, modify_task

# Attempt to import console and Spinner, fallback if rich is not installed
try:
    from rich.console import Console
    from rich.spinner import Spinner
    console = Console()
except ImportError:
    console = None
    Spinner = None # type: ignore

def print_status(message: str, style: str = "blue"):
    # Avoid spinner for cleaner debug output for now
    # if console and Spinner:
    #     spinner = Spinner("dots", text=f" {message}")
    #     console.print(spinner, style=style)
    # else:
    print(f"... {message}")

def print_final(message: str, style: str = "green"):
     if console:
         console.print(f"[bold {style}]✓ {message}[/bold {style}]")
     else:
         print(f"✓ {message}")

def print_error(message: str, style: str = "red"):
     if console:
         console.print(f"[bold {style}]✗ {message}[/bold {style}]")
     else:
         print(f"✗ {message}")


async def run_aider(project_goal: str, max_turns: int = 15) -> Tuple[ProjectContext, str]: # Increased default max_turns
    """
    Runs the Planner-Coder agent loop using the "Agent as Tool" pattern.
    The Planner agent orchestrates the process, including dynamic plan updates.

    Args:
        project_goal: The high-level goal for the project.
        max_turns: Maximum number of Planner agent turns before stopping.

    Returns:
        A tuple containing the final ProjectContext and the final message from the Planner.
    """
    print_status(f"Starting Aider for goal: '{project_goal}'", style="yellow")

    # 1. Initialize Context
    project_context = ProjectContext(project_goal=project_goal)

    # 2. Create Agents and Tools
    # Coder Agent is defined in its module
    code_task_tool = create_code_task_tool(CoderAgent)
    # Planner Agent needs all tools now
    planner_agent_with_tools = PlannerAgent.clone(
        tools=[code_task_tool]
    )

    final_planner_message = "Aider run did not complete."
    planner_input_message = f"Start project: {project_goal}" # Initial trigger

    # 3. Main Loop - Run Planner until completion or max_turns
    for turn in range(max_turns):
        current_turn = turn + 1
        print_status(f"--- Planner Turn {current_turn}/{max_turns} ---", style="yellow")
        print_status(f"Current Tasks: {[(t.id, t.status) for t in project_context.tasks]}")
        print_status(f"Planner Input: '{planner_input_message}'")

        try:
            # Run the Planner Agent for one logical step
            # It will decide whether to plan, code, add, modify, or finish based on context
            planner_result = await Runner.run(
                planner_agent_with_tools,
                input=planner_input_message,
                context=project_context,
                max_turns=1 # Planner should decide its next action in one turn
            )

            # Extract the text output from the planner
            final_planner_message = ItemHelpers.text_message_outputs(planner_result.new_items)
            print_status(f"Planner Output: {final_planner_message}")

            # --- Check for Tool Calls (for logging/debug) ---
            tool_called = False
            tool_output_summary = "" # Capture output to feed back to planner
            for item in planner_result.new_items:
                 if isinstance(item, ToolCallItem):
                     tool_called = True
                     print_status(f"  Planner called tool: {item.tool_name} with input {item.tool_input}")
                 elif isinstance(item, ToolCallOutputItem):
                     tool_output_summary = item.output # Get the summary string returned by the tool
                     print_status(f"  Tool output received by planner: {tool_output_summary}")
                     # Use tool output as input for the next planner turn for context
                     planner_input_message = f"Tool '{item.tool_name}' finished. Result: {tool_output_summary}"


            # --- Check for Completion ---
            # The Planner's instructions tell it to stop calling tools when done.
            # We check the context state *after* the planner run.
            if project_context.are_all_tasks_done():
                print_final("Project completed successfully (all tasks marked 'done').")
                # Use the planner's final message if it provided one and didn't call a tool
                if not tool_called:
                     final_planner_message = final_planner_message or "Project finished successfully."
                else:
                     # If it called a tool on the last turn but tasks are done, override message
                     final_planner_message = "Project finished successfully."
                break

            # If the planner didn't call a tool and tasks are not done, it might be blocked or finished incorrectly.
            if not tool_called:
                 print_error(f"Planner did not call a tool, but project is not complete. Stopping. Final message: {final_planner_message}")
                 break

            # If no tool output was generated to form the next input (e.g., planner just talked), use a generic prompt
            if not tool_output_summary:
                 planner_input_message = "Continue managing the project based on the current task status and previous actions."


        except MaxTurnsExceeded:
             # This applies to the *inner* run of the Planner (max_turns=1)
             print_error(f"Planner exceeded its single turn limit. This shouldn't happen.")
             final_planner_message = "Error: Planner failed to respond in a single turn."
             break
        except Exception as e:
            print_error(f"Error during Planner turn {current_turn}: {e}")
            final_planner_message = f"Error occurred: {e}"
            # Optionally, mark the current task as failed if applicable
            # current_task = project_context.get_next_pending_task()
            # if current_task and current_task.status == "in_progress":
            #    project_context.update_task_status(current_task.id, "failed", f"Planner error: {e}")
            break

        # Optional: Add a small delay between turns if needed
        # await asyncio.sleep(0.5)

    else:
        # Loop finished without break (max_turns reached)
        print_error(f"Max turns ({max_turns}) reached for Planner orchestration. Stopping.")
        final_planner_message = f"Max turns ({max_turns}) reached. Project may be incomplete."

    print("\n--- Aider Run Finished ---")
    print("Final Task Status:")
    if not project_context.tasks:
        print("  No tasks were planned.")
    for task in project_context.tasks:
        print(f"  - Task {task.id}: {task.description} [{task.status}]")
        # Check for the last error associated with this task if it failed
        # Note: This assumes coder_error reflects the error for the *last* failed task update
        if task.status == 'failed' and project_context.coder_error:
             # A more robust system might store errors per-task
             print(f"    Last Error: {project_context.coder_error}")
    print(f"Final Planner Message: {final_planner_message}")
    print("-------------------------")

    return project_context, final_planner_message


async def main():
    # Example Usage
    project_goal = input("Enter the project goal: ")
    if not project_goal:
        project_goal = "Create a python script that prints 'Hello, Aider!' and then writes it to a file 'output.txt'."

    # Setup environment (e.g., API keys) if needed
    # You might need to load .env file or set environment variables
    # Example using python-dotenv:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file")
    except ImportError:
        print("dotenv not installed, skipping .env file load")

    # Ensure API key is available (replace with your actual key loading mechanism)
    if not os.getenv("OPENAI_API_KEY"):
         print_error("OPENAI_API_KEY environment variable not set.")
         return # Exit if key is missing

    # Optional: Set default client if needed globally, though Agent/Runner handles it
    # import agents
    # from openai import AsyncOpenAI
    # agents.set_default_openai_client(AsyncOpenAI())


    await run_aider(project_goal, max_turns=20) # Increased max_turns for dynamic planning

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAider stopped by user.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

