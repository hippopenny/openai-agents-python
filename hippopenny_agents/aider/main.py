import asyncio
import json
import os
from typing import Tuple

# Ensure agents SDK is discoverable in the path
# This might be needed if running directly from the hippopenny_agents directory
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from agents import (
    Runner, Agent, RunResult, ItemHelpers, MessageOutputItem,
    ToolCallItem, ToolCallOutputItem, MaxTurnsExceeded, ResponseOutputText
)

from hippopenny_agents.aider.models import ProjectContext
# Import the revised PlannerAgent which now includes its tools
from hippopenny_agents.aider.agents import PlannerAgent

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


async def run_aider(project_goal: str, max_turns: int = 20) -> Tuple[ProjectContext, str]: # Increased default max_turns
    """
    Runs the Planner-Coder agent loop using the refined Planner agent and tools.
    The Planner agent orchestrates the process, managing tasks and delegating
    implementation via tools.

    Args:
        project_goal: The high-level goal for the project.
        max_turns: Maximum number of Planner agent turns before stopping.

    Returns:
        A tuple containing the final ProjectContext and the final message from the Planner.
    """
    print_status(f"Starting Aider for goal: '{project_goal}'", style="yellow")

    # 1. Initialize Context
    project_context = ProjectContext(project_goal=project_goal)

    # 2. Planner Agent (already includes its tools)
    # No separate tool creation needed here anymore
    planner_agent = PlannerAgent

    final_planner_message = "Aider run did not complete."
    # Initial input for the planner
    planner_input = f"Project Goal: {project_goal}. The task list is currently empty. Please plan the initial tasks."

    # 3. Main Loop - Run Planner until completion or max_turns
    for turn in range(max_turns):
        current_turn = turn + 1
        print_status(f"\n--- Planner Turn {current_turn}/{max_turns} ---", style="yellow")
        # Display tasks with status and ID for clarity
        task_summary = [(t.id, t.description, t.status) for t in project_context.tasks]
        print(f"Current Tasks: {task_summary if task_summary else 'No tasks yet.'}")
        if project_context.coder_error:
             # Find which task the error belongs to (the most recent failed one)
             failed_task_id = None
             for task in reversed(project_context.tasks):
                 if task.status == 'failed':
                     failed_task_id = task.id
                     break
             error_context = f" (for Task {failed_task_id})" if failed_task_id is not None else ""
             print_error(f"Last Coder Error{error_context}: {project_context.coder_error}")
        print(f"Planner Input: '{planner_input}'")

        try:
            # Run the Planner Agent for one logical step
            planner_result: RunResult[ProjectContext] = await Runner.run(
                planner_agent,
                input=planner_input,
                context=project_context, # Pass the mutable context
                max_turns=1 # Planner should decide its next action in one turn
            )

            # --- Process Planner Output ---
            planner_text_output = ""
            tool_called = False
            tool_name_called = ""
            tool_output_summary = "" # Capture output from the tool run

            # Process items generated in this turn
            for item in planner_result.new_items:
                if isinstance(item, MessageOutputItem):
                    # Extract text response from the planner
                    last_content = item.content[-1] if item.content else None
                    if isinstance(last_content, ResponseOutputText):
                        planner_text_output = last_content.text
                        print_status(f"Planner Message: {planner_text_output}")
                elif isinstance(item, ToolCallItem):
                    tool_called = True
                    tool_name_called = item.tool_name
                    # Log the tool call the planner decided to make
                    print_status(f"  Planner called tool: {item.tool_name} with input {item.tool_input}")
                elif isinstance(item, ToolCallOutputItem):
                    # Capture the result returned by the tool function
                    tool_output_summary = str(item.output) # Ensure it's a string
                    print_status(f"  Tool '{item.tool_name}' executed. Result: {tool_output_summary}")

            # --- Determine Next Planner Input ---
            if tool_called:
                # If a tool was called, feed its result back to the planner
                # The context was already modified *by the tool function*
                planner_input = f"Previous action: Called tool '{tool_name_called}'. Result: {tool_output_summary}. Current task status is updated in the context. Decide the next step based on the project goal and task list."
            elif planner_text_output:
                # If the planner just responded with text (e.g., completion message, clarification)
                # Use its text output as the final message if the loop ends here.
                final_planner_message = planner_text_output
                # Check if the project is actually done according to context
                if project_context.are_all_tasks_done():
                    print_final("Planner indicated completion, and all tasks are 'done'.")
                    break # Project is complete
                else:
                    # Planner talked but didn't finish and didn't call a tool. Might be asking a question or stuck.
                    print_error(f"Planner provided text output but project is not complete. Stopping. Planner message: {planner_text_output}")
                    break # Stop if planner talked but didn't finish
            else:
                 # Planner didn't call a tool and didn't provide text output (shouldn't happen with max_turns=1)
                 print_error("Planner did not call a tool or provide text output. Stopping.")
                 final_planner_message = "Error: Planner failed to produce output."
                 break

            # --- Check for Completion State (after processing turn results) ---
            if project_context.are_all_tasks_done():
                print_final("Project completed successfully (all tasks marked 'done' in context).")
                # Use the planner's final text message if it provided one, otherwise use a default.
                final_planner_message = planner_text_output or "Project finished successfully."
                break

        except MaxTurnsExceeded:
             # This applies to the *inner* run of the Planner (max_turns=1)
             print_error(f"Planner exceeded its single turn limit. This shouldn't happen.")
             final_planner_message = "Error: Planner failed to respond in a single turn."
             break
        except Exception as e:
            print_error(f"Error during Planner turn {current_turn}: {e}")
            final_planner_message = f"Error occurred: {e}"
            # Optionally mark the current task as failed if applicable and identifiable
            # (This might be complex if the error happened before tool execution)
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
        status_color = "green" if task.status == "done" else "red" if task.status == "failed" else "yellow" if task.status == "in_progress" else "blue"
        print_status(f"  - Task {task.id}: {task.description} [{task.status}]", style=status_color)
        # Check for the last error associated with this task if it failed
        if task.status == 'failed' and project_context.coder_error:
             # Check if this task is the one associated with the current coder_error
             # (Simple check: assume error belongs to the last task updated to 'failed')
             # A more robust system might store errors per-task or link error to task ID.
             is_last_failed = True # Simplistic assumption for now
             # More complex check: find highest ID among failed tasks, or store last failed ID
             # last_failed_id = None
             # for t in reversed(project_context.tasks):
             #     if t.status == 'failed':
             #         last_failed_id = t.id
             #         break
             # if task.id == last_failed_id:
             if is_last_failed:
                 print_error(f"    Last Error: {project_context.coder_error}")
    print(f"Final Planner Message: {final_planner_message}")
    print("-------------------------")

    return project_context, final_planner_message


async def main():
    # Example Usage
    project_goal = input("Enter the project goal: ")
    if not project_goal:
        project_goal = "Create a python script 'hello.py' that prints 'Hello, Aider!' to the console."
        # project_goal = "Create a python script that prints 'Hello, Aider!' and then writes it to a file 'output.txt'." # More complex goal

    # Setup environment (e.g., API keys) if needed
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("Loaded .env file")
        else:
            print("No .env file found or it is empty.")
    except ImportError:
        print("dotenv not installed, skipping .env file load")

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
         print_error("OPENAI_API_KEY environment variable not set.")
         # You might want to prompt for the key here instead of exiting
         # api_key = input("Please enter your OpenAI API key: ")
         # if not api_key:
         #     return
         # os.environ["OPENAI_API_KEY"] = api_key
         return # Exit if key is missing and not provided

    # Optional: Set default client if needed globally
    # import agents
    # from openai import AsyncOpenAI
    # agents.set_default_openai_client(AsyncOpenAI()) # Already done by default if key is set

    await run_aider(project_goal, max_turns=20)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAider stopped by user.")
    except Exception as e:
        print_error(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
