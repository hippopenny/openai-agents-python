import asyncio
import json
from typing import Tuple

from agents import Runner, Agent, RunResult, ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem

from hippopenny_agents.aider.models import ProjectContext
from hippopenny_agents.aider.agents import PlannerAgent, CoderAgent, create_code_task_tool

# Attempt to import console and Spinner, fallback if rich is not installed
try:
    from rich.console import Console
    from rich.spinner import Spinner
    console = Console()
except ImportError:
    console = None
    Spinner = None # type: ignore

def print_status(message: str, style: str = "blue"):
    if console and Spinner:
        spinner = Spinner("dots", text=f" {message}")
        console.print(spinner, style=style)
    else:
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


async def run_aider(project_goal: str, max_turns: int = 10) -> Tuple[ProjectContext, str]:
    """
    Runs the Planner-Coder agent loop.

    Args:
        project_goal: The high-level goal for the project.
        max_turns: Maximum number of Planner agent turns before stopping.

    Returns:
        A tuple containing the final ProjectContext and the final message from the Planner.
    """
    print_status(f"Starting Aider for goal: '{project_goal}'")

    # 1. Initialize Context
    project_context = ProjectContext(project_goal=project_goal)

    # 2. Create Agents and Tools
    # Coder Agent is defined in its module
    # Planner Agent needs the Coder tool
    code_task_tool = create_code_task_tool(CoderAgent)
    planner_agent_with_tool = PlannerAgent.clone(tools=[code_task_tool])

    final_planner_message = "Aider run did not complete."
    planner_result: RunResult | None = None

    # 3. Main Loop
    for turn in range(max_turns):
        print_status(f"Planner Turn {turn + 1}/{max_turns}")

        # --- Pre-computation for Planner ---
        # Check if tasks need to be added based on Planner's previous output (if any)
        # This is a simplified way; a more robust approach might involve the planner
        # outputting a structured request to add tasks.
        if planner_result and not project_context.tasks:
             # Crude check: If planner mentioned tasks and context is empty, try parsing
             planner_text = ItemHelpers.text_message_outputs(planner_result.new_items)
             # Example: Look for lines starting with "- " or "1. "
             potential_tasks = [
                 line.strip('-* ').split('.', 1)[-1].strip()
                 for line in planner_text.splitlines()
                 if line.strip().startswith(('-', '*', str(len(project_context.tasks) + 1) + '.'))
             ]
             if potential_tasks:
                 print_status(f"Planner suggested tasks: {potential_tasks}. Adding to context.")
                 project_context.add_tasks(potential_tasks)


        # --- Run Planner Agent ---
        try:
            # The input to the planner is conceptually the current state,
            # but we rely on its instructions and the context object.
            # Sending a simple prompt helps trigger its logic.
            planner_input = f"Current goal: {project_goal}. Review the tasks and proceed."
            if project_context.coder_error:
                planner_input += f"\nNote: The last coder attempt failed: {project_context.coder_error}"

            planner_result = await Runner.run(
                planner_agent_with_tool,
                input=planner_input,
                context=project_context,
                max_turns=1 # Planner should decide in one turn (call tool or finish)
            )

            # --- Process Planner Output ---
            final_planner_message = ItemHelpers.text_message_outputs(planner_result.new_items)
            print_status(f"Planner output: {final_planner_message}")

            # Debug: Print tool calls/outputs if any occurred *during* the planner run
            for item in planner_result.new_items:
                 if isinstance(item, ToolCallItem):
                     print_status(f"  Planner called tool: {item.tool_name} with input {item.tool_input}")
                 elif isinstance(item, ToolCallOutputItem):
                     print_status(f"  Tool output received by planner: {item.output}")


            # --- Check for Completion ---
            # Simple check: Did the planner explicitly say it's done?
            # More robust: Check context state via project_context.are_all_tasks_done()
            # Let's rely on the context state check after the planner run.
            if project_context.are_all_tasks_done():
                print_final("Project completed successfully by Planner.")
                final_planner_message = "Project finished successfully." # Override planner msg if needed
                break

            # Advanced check: Did the planner *not* call the tool? It might be blocked or finished.
            tool_called_in_turn = any(isinstance(item, ToolCallItem) for item in planner_result.new_items)
            if not tool_called_in_turn and not project_context.are_all_tasks_done():
                 # Planner decided to stop without finishing. Maybe blocked?
                 print_error(f"Planner stopped without completing all tasks. Final message: {final_planner_message}")
                 break


        except Exception as e:
            print_error(f"Error during Planner turn {turn + 1}: {e}")
            final_planner_message = f"Error occurred: {e}"
            break

        # Optional: Add a small delay between turns if needed
        # await asyncio.sleep(1)

    else:
        # Loop finished without break (max_turns reached)
        print_error(f"Max turns ({max_turns}) reached. Stopping.")
        final_planner_message = f"Max turns ({max_turns}) reached. Project may be incomplete."

    print("\n--- Aider Run Finished ---")
    print("Final Task Status:")
    for task in project_context.tasks:
        print(f"  - Task {task.id}: {task.description} [{task.status}]")
    print(f"Final Planner Message: {final_planner_message}")
    print("-------------------------")

    return project_context, final_planner_message


async def main():
    # Example Usage
    project_goal = input("Enter the project goal: ")
    if not project_goal:
        project_goal = "Create a python script that prints 'Hello, Aider!'"

    await run_aider(project_goal, max_turns=10)

if __name__ == "__main__":
    # Setup environment (e.g., API keys) if needed
    # from dotenv import load_dotenv
    # load_dotenv()
    # import agents
    # agents.set_default_openai_key(os.environ["OPENAI_API_KEY"])

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAider stopped by user.")

