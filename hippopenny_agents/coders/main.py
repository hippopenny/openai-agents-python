import asyncio
import os

from agents import Agent, Runner, trace
from agents.run_context import RunContextWrapper

from .context import CoderContext, Task
from .planner import create_planner_agent
from .aider import create_aider_agent

# --- Optional: Rich printing for progress ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text
    console = Console()
    USE_RICH = True
except ImportError:
    console = None
    USE_RICH = False
    # Define dummy classes/functions if rich is not installed
    class Spinner:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass
    def Panel(*args, **kwargs): return args[0]
# --- End Optional Rich Printing ---


async def run_planner(
    planner: Agent[CoderContext],
    context: CoderContext,
    input_prompt: str,
) -> None:
    """Runs the planner agent to generate or update tasks."""
    print_status(f"Running Planner: {input_prompt}")
    # Use RunContextWrapper manually for type safety if needed, or pass context directly
    await Runner.run(planner, input_prompt, context=context)
    print_status("Planner finished.")


async def run_aider(
    aider: Agent[CoderContext],
    context: CoderContext,
    task: Task,
) -> tuple[str | None, str | None]:
    """Runs the aider agent for a specific task."""
    status_text = f"Running Aider for Task {task.id}: {task.description}"
    spinner = Spinner("dots", text=Text(status_text, style="yellow"))

    if USE_RICH and console:
        console.print(spinner.__enter__()) # Manual context management for print
    else:
        print(status_text + "...")

    try:
        # Pass only the task description as input to aider
        result = await Runner.run(aider, task.description, context=context)
        final_output = str(result.final_output) # Assuming aider returns text

        if final_output.strip().lower().startswith("error:"):
            error_message = final_output.strip()
            task_result = None
            print_status(f"Aider Task {task.id} Failed: {error_message}", style="red")
        else:
            error_message = None
            task_result = final_output
            print_status(f"Aider Task {task.id} Completed.", style="green")

        return task_result, error_message

    except Exception as e:
        error_message = f"Aider execution error: {e}"
        task_result = None
        print_status(f"Aider Task {task.id} Failed unexpectedly: {error_message}", style="red")
        return task_result, error_message
    finally:
        if USE_RICH and console:
             spinner.__exit__(None, None, None)


def print_status(message: str, style: str = "blue"):
    """Prints status messages, using rich if available."""
    if USE_RICH and console:
        console.print(f"[{style}]{message}[/{style}]")
    else:
        print(message)

def print_tasks(tasks: list[Task]):
    """Prints the current task list."""
    if not tasks:
        print_status("No tasks generated yet.", style="yellow")
        return

    lines = ["Current Tasks:"]
    for task in tasks:
        status_color = "grey"
        if task.status == "completed":
            status_color = "green"
        elif task.status == "failed":
            status_color = "red"
        elif task.status == "in_progress":
            status_color = "yellow"

        line = f"  Task {task.id}: {task.description} ([{status_color}]{task.status}[/{status_color}])"
        lines.append(line)
        if task.result:
            lines.append(f"    Result: {task.result}")
        if task.error_message:
             lines.append(f"    Error: {task.error_message}")

    if USE_RICH and console:
        console.print(Panel("\n".join(lines), title="Task Plan", border_style="dim"))
    else:
        print("\n".join(lines))


async def main():
    """Main orchestration loop."""
    initial_request = input("Enter your coding request: ")

    # Ensure proxy URL is set if needed
    if not os.environ.get("AIDER_PROXY_BASE_URL"):
        print_status(
            "Warning: AIDER_PROXY_BASE_URL environment variable not set. Using default http://localhost:8080/v1",
            style="yellow"
        )

    # Initialize context and agents
    coder_context = CoderContext(initial_request=initial_request)
    planner_agent = create_planner_agent()
    aider_agent = create_aider_agent()

    trace_id = None # Allow trace to generate one

    # Use trace context manager for the entire operation
    with trace("Coder Workflow", trace_id=trace_id) as root_span:
        trace_id = root_span.trace_id
        print_status(f"Starting workflow. Trace ID: {trace_id}", style="cyan")
        print_status(f"View trace (if backend configured): https://platform.openai.com/traces/{trace_id}", style="cyan")


        # 1. Initial Planning
        await run_planner(planner_agent, coder_context, f"Generate plan for: {initial_request}")
        print_tasks(coder_context.tasks)

        if not coder_context.tasks:
            print_status("Planner did not generate any tasks. Exiting.", style="red")
            return

        # 2. Execute Tasks Sequentially
        while True:
            next_task = None
            for task in coder_context.tasks:
                if task.status == "pending":
                    next_task = task
                    break

            if not next_task:
                print_status("All tasks processed.", style="green")
                break # No more pending tasks

            # Mark task as in progress
            next_task.status = "in_progress"
            coder_context.current_task_id = next_task.id
            print_tasks(coder_context.tasks) # Show updated status

            # Run Aider
            task_result, error_message = await run_aider(aider_agent, coder_context, next_task)

            # Update task based on aider result
            if error_message:
                next_task.status = "failed"
                next_task.error_message = error_message
                # --- Optional: Add replanning logic here ---
                # print_status("Task failed. Running planner again to potentially replan...", style="yellow")
                # await run_planner(planner_agent, coder_context, f"Task {next_task.id} failed with error: {error_message}. Review the plan.")
                # For now, we just mark as failed and continue/stop
                print_status(f"Task {next_task.id} failed. Stopping execution.", style="red")
                break # Stop on first failure for simplicity
                # --- End Optional Replanning ---
            else:
                next_task.status = "completed"
                next_task.result = task_result

            coder_context.current_task_id = None
            print_tasks(coder_context.tasks) # Show final status for the task

        # 3. Final Summary (Optional)
        print_status("\nWorkflow finished.", style="cyan")
        print_tasks(coder_context.tasks)

        # You could run the planner one last time to generate a summary report
        # await run_planner(planner_agent, coder_context, "Summarize the results of the completed tasks.")
        # print(coder_context.final_summary) # Assuming context has a final_summary field


if __name__ == "__main__":
    # Add asyncio boilerplate
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
