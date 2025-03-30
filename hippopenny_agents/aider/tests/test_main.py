import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agents import Runner, RunResult, FunctionTool

from hippopenny_agents.aider.main import run_aider
from hippopenny_agents.aider.models import ProjectContext, CoderOutput
from hippopenny_agents.aider.agents import PlannerAgent, CoderAgent


# --- Mocks ---

# Mock Planner results
mock_planner_result_planning = RunResult(
    input="Goal: Simple script",
    final_output="Okay, I've planned the tasks. Assigning task 0.",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=PlannerAgent,
    model_name="mock_planner", response_id="p1"
)
mock_planner_result_assign_task_1 = RunResult(
    input="Task 0 done. Next?",
    final_output="Task 0 complete. Assigning task 1.",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=PlannerAgent,
    model_name="mock_planner", response_id="p2"
)
mock_planner_result_completion = RunResult(
    input="Task 1 done. Next?",
    final_output="All tasks are completed. Project finished.",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=PlannerAgent,
    model_name="mock_planner", response_id="p3"
)

# Mock Coder result (used within the mocked tool)
mock_coder_result = RunResult(
    input="Mock Task Desc",
    final_output=CoderOutput(status="completed", summary="Task done.", code_changes="..."),
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder", response_id="c1"
)

# --- Test ---

@pytest.mark.asyncio
async def test_run_aider_full_loop(monkeypatch):
    """Test the main loop from goal to completion."""

    project_goal = "Create a simple script"
    initial_context = ProjectContext(project_goal=project_goal)

    # --- Mocking Strategy ---
    # 1. Mock `Runner.run` for the Planner: Control its output and simulate task creation/tool calls.
    # 2. Mock the `code_task` tool function directly: Simulate its effect on the context.

    planner_run_call_count = 0

    async def mock_planner_runner_run(*args, **kwargs):
        nonlocal planner_run_call_count
        planner_run_call_count += 1
        context: ProjectContext = kwargs['context']

        if planner_run_call_count == 1:
            # First call: Planner creates tasks
            context.add_tasks(["Task 0: Setup", "Task 1: Logic"])
            # Simulate planner deciding to call the tool for task 0
            # We need to manually call the *mocked* tool function here
            # because the real Runner.run is completely replaced.
            await mock_code_task_tool_func(context=MagicMock(context=context), task_description="Task 0: Setup")
            return mock_planner_result_planning
        elif planner_run_call_count == 2:
            # Second call: Planner sees task 0 done, assigns task 1
            assert context.tasks[0].status == "done"
            assert context.tasks[1].status == "pending"
            await mock_code_task_tool_func(context=MagicMock(context=context), task_description="Task 1: Logic")
            return mock_planner_result_assign_task_1
        elif planner_run_call_count == 3:
            # Third call: Planner sees task 1 done, declares completion
            assert context.tasks[0].status == "done"
            assert context.tasks[1].status == "done"
            return mock_planner_result_completion
        else:
            raise AssertionError("Planner called too many times")

    # Mock the actual tool function's behavior
    async def mock_code_task_tool_func(*args, **kwargs):
        # Simulate the tool running the coder and updating context
        context: ProjectContext = kwargs['context'].context # Access context from wrapper
        task_desc = kwargs['task_description']
        task = next(t for t in context.tasks if t.description == task_desc)
        context.update_task_status(task.id, "done") # Simulate coder success
        # Return the summary the real tool would return
        return mock_coder_result.final_output.summary

    # Apply mocks
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=mock_planner_runner_run))
    # Patch the *location where the tool function is defined and imported from* in main.py
    monkeypatch.setattr("hippopenny_agents.aider.main.create_code_task_tool",
                        lambda coder_agent: MagicMock(spec=FunctionTool, func=AsyncMock(side_effect=mock_code_task_tool_func)))


    # --- Run the aider ---
    final_context, final_message = await run_aider(project_goal, max_turns=5)

    # --- Assertions ---
    assert planner_run_call_count == 3
    assert final_context.are_all_tasks_done()
    assert len(final_context.tasks) == 2
    assert final_context.tasks[0].status == "done"
    assert final_context.tasks[1].status == "done"
    assert final_message == "All tasks are completed. Project finished."


@pytest.mark.asyncio
async def test_run_aider_max_turns(monkeypatch):
    """Test that the loop terminates due to max_turns."""

    project_goal = "Infinite loop test"
    initial_context = ProjectContext(project_goal=project_goal)

    # Mock Planner to always assign a task and never complete
    async def mock_planner_runner_run(*args, **kwargs):
        context: ProjectContext = kwargs['context']
        if not context.tasks:
            context.add_tasks(["Task 0"])
        # Always simulate calling the tool, but don't complete the task
        # (or simulate coder failure repeatedly)
        task = context.get_next_pending_task()
        if task:
             # Simulate tool call without marking done
             pass # Or call a mocked tool that marks as failed/pending
        return mock_planner_result_planning # Return some non-completion message

    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=mock_planner_runner_run))
    # Mock the tool creation to return a dummy tool
    monkeypatch.setattr("hippopenny_agents.aider.main.create_code_task_tool",
                        lambda coder_agent: MagicMock(spec=FunctionTool, func=AsyncMock(return_value="Tool ran")))


    final_context, final_message = await run_aider(project_goal, max_turns=3)

    assert "Max turns (3) reached" in final_message
    # Check that the planner was called max_turns times
    assert Runner.run.call_count == 3

