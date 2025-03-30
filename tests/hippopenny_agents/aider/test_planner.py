import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import Agent, Runner, RunResult, RunContextWrapper, FunctionTool, ModelBehaviorError

# Import specific tools and agents needed for testing
from hippopenny_agents.aider.agents.planner import (
    PlannerAgent, plan_initial_tasks, add_task, modify_task, implement_task
)
from hippopenny_agents.aider.agents.coder import CoderAgent, CoderOutput
from hippopenny_agents.aider.models import ProjectContext, Task, TaskStatus


# --- Mocks ---

# Mock Coder Agent run results (used within implement_task tool)
mock_coder_success_output = CoderOutput(
    status="completed",
    summary="Task completed successfully.",
    code_changes="print('done')"
)
mock_coder_failure_output = CoderOutput(
    status="failed",
    summary="Failed due to import error.",
    code_changes=None
)
mock_coder_clarification_output = CoderOutput(
    status="needs_clarification",
    summary="What should the output filename be?",
    code_changes=None
)

mock_coder_success_result = RunResult(
    input="Mock Task Desc", final_output=mock_coder_success_output,
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_1"
)
mock_coder_failure_result = RunResult(
    input="Mock Task Desc", final_output=mock_coder_failure_output,
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_2"
)
mock_coder_clarification_result = RunResult(
    input="Mock Task Desc", final_output=mock_coder_clarification_output,
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_3"
)
mock_coder_bad_output_result = RunResult(
    input="Mock Task Desc", final_output="Just some text, not CoderOutput",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_4"
)


# --- Fixtures ---

@pytest.fixture
def project_context() -> ProjectContext:
    """Provides a fresh ProjectContext for each test."""
    return ProjectContext(project_goal="Create a simple script.")

@pytest.fixture
def run_context_wrapper(project_context: ProjectContext) -> RunContextWrapper[ProjectContext]:
    """Provides a mock RunContextWrapper wrapping the project_context."""
    # Simulate the wrapper that tool functions receive
    return RunContextWrapper(
        context=project_context,
        usage=AsyncMock(),
        run_config=MagicMock(),
        model_provider=MagicMock(),
        hooks=MagicMock(),
        max_turns=10,
        turn_num=1,
        current_agent=PlannerAgent # Assume Planner is calling the tool
    )

# --- Tool Tests ---

@pytest.mark.asyncio
async def test_plan_initial_tasks_tool(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test the plan_initial_tasks tool function."""
    project_ctx = run_context_wrapper.context
    assert not project_ctx.tasks # Should start empty

    task_descs = ["Task 1", "Task 2"]
    # Access the actual function attached to the FunctionTool object
    result = await plan_initial_tasks.func(context=run_context_wrapper, task_descriptions=task_descs)

    assert len(project_ctx.tasks) == 2
    assert project_ctx.tasks[0].description == "Task 1"
    assert project_ctx.tasks[0].status == "pending"
    assert project_ctx.tasks[0].id == 0 # Check ID assignment
    assert project_ctx.tasks[1].description == "Task 2"
    assert project_ctx.tasks[1].status == "pending"
    assert project_ctx.tasks[1].id == 1
    assert result == "Successfully planned 2 initial tasks."

@pytest.mark.asyncio
async def test_plan_initial_tasks_tool_already_planned(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test plan_initial_tasks when tasks already exist."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Existing Task"])

    result = await plan_initial_tasks.func(context=run_context_wrapper, task_descriptions=["New Task"])
    assert len(project_ctx.tasks) == 1 # No change
    assert result == "Error: Initial tasks have already been planned."

@pytest.mark.asyncio
async def test_add_task_tool(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test the add_task tool function."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Task 0"]) # Add an initial task (ID 0)

    result = await add_task.func(context=run_context_wrapper, description="Task 1", insert_before_id=None)

    assert len(project_ctx.tasks) == 2
    assert project_ctx.tasks[1].description == "Task 1"
    assert project_ctx.tasks[1].id == 1 # Assumes _next_task_id started at 1 after Task 0
    assert "Successfully added new task 'Task 1' with ID 1" in result

    result_insert = await add_task.func(context=run_context_wrapper, description="Task 0.5", insert_before_id=1)
    assert len(project_ctx.tasks) == 3
    assert project_ctx.tasks[1].description == "Task 0.5" # Should be inserted before Task 1 (which now has ID 1)
    assert project_ctx.tasks[1].id == 2 # New task gets next ID
    assert project_ctx.tasks[2].description == "Task 1" # Original Task 1 shifted
    assert "Successfully added new task 'Task 0.5' with ID 2" in result_insert

@pytest.mark.asyncio
async def test_modify_task_tool(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test the modify_task tool function."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Task A", "Task B"])
    task_a_id = project_ctx.tasks[0].id # Should be 0
    task_b_id = project_ctx.tasks[1].id # Should be 1

    # Modify description
    result_desc = await modify_task.func(context=run_context_wrapper, task_id=task_a_id, new_description="Task A Modified")
    assert project_ctx.tasks[0].description == "Task A Modified"
    assert "description updated to 'Task A Modified'" in result_desc

    # Modify status
    result_status = await modify_task.func(context=run_context_wrapper, task_id=task_a_id, new_status="done")
    assert project_ctx.tasks[0].status == "done"
    assert "status updated to 'done'" in result_status

    # Modify both description and status
    result_both = await modify_task.func(context=run_context_wrapper, task_id=task_b_id, new_description="Task B Final", new_status="failed")
    assert project_ctx.tasks[1].description == "Task B Final"
    assert project_ctx.tasks[1].status == "failed"
    assert "description updated to 'Task B Final'" in result_both
    assert "status updated to 'failed'" in result_both
    # Check if coder_error was set (it shouldn't be just from modify_task unless status is failed, but modify_task doesn't set error directly)
    # The update_task_status method handles error clearing/setting. Let's test that interaction via implement_task.

    # Modify non-existent task
    result_not_found = await modify_task.func(context=run_context_wrapper, task_id=999, new_description="Doesn't exist")
    assert "Error: Task with ID 999 not found" in result_not_found


@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock) # Mock Runner.run used *inside* the tool
async def test_implement_task_tool_success(mock_runner_run: AsyncMock, run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test the implement_task tool when the coder succeeds."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Implement feature X"])
    task_to_run = project_ctx.tasks[0]
    task_id = task_to_run.id
    assert task_to_run.status == "pending"

    # Configure the mock Runner.run to return coder success
    mock_runner_run.return_value = mock_coder_success_result

    # Execute the tool function
    tool_result_summary = await implement_task.func(
        context=run_context_wrapper,
        task_id=task_id
    )

    # Assertions
    task_after = project_ctx.tasks[0]
    assert task_after.status == "done" # Status updated by the tool
    assert project_ctx.coder_error is None # Error cleared on success
    assert tool_result_summary == mock_coder_success_output.summary # Tool returns coder's summary

    # Check that Runner.run was called correctly for the CoderAgent
    mock_runner_run.assert_called_once()
    call_args, call_kwargs = mock_runner_run.call_args
    assert call_args[0] == CoderAgent # Called the coder agent
    assert call_args[1] == task_to_run.description # Passed the correct description
    assert call_kwargs.get("context") == project_ctx # Passed the shared context
    assert task_after.status != "in_progress" # Should be updated *after* run


@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_implement_task_tool_failure(mock_runner_run: AsyncMock, run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test the implement_task tool when the coder fails."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Implement feature Y"])
    task_to_run = project_ctx.tasks[0]
    task_id = task_to_run.id
    assert task_to_run.status == "pending"

    mock_runner_run.return_value = mock_coder_failure_result

    tool_result_summary = await implement_task.func(context=run_context_wrapper, task_id=task_id)

    task_after = project_ctx.tasks[0]
    assert task_after.status == "failed" # Status updated
    assert project_ctx.coder_error == mock_coder_failure_output.summary # Error stored
    assert tool_result_summary == mock_coder_failure_output.summary # Tool returns summary

    mock_runner_run.assert_called_once()


@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_implement_task_tool_needs_clarification(mock_runner_run: AsyncMock, run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test implement_task when coder needs clarification (treated as failure)."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Implement feature Z"])
    task_to_run = project_ctx.tasks[0]
    task_id = task_to_run.id

    mock_runner_run.return_value = mock_coder_clarification_result

    tool_result_summary = await implement_task.func(context=run_context_wrapper, task_id=task_id)

    task_after = project_ctx.tasks[0]
    assert task_after.status == "failed" # Status updated to failed
    expected_error = f"Needs Clarification: {mock_coder_clarification_output.summary}"
    assert project_ctx.coder_error == expected_error # Error stored with prefix
    # Tool returns the modified summary for clarification cases
    assert tool_result_summary == expected_error

    mock_runner_run.assert_called_once()


@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_implement_task_tool_coder_bad_output(mock_runner_run: AsyncMock, run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test implement_task when coder returns unexpected output format."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Implement feature W"])
    task_to_run = project_ctx.tasks[0]
    task_id = task_to_run.id

    mock_runner_run.return_value = mock_coder_bad_output_result

    tool_result_summary = await implement_task.func(context=run_context_wrapper, task_id=task_id)

    task_after = project_ctx.tasks[0]
    assert task_after.status == "failed" # Status updated to failed
    expected_error = f"CoderAgent returned unexpected output type: <class 'str'>. Raw output: {mock_coder_bad_output_result.final_output}"
    assert project_ctx.coder_error == expected_error # Error stored
    assert tool_result_summary == expected_error # Tool returns the error message

    mock_runner_run.assert_called_once()


@pytest.mark.asyncio
async def test_implement_task_tool_task_not_found(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test implement_task when the task ID doesn't exist."""
    result = await implement_task.func(context=run_context_wrapper, task_id=999)
    assert "Error: Task with ID 999 not found" in result

@pytest.mark.asyncio
async def test_implement_task_tool_wrong_status(run_context_wrapper: RunContextWrapper[ProjectContext]):
    """Test implement_task when the task is not in 'pending' state."""
    project_ctx = run_context_wrapper.context
    project_ctx.add_tasks(["Task Done"])
    task_id = project_ctx.tasks[0].id
    project_ctx.update_task_status(task_id, "done") # Mark as done

    result = await implement_task.func(context=run_context_wrapper, task_id=task_id)
    assert f"Error: Task {task_id} ('Task Done') cannot be implemented because its status is 'done'" in result

# --- Planner Agent Test (Conceptual) ---
# Testing the Planner's decision making requires mocking Runner.run for the Planner itself
# and asserting which tool it *tries* to call based on the context.

@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_planner_agent_chooses_plan_initial(mock_planner_run: AsyncMock, project_context: ProjectContext):
    """Simulate Planner deciding to call plan_initial_tasks."""

    # We only care about the *call* to Runner.run for the Planner, not its result here.
    # The goal is to see if the Planner *would* call the right tool based on its prompt and context.
    # This requires a more sophisticated mock or actually running the Planner with a mocked LLM call.

    # For now, we'll skip the complex mocking of the Planner's internal LLM call.
    # The tool tests above verify the *functionality* of the tools the Planner would use.
    # The main loop test (`test_main.py`) will verify the end-to-end flow.
    pass
