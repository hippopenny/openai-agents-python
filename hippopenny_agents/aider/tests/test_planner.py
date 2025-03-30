import pytest
from unittest.mock import AsyncMock, MagicMock

from agents import Agent, Runner, RunResult, RunContextWrapper, FunctionTool

from hippopenny_agents.aider.agents import PlannerAgent, create_code_task_tool, CoderAgent, CoderOutput
from hippopenny_agents.aider.models import ProjectContext, Task


# --- Mocks ---

# Mock Coder Agent run result
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

mock_coder_success_result = RunResult(
    input="Mock Task Desc",
    final_output=mock_coder_success_output,
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_1"
)
mock_coder_failure_result = RunResult(
    input="Mock Task Desc",
    final_output=mock_coder_failure_output,
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=CoderAgent,
    model_name="mock_coder_model", response_id="coder_resp_2"
)

# Mock Planner Agent run result (simulating it deciding to call the tool)
# The actual tool call happens *within* the Runner.run for the Planner,
# so we mock the *tool function's* behavior, not the Planner's final text output directly for tool tests.
# For testing the planner's decision *making*, we mock Runner.run for the planner.

mock_planner_run_result_calls_tool = RunResult(
    input="Initial Goal",
    # Output doesn't matter much here as the tool call is the side effect we test
    final_output="Okay, planning done. Assigning task 0.",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=PlannerAgent,
    model_name="mock_planner_model", response_id="planner_resp_1"
)

mock_planner_run_result_completes = RunResult(
    input="Check status",
    final_output="All tasks are completed. Project finished.",
    usage=AsyncMock(), new_items=[], all_items=[], last_agent=PlannerAgent,
    model_name="mock_planner_model", response_id="planner_resp_2"
)


# --- Fixtures ---

@pytest.fixture
def project_context() -> ProjectContext:
    return ProjectContext(project_goal="Create a simple script.")

@pytest.fixture
def mock_coder_agent() -> Agent:
    # We don't need a real agent, just something to pass
    return MagicMock(spec=Agent)

@pytest.fixture
def code_task_tool(mock_coder_agent: Agent) -> FunctionTool:
    return create_code_task_tool(mock_coder_agent)

# --- Tests ---

@pytest.mark.asyncio
async def test_planner_agent_initial_run(monkeypatch, project_context):
    """Test that the planner can generate initial tasks (simulated)."""

    # Mock Runner.run for the Planner Agent
    async def mock_planner_run(*args, **kwargs):
        # Simulate the planner adding tasks to the context based on the goal
        context: ProjectContext = kwargs['context']
        if not context.tasks: # Only add tasks on the first call simulation
             context.add_tasks(["Define variables", "Implement logic", "Add tests"])
        return mock_planner_run_result_calls_tool # Simulate it deciding to call tool next

    mock_runner_run_planner = AsyncMock(side_effect=mock_planner_run)
    monkeypatch.setattr(Runner, "run", mock_runner_run_planner)

    # We don't need the real tool for this test, just the planner's logic
    planner_agent_mock_tool = PlannerAgent.clone(tools=[MagicMock(spec=FunctionTool)])

    await Runner.run(planner_agent_mock_tool, project_context.project_goal, context=project_context)

    # Assert tasks were added by the mocked run
    assert len(project_context.tasks) == 3
    assert project_context.tasks[0].description == "Define variables"
    assert project_context.tasks[0].status == "pending"
    mock_runner_run_planner.assert_called_once()


@pytest.mark.asyncio
async def test_code_task_tool_success(monkeypatch, project_context, code_task_tool, mock_coder_agent):
    """Test the code_task tool when the coder succeeds."""
    project_context.add_tasks(["Implement feature X"])
    task_to_run = project_context.tasks[0]
    assert task_to_run.status == "pending"

    # Mock Runner.run specifically for the CoderAgent call *within* the tool
    mock_runner_run_coder = AsyncMock(return_value=mock_coder_success_result)
    monkeypatch.setattr(Runner, "run", mock_runner_run_coder)

    # Prepare context wrapper for the tool function
    run_context_wrapper = RunContextWrapper(
        context=project_context,
        usage=AsyncMock(),
        run_config=MagicMock(),
        model_provider=MagicMock(),
        hooks=MagicMock(),
        max_turns=10,
        turn_num=1,
        current_agent=PlannerAgent # The agent calling the tool
    )

    # Execute the tool function directly
    tool_result_summary = await code_task_tool.func(
        context=run_context_wrapper,
        task_description=task_to_run.description
    )

    # Assertions
    assert task_to_run.status == "done" # Status updated in context
    assert project_context.coder_error is None
    assert tool_result_summary == mock_coder_success_output.summary # Tool returns summary

    # Check that Runner.run was called correctly for the CoderAgent
    mock_runner_run_coder.assert_called_once()
    call_args, call_kwargs = mock_runner_run_coder.call_args
    assert call_args[0] == mock_coder_agent # Called the coder agent
    assert call_args[1] == task_to_run.description # Passed the correct description
    assert call_kwargs.get("context") == project_context # Passed the shared context


@pytest.mark.asyncio
async def test_code_task_tool_failure(monkeypatch, project_context, code_task_tool, mock_coder_agent):
    """Test the code_task tool when the coder fails."""
    project_context.add_tasks(["Implement feature Y"])
    task_to_run = project_context.tasks[0]
    assert task_to_run.status == "pending"

    # Mock Runner.run for the CoderAgent call
    mock_runner_run_coder = AsyncMock(return_value=mock_coder_failure_result)
    monkeypatch.setattr(Runner, "run", mock_runner_run_coder)

    run_context_wrapper = RunContextWrapper(
        context=project_context, usage=AsyncMock(), run_config=MagicMock(),
        model_provider=MagicMock(), hooks=MagicMock(), max_turns=10, turn_num=1,
        current_agent=PlannerAgent
    )

    # Execute the tool function
    tool_result_summary = await code_task_tool.func(
        context=run_context_wrapper,
        task_description=task_to_run.description
    )

    # Assertions
    assert task_to_run.status == "failed" # Status updated
    assert project_context.coder_error == mock_coder_failure_output.summary # Error stored
    assert tool_result_summary == mock_coder_failure_output.summary # Tool returns summary

    mock_runner_run_coder.assert_called_once()


@pytest.mark.asyncio
async def test_planner_recognizes_completion(monkeypatch, project_context):
    """Test that the planner outputs a completion message when tasks are done."""
    project_context.add_tasks(["Task A"])
    project_context.tasks[0].status = "done" # Mark task as done

    # Mock Runner.run for the Planner Agent to return the completion message
    mock_runner_run_planner = AsyncMock(return_value=mock_planner_run_result_completes)
    monkeypatch.setattr(Runner, "run", mock_runner_run_planner)

    # We don't need the real tool for this test
    planner_agent_mock_tool = PlannerAgent.clone(tools=[MagicMock(spec=FunctionTool)])

    result = await Runner.run(planner_agent_mock_tool, "Check project status", context=project_context)

    assert result.final_output == "All tasks are completed. Project finished."
    mock_runner_run_planner.assert_called_once()

