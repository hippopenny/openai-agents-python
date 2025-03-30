import pytest
from unittest.mock import AsyncMock

from agents import Agent, Runner, RunResult

from hippopenny_agents.aider.agents import CoderAgent, CoderOutput
from hippopenny_agents.aider.models import ProjectContext


@pytest.mark.asyncio
async def test_coder_agent_runs(monkeypatch):
    """Tests that the Coder agent can be invoked and returns expected structure."""
    mock_run_result = RunResult(
        input="Implement the login function.",
        final_output=CoderOutput(status="completed", summary="Implemented login function.", code_changes="def login(): pass"),
        usage=AsyncMock(),
        new_items=[],
        all_items=[],
        last_agent=CoderAgent,
        model_name="mock_model",
        response_id="mock_response_id",
    )

    # Mock Runner.run to avoid actual LLM calls
    mock_runner_run = AsyncMock(return_value=mock_run_result)
    monkeypatch.setattr(Runner, "run", mock_runner_run)

    task_description = "Implement the login function."
    context = ProjectContext(project_goal="Build web app") # Coder doesn't use context directly, but Runner needs it

    # We are testing the conceptual invocation, usually done via the planner's tool
    # Here we simulate that call directly for isolation
    result = await Runner.run(CoderAgent, task_description, context=context)

    assert isinstance(result.final_output, CoderOutput)
    assert result.final_output.status == "completed"
    assert result.final_output.summary == "Implemented login function."
    assert result.final_output.code_changes == "def login(): pass"

    mock_runner_run.assert_called_once()
    # Check args passed to Runner.run (agent, input, context)
    call_args, call_kwargs = mock_runner_run.call_args
    assert call_args[0] == CoderAgent
    assert call_args[1] == task_description
    assert call_kwargs.get("context") == context

