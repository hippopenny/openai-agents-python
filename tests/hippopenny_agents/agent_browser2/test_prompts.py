import pytest
import json
from datetime import datetime

from hippopenny_agents.agent_browser2.prompts import (
    AgentMessagePrompt,
    SystemPromptBuilder,
    PlannerPromptBuilder,
)
from hippopenny_agents.agent_browser2.models import PlannerOutput

@pytest.fixture
def mock_state():
    return {
        "url": "http://example.com/test",
        "title": "Test Page",
        "elements": "[1]<button>Click Me</button>",
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def mock_results():
    return [
        {"result": "Clicked button", "error": None},
        {"result": None, "error": "Element not found"},
    ]

@pytest.fixture
def mock_plan():
    return PlannerOutput(
        state_analysis="Initial state.",
        progress_evaluation="0% complete.",
        challenges=[],
        next_steps=["Click the button"],
        reasoning="Button needs clicking."
    )

@pytest.fixture
def mock_step_info():
    return {"step": 3, "max_steps": 10}

def test_agent_message_prompt_format(mock_state, mock_results, mock_plan, mock_step_info):
    """Test formatting of AgentMessagePrompt."""
    formatter = AgentMessagePrompt(
        state=mock_state,
        previous_results=mock_results,
        plan=mock_plan.model_dump(), # Pass as dict
        step_info=mock_step_info
    )
    output = formatter.format()

    assert isinstance(output, str)
    # Check for key components
    assert "Current Plan:" in output
    assert '"state_analysis": "Initial state."' in output # Check plan content
    assert "Previous Action Results:" in output
    assert "Result 1: Clicked button" in output
    assert "Result 2: None Error: Element not found" in output
    assert "Current Page State:" in output
    assert f"Current URL: {mock_state['url']}" in output
    assert f"Page Title: {mock_state['title']}" in output
    assert f"Elements: {mock_state['elements']}" in output
    assert "Step: 3/10" in output
    assert "Current DateTime:" in output

def test_agent_message_prompt_format_no_optionals(mock_state):
    """Test formatting with only state provided."""
    formatter = AgentMessagePrompt(state=mock_state)
    output = formatter.format()

    assert "[No plan provided]" in output
    assert "[No previous results]" in output
    assert "[Step info unavailable]" in output
    assert f"Current URL: {mock_state['url']}" in output

def test_system_prompt_builder():
    """Test SystemPromptBuilder generates a system message."""
    builder = SystemPromptBuilder(action_description="Tool descriptions here")
    task = "Test the system prompt."
    content = builder.get_system_message_content(task)

    assert isinstance(content, str)
    assert task in content
    assert "IMPORTANT RULES:" in content
    # Make assertion more specific/flexible
    assert "INPUT STRUCTURE (Provided in the User Message):" in content
    # Tool description is added by SDK, not builder itself, so don't check for it here.

def test_planner_prompt_builder():
    """Test PlannerPromptBuilder generates a system message."""
    builder = PlannerPromptBuilder()
    content = builder.get_system_message_content()

    assert isinstance(content, str)
    assert "You are a planning agent" in content
    assert "Your output format MUST ALWAYS be a JSON object" in content
    assert '"state_analysis":' in content
    assert '"progress_evaluation":' in content
    assert '"challenges":' in content
    assert '"next_steps":' in content
    assert '"reasoning":' in content

