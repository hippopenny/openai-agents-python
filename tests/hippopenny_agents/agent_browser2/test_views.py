import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from hippopenny_agents.agent_browser2.views import (
    ActionResult,
    AgentStateUpdate,
    PlannerOutput,
    AgentHistory,
    AgentHistoryList,
)

# --- Basic Model Tests ---

def test_action_result_instantiation():
    res = ActionResult(is_done=True, extracted_content="Done!", error=None)
    assert res.is_done is True
    assert res.extracted_content == "Done!"
    assert res.error is None

def test_agent_state_update_instantiation():
    update = AgentStateUpdate(
        page_summary="Found contact info.",
        evaluation_previous_goal="Success",
        memory="Contact info extracted.",
        next_goal="Complete task."
    )
    assert update.page_summary == "Found contact info."
    assert update.evaluation_previous_goal == "Success"

def test_planner_output_instantiation():
    plan = PlannerOutput(
        state_analysis="Analyzed state.",
        progress_evaluation="50%",
        challenges=["Login required"],
        next_steps=["Attempt login"],
        reasoning="Need to login first."
    )
    assert plan.state_analysis == "Analyzed state."
    assert plan.challenges == ["Login required"]

# --- AgentHistory Tests ---

@pytest.fixture
def sample_agent_history_item():
    state = {"url": "http://example.com", "title": "Example"}
    plan = PlannerOutput(
        state_analysis="Initial", progress_evaluation="0%", challenges=[], next_steps=["Step 1"], reasoning="Start"
    )
    results = [{"result": "Step 1 done", "error": None}]
    output_items = [{"type": "message", "role": "assistant", "content": "Okay, doing Step 1."}]
    return AgentHistory(
        step_number=1,
        state_before=state,
        plan=plan,
        orchestrator_input="Do Step 1",
        orchestrator_output_items=output_items,
        action_results=results
    )

def test_agent_history_instantiation(sample_agent_history_item: AgentHistory):
    assert sample_agent_history_item.step_number == 1
    assert sample_agent_history_item.state_before["url"] == "http://example.com"
    assert isinstance(sample_agent_history_item.plan, PlannerOutput)
    assert sample_agent_history_item.action_results[0]["result"] == "Step 1 done"

def test_agent_history_model_dump(sample_agent_history_item: AgentHistory):
    dumped = sample_agent_history_item.model_dump()
    assert dumped["step_number"] == 1
    assert dumped["state_before"]["url"] == "http://example.com"
    assert isinstance(dumped["plan"], dict) # Should be dumped dict
    assert dumped["plan"]["state_analysis"] == "Initial"
    assert dumped["action_results"][0]["result"] == "Step 1 done"
    assert dumped["orchestrator_output_items"][0]["content"] == "Okay, doing Step 1."

# --- AgentHistoryList Tests ---

@pytest.fixture
def sample_agent_history_list(sample_agent_history_item: AgentHistory):
    error_item = AgentHistory(
        step_number=2,
        action_results=[{"result": None, "error": "Failed"}]
    )
    done_item = AgentHistory(
        step_number=3,
        action_results=[{"result": "Final Answer", "error": None, "is_done": True}]
    )
    return AgentHistoryList(history=[sample_agent_history_item, error_item, done_item])

def test_agent_history_list_instantiation(sample_agent_history_list: AgentHistoryList):
    assert len(sample_agent_history_list.history) == 3

@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.mkdir")
def test_agent_history_list_save_to_file(mock_mkdir, mock_file, sample_agent_history_list: AgentHistoryList):
    filepath = "test_history.json"
    sample_agent_history_list.save_to_file(filepath)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_file.assert_called_once_with(filepath, 'w', encoding='utf-8')
    handle = mock_file()

    # Check if json.dump was called (content check is complex due to mocking)
    assert handle.write.call_count > 0
    # Basic check if it looks like JSON
    args, _ = handle.write.call_args
    assert args[0].strip().startswith('{')

@patch("builtins.open", new_callable=mock_open, read_data='{"history": []}')
def test_agent_history_list_load_from_file(mock_file):
    filepath = "test_history.json"
    # Note: Current implementation warns and returns empty list. Test this behavior.
    with pytest.warns(UserWarning, match="needs reimplementation"):
        loaded_history = AgentHistoryList.load_from_file(filepath)

    mock_file.assert_called_once_with(filepath, 'r', encoding='utf-8')
    assert isinstance(loaded_history, AgentHistoryList)
    assert loaded_history.history == []

def test_agent_history_list_errors(sample_agent_history_list: AgentHistoryList):
    errors = sample_agent_history_list.errors()
    assert errors == ["Failed"]

def test_agent_history_list_has_errors(sample_agent_history_list: AgentHistoryList):
    assert sample_agent_history_list.has_errors() is True
    empty_list = AgentHistoryList(history=[sample_agent_history_list.history[0]]) # Only first item
    assert empty_list.has_errors() is False

def test_agent_history_list_is_done(sample_agent_history_list: AgentHistoryList):
    assert sample_agent_history_list.is_done() is True
    not_done_list = AgentHistoryList(history=sample_agent_history_list.history[:2]) # First two items
    assert not_done_list.is_done() is False

def test_agent_history_list_final_result(sample_agent_history_list: AgentHistoryList):
    # Test case where last action has is_done=True and a result
    assert sample_agent_history_list.final_result() == "Final Answer"

    # Test case where last action is not done, but orchestrator output exists
    orchestrator_output = AgentHistory(
        step_number=4,
        orchestrator_output_items=[{"type": "message", "role": "assistant", "content": "Task finished."}]
    )
    list_with_text_output = AgentHistoryList(history=[orchestrator_output])
    assert list_with_text_output.final_result() == "Task finished."

    # Test case where nothing indicates completion
    not_done_list = AgentHistoryList(history=sample_agent_history_list.history[:2])
    assert not_done_list.final_result() is None


def test_agent_history_list_urls(sample_agent_history_list: AgentHistoryList):
    urls = sample_agent_history_list.urls()
    # Only the first item has state_before with url
    assert urls == ["http://example.com"]

def test_agent_history_list_screenshots(sample_agent_history_list: AgentHistoryList):
    # Placeholder context doesn't add screenshots, so this should be empty
    screenshots = sample_agent_history_list.screenshots()
    assert screenshots == []

