import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from hippopenny_agents.agent_browser2.models import (
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
@patch("json.dump") # Mock json.dump directly
def test_agent_history_list_save_to_file(mock_json_dump, mock_mkdir, mock_file, sample_agent_history_list: AgentHistoryList):
    filepath = "test_history.json"
    sample_agent_history_list.save_to_file(filepath)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_file.assert_called_once_with(filepath, 'w', encoding='utf-8')
    handle = mock_file()

    # Assert json.dump was called with the correct structure and file handle
    assert mock_json_dump.call_count == 1
    call_args, call_kwargs = mock_json_dump.call_args
    # Check the data structure passed to json.dump
    dumped_data = call_args[0]
    assert isinstance(dumped_data, dict)
    assert "history" in dumped_data
    assert isinstance(dumped_data["history"], list)
    assert len(dumped_data["history"]) == len(sample_agent_history_list.history)
    # Check the file handle and indent
    assert call_args[1] == handle
    assert call_kwargs.get("indent") == 2


@patch("builtins.open", new_callable=mock_open, read_data='{"history": []}')
def test_agent_history_list_load_from_file(mock_file):
    filepath = "test_history.json"
    # Remove the warns check, as loading valid empty JSON should not warn
    loaded_history = AgentHistoryList.load_from_file(filepath)

    mock_file.assert_called_once_with(filepath, 'r', encoding='utf-8')
    assert isinstance(loaded_history, AgentHistoryList)
    assert loaded_history.history == [] # Check it returns empty list for valid empty JSON

@patch("builtins.open", new_callable=mock_open, read_data='invalid json')
@patch("hippopenny_agents.agent_browser2.models.logger") # Mock logger within views
def test_agent_history_list_load_from_file_invalid_json(mock_logger, mock_file):
    """Test loading with invalid JSON data, expecting a warning."""
    filepath = "test_history_invalid.json"
    with pytest.warns(UserWarning, match="Failed to load or parse history"):
        loaded_history = AgentHistoryList.load_from_file(filepath)

    mock_file.assert_called_once_with(filepath, 'r', encoding='utf-8')
    assert isinstance(loaded_history, AgentHistoryList)
    assert loaded_history.history == [] # Should return empty on error
    mock_logger.error.assert_called_once() # Check that error was logged

@patch("builtins.open", side_effect=FileNotFoundError("File not found"))
@patch("hippopenny_agents.agent_browser2.models.logger") # Mock logger within views
def test_agent_history_list_load_from_file_not_found(mock_logger, mock_open):
    """Test loading when file does not exist, expecting a warning."""
    filepath = "non_existent_history.json"
    with pytest.warns(UserWarning, match="History file not found"):
        loaded_history = AgentHistoryList.load_from_file(filepath)

    mock_open.assert_called_once_with(filepath, 'r', encoding='utf-8')
    assert isinstance(loaded_history, AgentHistoryList)
    assert loaded_history.history == [] # Should return empty on error
    mock_logger.error.assert_called_once() # Check that error was logged


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

