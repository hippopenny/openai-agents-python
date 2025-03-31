import pytest
from unittest.mock import AsyncMock, MagicMock, call

from hippopenny_agents.agent_browser2.controller import ActionController
from hippopenny_agents.agent_browser2.context import BaseContext

@pytest.fixture
def mock_context():
    """Fixture for a mocked BaseContext."""
    mock = AsyncMock(spec=BaseContext)
    mock.execute_action = AsyncMock() # Ensure execute_action is async
    return mock

@pytest.fixture
def action_controller(mock_context):
    """Fixture for ActionController with a mocked context."""
    return ActionController(context=mock_context)

@pytest.mark.asyncio
async def test_execute_actions_calls_context(action_controller: ActionController, mock_context: AsyncMock):
    """Test that execute_actions calls context.execute_action for each action."""
    actions = [
        {"action1": {"param": "value1"}},
        {"action2": {"param": "value2"}},
    ]
    mock_context.execute_action.side_effect = [
        {"result": "result1", "error": None},
        {"result": "result2", "error": None},
    ]

    results = await action_controller.execute_actions(actions)

    assert mock_context.execute_action.call_count == 2
    mock_context.execute_action.assert_has_calls([
        call(actions[0]),
        call(actions[1]),
    ])
    assert len(results) == 2
    assert results[0]["result"] == "result1"
    assert results[1]["result"] == "result2"

@pytest.mark.asyncio
async def test_execute_actions_stops_on_error_result(action_controller: ActionController, mock_context: AsyncMock):
    """Test that execute_actions stops if an action result contains an error."""
    actions = [
        {"action1": {"param": "value1"}},
        {"action2": {"param": "value2"}}, # This action should not be called
    ]
    mock_context.execute_action.side_effect = [
        {"result": "result1", "error": "Something went wrong"},
        # No second result needed as it should stop
    ]

    results = await action_controller.execute_actions(actions)

    assert mock_context.execute_action.call_count == 1
    mock_context.execute_action.assert_called_once_with(actions[0])
    assert len(results) == 1
    assert results[0]["error"] == "Something went wrong"

@pytest.mark.asyncio
async def test_execute_actions_stops_on_exception(action_controller: ActionController, mock_context: AsyncMock):
    """Test that execute_actions stops and returns error if context.execute_action raises."""
    actions = [
        {"action1": {"param": "value1"}},
        {"action2": {"param": "value2"}}, # This action should not be called
    ]
    mock_context.execute_action.side_effect = [
        RuntimeError("Context execution failed"),
        # No second call needed
    ]

    results = await action_controller.execute_actions(actions)

    assert mock_context.execute_action.call_count == 1
    mock_context.execute_action.assert_called_once_with(actions[0])
    assert len(results) == 1
    assert results[0]["action_executed"] == actions[0]
    assert results[0]["result"] == "Failed to execute action"
    assert "Context execution failed" in results[0]["error"]

@pytest.mark.asyncio
async def test_execute_actions_empty_list(action_controller: ActionController, mock_context: AsyncMock):
    """Test execute_actions with an empty list of actions."""
    actions = []
    results = await action_controller.execute_actions(actions)

    assert mock_context.execute_action.call_count == 0
    assert len(results) == 0

