import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime

from hippopenny_agents.agent_browser2.context import BrowserContextImpl, BaseContext

@pytest.fixture
def browser_context():
    """Fixture for BrowserContextImpl."""
    return BrowserContextImpl()

@pytest.mark.asyncio
async def test_browser_context_impl_get_state(browser_context: BrowserContextImpl):
    """Test the placeholder get_state method."""
    state = await browser_context.get_state()
    assert isinstance(state, dict)
    assert "url" in state
    assert "title" in state
    assert "elements" in state
    assert "screenshot" in state
    assert "timestamp" in state
    assert state["url"] == "http://example.com/placeholder"
    assert state["title"] == "Placeholder Page"

@pytest.mark.asyncio
async def test_browser_context_impl_execute_action(browser_context: BrowserContextImpl):
    """Test the placeholder execute_action method."""
    action = {"go_to_url": {"url": "http://test.com"}}
    result = await browser_context.execute_action(action)

    assert isinstance(result, dict)
    assert "action_executed" in result
    assert "result" in result
    assert "error" in result
    assert "is_done" in result
    assert result["action_executed"] == action
    assert "Simulated success for go_to_url" in result["result"]
    assert result["error"] is None
    assert not result["is_done"]

@pytest.mark.asyncio
async def test_browser_context_impl_execute_done_action(browser_context: BrowserContextImpl):
    """Test the placeholder execute_action method with a 'done' action."""
    action = {"done": {"final_answer": "Task complete"}}
    result = await browser_context.execute_action(action)
    assert result["is_done"] is True

@pytest.mark.asyncio
async def test_browser_context_impl_close(browser_context: BrowserContextImpl):
    """Test the placeholder close method runs without error."""
    try:
        await browser_context.close()
    except Exception as e:
        pytest.fail(f"BrowserContextImpl.close() raised an exception: {e}")

def test_base_context_is_abc():
    """Verify BaseContext is an Abstract Base Class."""
    assert getattr(BaseContext, '__abstractmethods__', False)
    # Check if abstract methods exist
    assert 'get_state' in BaseContext.__abstractmethods__
    assert 'execute_action' in BaseContext.__abstractmethods__
    assert 'add_message' in BaseContext.__abstractmethods__ # Added check
    assert 'get_history' in BaseContext.__abstractmethods__ # Added check

# --- Tests for new history methods ---

def test_browser_context_impl_history_init(browser_context: BrowserContextImpl):
    """Test that history is initialized as empty list."""
    assert browser_context.get_history() == []
    assert browser_context._history == [] # Check internal attribute too

def test_browser_context_impl_add_message(browser_context: BrowserContextImpl):
    """Test adding messages to history."""
    browser_context.add_message("First message")
    assert browser_context.get_history() == ["First message"]
    assert browser_context._history == ["First message"]
    browser_context.add_message("Second message")
    assert browser_context.get_history() == ["First message", "Second message"]
    assert browser_context._history == ["First message", "Second message"]

def test_browser_context_impl_get_history(browser_context: BrowserContextImpl):
    """Test retrieving history after adding messages."""
    assert browser_context.get_history() == []
    browser_context.add_message("Test")
    assert browser_context.get_history() == ["Test"]

