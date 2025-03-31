import pytest

from hippopenny_agents.agent_browser2.message_manager import MessageManager

@pytest.fixture
def msg_manager():
    """Fixture for MessageManager."""
    return MessageManager()

def test_message_manager_init(msg_manager: MessageManager):
    """Test initialization of MessageManager."""
    assert msg_manager.history == []

def test_add_message(msg_manager: MessageManager):
    """Test adding messages."""
    msg_manager.add_message("Hello")
    assert msg_manager.history == ["Hello"]
    msg_manager.add_message("World")
    assert msg_manager.history == ["Hello", "World"]

def test_get_history(msg_manager: MessageManager):
    """Test retrieving history."""
    assert msg_manager.get_history() == []
    msg_manager.add_message("Test message")
    assert msg_manager.get_history() == ["Test message"]

def test_clear_history(msg_manager: MessageManager):
    """Test clearing history."""
    msg_manager.add_message("Message 1")
    msg_manager.add_message("Message 2")
    assert len(msg_manager.history) == 2
    msg_manager.clear_history()
    assert msg_manager.history == []
    assert msg_manager.get_history() == []

