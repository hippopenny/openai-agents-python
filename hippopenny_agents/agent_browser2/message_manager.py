from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 2. Message Manager (Simplified from example)
# ----------------------------------------------------------

class MessageManager:
    """
    Tracks messages (state, previous results, plans) and prepares input
    for planner and agent calls.
    (Simplified version based on high-level example)
    """
    def __init__(self) -> None:
        self.history: List[str] = []
        logger.debug("MessageManager initialized.")

    def add_message(self, message: str) -> None:
        """Adds a message string to the history."""
        logger.debug(f"Adding message to history: {message[:100]}...")
        self.history.append(message)

    def get_history(self) -> List[str]:
        """Returns the list of message strings."""
        return self.history

    def clear_history(self) -> None:
        """Clears the message history."""
        logger.debug("Clearing message history.")
        self.history = []
