from __future__ import annotations

import logging
from typing import Any, Dict, List

from .context import BaseContext

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 4. Action Controller: Executes actions on the given context.
# (Based on high-level example)
# ----------------------------------------------------------

class ActionController:
    """Executes a list of actions using the provided context."""
    def __init__(self, context: BaseContext) -> None:
        """
        Initializes the ActionController.

        Args:
            context: An instance of BaseContext (or its subclass) to execute actions on.
        """
        self.context = context
        logger.info(f"ActionController initialized with context type: {type(context).__name__}")

    async def execute_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes a sequence of actions.

        Args:
            actions: A list of dictionaries, where each dictionary represents an action.
                     The structure of the action dictionary depends on the context implementation.

        Returns:
            A list of result dictionaries, one for each executed action.
        """
        results = []
        logger.info(f"Executing {len(actions)} actions...")
        for i, action in enumerate(actions):
            action_name = next(iter(action), f"action_{i+1}")
            logger.debug(f"Executing action {i+1}/{len(actions)}: {action_name}")
            try:
                result = await self.context.execute_action(action)
                logger.debug(f"Action {action_name} result: {result.get('result', 'No result field')}")
                results.append(result)
                # Optional: Check result for errors or conditions to stop execution
                if isinstance(result, dict) and result.get('error'):
                    logger.warning(f"Action {action_name} resulted in error: {result['error']}. Stopping sequence.")
                    break # Stop executing further actions if one fails
            except Exception as e:
                logger.error(f"Error executing action {action_name}: {e}", exc_info=True)
                error_result = {
                    "action_executed": action,
                    "result": "Failed to execute action",
                    "error": str(e)
                }
                results.append(error_result)
                break # Stop executing further actions on exception
        logger.info(f"Finished executing actions. Results count: {len(results)}")
        return results
