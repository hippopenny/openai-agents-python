from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from dotenv import load_dotenv
from pydantic import BaseModel # Keep pydantic for internal models if needed
from PIL import Image, ImageDraw, ImageFont

# Re-introduce agents SDK imports
from agents import Agent, Runner, trace, TResponseInputItem, ItemHelpers

# Imports from this refactored package
from .agent import orchestrator_agent # Import the main agent
from .context import BaseContext, BrowserContextImpl # Import context classes
from .controller import ActionController # Import controller
from .message_manager import MessageManager # Import message manager
from .planner import Planner # Import planner
from .prompts import AgentMessagePrompt # Import prompt formatter
from .views import AgentHistory, AgentHistoryList # Keep history views

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


# ----------------------------------------------------------
# 5. Orchestration Logic (Based on high-level example)
# ----------------------------------------------------------

@trace("Browser Agent Orchestration") # Add tracing to the main function
async def main_orchestration(task: str = "Example Task: Find contact info on example.com") -> None:
    """
    Runs the orchestration flow based on the high-level example structure,
    using the agents SDK's Agent and Runner.
    """
    logger.info(f"Starting orchestration for task: {task}")

    # --- Initialization ---
    # Create context instance (using placeholder implementation)
    browser_context = BrowserContextImpl()
    # Create message manager instance
    msg_manager = MessageManager()
    # Create Planner instance (using placeholder LLM logic)
    planner = Planner(llm=None) # Pass a real LLM here for actual planning
    # Create Action controller using the browser context
    action_controller = ActionController(browser_context)
    # History tracking (optional, based on previous structure)
    full_history = AgentHistoryList(history=[])
    # Conversation state for the Runner
    conversation_inputs: list[TResponseInputItem] = [{"role": "user", "content": task}]

    max_steps = 5 # Limit steps for this example
    previous_action_results: List[Dict] = []

    try:
        for current_step in range(1, max_steps + 1):
            logger.info(f"--- Orchestration Step {current_step}/{max_steps} ---")

            # 1. Get state from context
            current_state = await browser_context.get_state()
            msg_manager.add_message(f"State @ Step {current_step}: {json.dumps(current_state)}")

            # 2. Generate a plan (optional, based on planner logic)
            # Using placeholder planner logic here
            plan = await planner.generate_plan(msg_manager.get_history())
            plan_str = json.dumps(plan) if isinstance(plan, dict) else str(plan)
            msg_manager.add_message(f"Plan @ Step {current_step}: {plan_str}")
            logger.info(f"Plan generated: {plan_str}")

            # 3. Format input for the orchestrator agent
            # The orchestrator needs the task, plan, state, and potentially previous results
            # We format this into a single input string or structured input for the Runner
            prompt_formatter = AgentMessagePrompt(
                state=current_state,
                previous_results=previous_action_results,
                plan=plan,
                step_info={"step": current_step, "max_steps": max_steps}
            )
            orchestrator_input_content = prompt_formatter.format()

            # Add formatted content as the latest user message for the orchestrator
            # Overwrite previous user message if it exists, otherwise append
            if conversation_inputs[-1]["role"] == "user":
                 conversation_inputs[-1]["content"] = orchestrator_input_content
            else:
                 conversation_inputs.append({"role": "user", "content": orchestrator_input_content})

            logger.debug(f"Input to orchestrator agent:\n{orchestrator_input_content}")

            # 4. Run the orchestrator agent using agents.Runner
            # The orchestrator should decide which tool (browser_tool or api_tool) to call
            orchestrator_result = await Runner.run(
                orchestrator_agent,
                input=conversation_inputs,
                # context=... # Pass context if tools need it directly (less common with as_tool)
                max_turns=3 # Allow agent -> tool -> agent response cycle
            )

            logger.info("Orchestrator agent finished.")
            # Update conversation history for the next loop iteration
            conversation_inputs = orchestrator_result.to_input_list()

            # Extract results/output from the orchestrator run
            # The 'final_output' might be text, or it might be the result of the last tool call
            final_text_output = ItemHelpers.text_message_outputs(orchestrator_result.new_items)
            logger.info(f"Orchestrator final text output for step: {final_text_output}")

            # --- This part deviates from the high-level example ---
            # The example executes actions based *directly* on the planner's output.
            # Here, we assume the *orchestrator agent's* job is to call the tools,
            # so the actions have already been executed via the Runner handling tool calls.
            # We extract the results of those tool calls instead.

            tool_results_this_step = []
            for item in orchestrator_result.new_items:
                if isinstance(item, dict) and item.get("type") == "tool_output":
                    # Assuming tool output is captured in the history items
                    tool_output_data = item.get("tool_output", {})
                    # Adapt this based on how as_tool results are actually structured in the output list
                    # For now, just append the raw output
                    tool_results_this_step.append(tool_output_data)
                    logger.info(f"Tool execution result captured: {tool_output_data}")

            previous_action_results = tool_results_this_step # Feed results back for next step

            # Optional: Detailed History Tracking (using AgentHistoryList)
            # This requires mapping the Runner's output back to the AgentHistory structure
            # step_history = AgentHistory(
            #     browser_state=current_state, # Need to adapt placeholder state
            #     plan=plan,
            #     # Extract tool calls and agent state update from orchestrator_result.new_items
            #     tool_calls=[...],
            #     agent_state_update=None, # Orchestrator doesn't use update_agent_state tool here
            #     tool_results=previous_action_results # Map results to ActionResult placeholder
            # )
            # full_history.history.append(step_history)

            # Check for completion (e.g., if orchestrator indicates done, or a tool result signifies completion)
            # This logic needs to be defined based on expected agent/tool behavior.
            # Example: if final_text_output.lower().startswith("task complete"):
            #    logger.info("Orchestrator indicated task completion.")
            #    break

    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
    finally:
        logger.info("Orchestration loop finished.")
        # --- Cleanup ---
        if hasattr(browser_context, 'close'):
            await browser_context.close()

        # --- Final Output ---
        # Synthesize final output (could involve another agent call)
        final_output_message = f"Orchestration complete after {current_step} steps. Final state/results gathered."
        logger.info(final_output_message)
        print("\n--- Orchestration Summary ---")
        print(f"Task: {task}")
        print(f"Final Message: {final_output_message}")
        # print(f"Full History:\n{full_history.model_dump_json(indent=2)}") # If history tracking implemented


# ----------------------------------------------------------
# 6. Entry Point (Kept from high-level example)
# ----------------------------------------------------------

if __name__ == "__main__":
    # Example of running the orchestration
    asyncio.run(main_orchestration(task="Use browser_tool to navigate to example.com and extract the title."))

    # Note: GIF generation logic from the previous BrowserAgentRunner is removed
    # as this service file now focuses on the orchestration logic from the example.
    # GIF generation could be added back as a separate utility function if needed.

