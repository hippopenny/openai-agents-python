from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Re-introduce agents SDK imports
from agents import Agent


logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Agent Definitions (Based on high-level example)
# ----------------------------------------------------------

# Define specialized agents first
# Note: Instructions are simplified placeholders from the example.
# For real use, more detailed instructions (like those in SystemPromptBuilder)
# would likely be needed.

browser_agent = Agent(
    name="browser_agent",
    instructions="You execute browser navigation and interaction steps based on the input. Input will specify the exact action (e.g., go_to_url, click_element). Respond only with the result of the action.",
    handoff_description="An agent specialized in executing browser actions like navigating, clicking, typing, and extracting content.",
    # output_type=ActionResult, # Define output type if browser actions return a specific Pydantic model
)
logger.info("Defined browser_agent")

api_agent = Agent(
    name="api_agent",
    instructions="You execute API calls based on the input (e.g., endpoint, method, payload) and process the responses. Respond only with the API response data or error.",
    handoff_description="An agent specialized in making API calls and returning the results.",
    # output_type=dict, # Example output type
)
logger.info("Defined api_agent")


# Define the main orchestrator agent
# It uses the specialized agents as tools.
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are an orchestration agent. Your goal is to accomplish the user's overall task (e.g., 'Book a flight and reserve a hotel'). "
        "Analyze the task, the current plan (if provided), and the current state. "
        "Decide the next logical step and delegate it to the appropriate tool: 'browser_tool' for web interactions or 'api_tool' for API calls. "
        "Provide the necessary input for the chosen tool based on the task and current context. "
        "You never perform browser or API actions directly; you MUST use the tools."
        "If a plan is provided, use it to guide your tool selection and inputs."
        "After a tool is called, evaluate its result and decide the next step."
    ),
    handoff_description="Orchestrates complex tasks across browser and API interactions.",
    tools=[
        browser_agent.as_tool(
            tool_name="browser_tool",
            tool_description="Use this tool for any tasks requiring web browser interaction, such as navigating to URLs, clicking buttons, filling forms, scrolling, or extracting web content. Provide the specific action needed as input (e.g., 'Navigate to google.com', 'Click the login button').",
        ),
        api_agent.as_tool(
            tool_name="api_tool",
            tool_description="Use this tool for tasks involving direct API interactions, such as fetching data from an API endpoint or sending data to an API. Provide the API endpoint, method, and payload as input.",
        ),
        # Potentially add the 'update_agent_state' tool back if needed for structured thinking,
        # although the example implies the orchestrator directly calls browser/api tools.
        # update_agent_state_tool, # Requires defining this tool using @function_tool
    ],
    # model_settings=ModelSettings(tool_choice="required"), # Optional: Force tool use
)
logger.info("Defined orchestrator_agent with browser_tool and api_tool")


