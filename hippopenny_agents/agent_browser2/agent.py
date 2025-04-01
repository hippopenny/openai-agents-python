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
    model="gpt-4o-mini"
)
logger.info("Defined browser_agent")

api_agent = Agent(
    name="api_agent",
    instructions="You execute API calls based on the input (e.g., endpoint, method, payload) and process the responses. Respond only with the API response data or error.",
    handoff_description="An agent specialized in making API calls and returning the results.",
    # output_type=dict, # Example output type
    model="gpt-4o-mini"

)
logger.info("Defined api_agent")

game_designer_agent = Agent(
    name="game_designer_agent",
    instructions="You are a game designer. You create a design document with clear breakdown of tasks in a designed documents for software engineers to follow and implement. You also provide a list of tasks to be done in the design document.",
    handoff_description="An agent specialized in creating design documents for game development.",
    # output_type=dict, # Example output type
    model="gpt-4o-mini"
)
logger.info("Defined game_designer_agent")

software_engineer_agent = Agent(
    name="software_engineer_agent",
    instructions="You are a software engineer. You implement the tasks provided. Before writing code, you must briefly say what tasks are you implemnting. You don't pick more than 2 tasks at at time. You write COMPLETE code in html and javascript in one single file.",
    handoff_description="software engineer specialized in implementing tasks in a design document.",
    # output_type=dict, # Example output type
    model="gpt-4o-mini"
)


# Define the main orchestrator agent
# It uses the specialized agents as tools.
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are an orchestration agent. Your goal is to accomplish the user's overall task by any mean possible."
        "You will be given a plan, you decide the next logical step and delegate tasks to relevant agents or tools."
        "Provide the necessary input for the chosen agent based on the task and current context. "
        "It's your responsibility to ensure the task is completed successfully."
        "You can negotiate the plan with the planner, and your agent teams, but make sure project is completed within a few turns"
    ),
    handoff_description="Orchestrates complex tasks",
    tools=[
        # browser_agent.as_tool(
        #     tool_name="browser_tool",
        #     tool_description="Navigate websites, fill forms, extracte web content",
        # ),
        # api_agent.as_tool(
        #     tool_name="api_tool",
        #     tool_description="Use this tool for tasks involving direct API interactions, such as fetching data from an API endpoint or sending data to an API. Provide the API endpoint, method, and payload as input.",
        # ),
        game_designer_agent.as_tool(
            tool_name="game_designer",
            tool_description="design game, app UI/UX",
        ),
        software_engineer_agent.as_tool(
            tool_name="software_engineer",
            tool_description="write COMPLETE code for requested tasks",
        ),
        # Potentially add the 'update_agent_state' tool back if needed for structured thinking,
        # although the example implies the orchestrator directly calls browser/api tools.
        # update_agent_state_tool, # Requires defining this tool using @function_tool
    ],
    # model_settings=ModelSettings(tool_choice="required"), # Optional: Force tool use
    model="gpt-4o-mini"
)
logger.info("Defined orchestrator_agent with browser_tool and api_tool")


