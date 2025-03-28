import os

from agents import Agent
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from .context import CoderContext

# Placeholder instructions - refine these based on desired coder behavior
AIDER_INSTRUCTIONS = """
You are 'aider', a coding agent. You receive specific, well-defined coding tasks from a planner agent.
Execute the task described in the input.
Provide the results of your work, including any code generated, file changes, or error messages.
If you encounter issues or cannot complete the task, explain the problem clearly.
Focus solely on the single task provided. Do not attempt to manage the overall plan.
"""

# --- Configuration for OpenAI Proxy ---
# Adjust these as needed for your proxy setup
AIDER_PROXY_BASE_URL = os.environ.get("AIDER_PROXY_BASE_URL", "http://localhost:8080/v1") # Example URL
AIDER_PROXY_API_KEY = os.environ.get("AIDER_PROXY_API_KEY", "dummy-key") # Use proxy key if required
AIDER_MODEL_NAME = os.environ.get("AIDER_MODEL_NAME", "gpt-4o") # Model served by the proxy


def create_aider_agent() -> Agent[CoderContext]:
    """Creates the aider agent configured to use the OpenAI proxy."""

    # Configure the OpenAI client to point to your proxy server
    proxied_openai_client = AsyncOpenAI(
        base_url=AIDER_PROXY_BASE_URL,
        api_key=AIDER_PROXY_API_KEY,
    )

    # Use the proxied client with the OpenAIResponsesModel
    aider_model = OpenAIResponsesModel(
        model=AIDER_MODEL_NAME,
        openai_client=proxied_openai_client,
    )

    aider_agent = Agent[CoderContext](
        model=aider_model,
        instructions=AIDER_INSTRUCTIONS,
        # Add tools specific to the coder if needed (e.g., file system access)
        # tools=[...],
        handoff_description="An AI agent specialized in writing and modifying code based on specific instructions.",
    )
    return aider_agent
