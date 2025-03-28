import os

from agents import Agent
# Assuming the provider is located here. Please verify this path.
from agents.models.hippopenny_aider_provider import HippoPennyAiderModelProvider

from .context import CoderContext

# Refined instructions
AIDER_INSTRUCTIONS = """
You are 'aider', an expert coding assistant working via an API.
You will receive *one* specific, well-defined coding task as input.
Your goal is to execute this single task accurately.

**Instructions:**
1.  **Understand the Task:** Carefully read the task description provided in the user message.
2.  **Execute:** Perform the coding task. This might involve writing code, modifying files (conceptually, as you don't have direct file access unless provided via tools), or answering a coding-related question.
3.  **Output:**
    *   **On Success:** Provide the direct result of your work. This could be code snippets, confirmation of changes, or the answer to a question. Be concise and directly address the task.
    *   **On Failure:** If you cannot complete the task due to ambiguity, errors, or limitations, clearly state the problem. Start your response with "Error:".

**Important:**
*   Focus *only* on the single task given. Do not refer to past tasks or the overall plan unless the task explicitly requires it.
*   Do not add conversational filler. Return only the result or the error message.
*   Assume necessary context (like file contents) would be provided *within* the task description if needed, or via tools (if any were configured).
"""

# --- Configuration for OpenAI Proxy ---
# Adjust these as needed for your proxy setup
AIDER_PROXY_BASE_URL = os.environ.get("AIDER_PROXY_BASE_URL", "http://localhost:8080/v1") # Example URL
AIDER_PROXY_API_KEY = os.environ.get("AIDER_PROXY_API_KEY", "dummy-key") # Use proxy key if required
AIDER_MODEL_NAME = os.environ.get("AIDER_MODEL_NAME", "gpt-4o") # Model served by the proxy


def create_aider_agent() -> Agent[CoderContext]:
    """Creates the aider agent configured to use the HippoPennyAiderModelProvider."""

    # Configure the HippoPennyAiderModelProvider
    # It takes no arguments, presumably reads config from environment or defaults
    aider_model_provider = HippoPennyAiderModelProvider()

    # Get the actual model instance from the provider
    # Assuming a .get_model() method exists. Adjust if the method name is different.
    aider_model = aider_model_provider.get_model()

    aider_agent = Agent[CoderContext](
        model=aider_model, # Pass the actual model instance
        instructions=AIDER_INSTRUCTIONS,
        name="AiderAgent", # Give the agent a name
        # Add tools specific to the coder if needed (e.g., file system access tools)
        # tools=[read_file_tool, write_file_tool],
        handoff_description="An AI agent specialized in writing and modifying code based on specific instructions.",
        # Consider adding an output_type if aider should return structured data,
        # otherwise, it will return text.
        # output_type=AiderTaskResult,
    )
    return aider_agent
