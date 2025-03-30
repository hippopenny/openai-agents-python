from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from agents import Agent, ModelSettings

from .context import BrowserAgentContext
from .prompts import SystemPromptBuilder
from .tools import BROWSER_TOOLS

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def create_browser_agent(
    llm: BaseChatModel,
    task: str,
    use_vision: bool = True,
    max_actions_per_step: int = 10, # Passed to prompt builder, not directly enforced by Agent SDK
    tool_calling_method: Optional[str] = 'auto', # e.g., 'function_calling' for OpenAI
) -> Agent[BrowserAgentContext]:
    """
    Creates the main browser agent instance using the Agent SDK.
    """

    # 1. Build the System Prompt Content
    # Action descriptions are implicitly handled by the Agent SDK via tool definitions.
    # We provide a placeholder here as the builder expects it, but it's not strictly used.
    action_description = "Browser interaction tools are available."
    prompt_builder = SystemPromptBuilder(
        action_description=action_description,
        max_actions_per_step=max_actions_per_step
    )
    system_prompt_content = prompt_builder.get_system_message_content(task=task)

    # 2. Configure Model Settings
    model_settings = ModelSettings()
    if use_vision:
        logger.info("Vision mode enabled for agent (ensure LLM supports image input).")
        # Actual vision enablement depends on the LLM and how image data is passed (via HumanMessage content).

    # Determine tool calling method if 'auto'
    effective_tool_calling_method = None # Default to None (SDK default)
    if tool_calling_method == 'auto':
        model_library = llm.__class__.__name__
        if model_library in ('ChatOpenAI', 'AzureChatOpenAI'):
            effective_tool_calling_method = 'function_calling'
            logger.info("Auto-detected OpenAI model; using 'function_calling' method.")
        # Add elif for other models like Gemini if needed
        # elif model_library == 'ChatGoogleGenerativeAI':
        #     effective_tool_calling_method = 'tool_calling' # Check correct method name for Gemini
        #     logger.info("Auto-detected Google model; using 'tool_calling' method.")
        else:
            logger.info("Tool calling method set to SDK default (auto-detection failed or model doesn't require specific method).")
    else:
        effective_tool_calling_method = tool_calling_method # Use explicitly provided method
        logger.info(f"Using explicitly provided tool calling method: '{effective_tool_calling_method}'")


    # Set tool_choice in model_settings if a method was determined
    # Note: Some models/methods might not require explicit tool_choice setting.
    # Adjust based on specific LLM requirements.
    # Setting it might force tool use, which could be desirable or problematic depending on the flow.
    # For browser agent, forcing tool use (at least update_agent_state) is likely intended.
    # model_settings.tool_choice = effective_tool_calling_method # Example: Might force tool use

    # 3. Create the Agent Instance
    browser_agent = Agent[BrowserAgentContext](
        name="Browser Agent",
        instructions=system_prompt_content,
        tools=BROWSER_TOOLS,
        model_settings=model_settings,
        # Context, max_turns, hooks, run_config are managed by the Runner.
        # output_type=None, # Agent output is primarily via tool calls and state updates.
    )

    logger.info(f"Browser Agent created with {len(BROWSER_TOOLS)} tools.")
    return browser_agent
