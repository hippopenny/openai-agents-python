import asyncio
import logging
from typing import Any, cast

from agents import (
    Agent,
    FunctionToolResult,
    RunContextWrapper,
    ToolExecutionError,
    function_tool,
)

# Keep original ActionResult for tool return type consistency
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.exceptions import ActionException
from browser_use.controller.service import Controller

from .context import BrowserAgentContext
from .views import (
    AgentStateUpdate,
    AskHumanParams,
    CheckboxParams,
    ClickParams,
    CloseTabParams,
    DoneParams,
    ExtractContentParams,
    GoBackParams,
    GoToURLParams,
    InputTextParams,
    OpenNewTabParams,
    ScrollParams,
    SelectOptionParams,
    SwitchTabParams,
    UploadFileParams,
)

logger = logging.getLogger(__name__)

# Helper function to execute controller actions and handle errors
async def _execute_browser_action(
    controller: Controller,
    browser_context: BrowserContext,
    action_name: str,
    params: dict[str, Any],
    context: BrowserAgentContext,
) -> ActionResult:
    """Executes a single action using the controller and returns ActionResult."""
    action_model_type = controller.registry.get_action_model_by_name(action_name)
    if not action_model_type:
        # Use ToolExecutionError for errors within tools
        raise ToolExecutionError(f"Action '{action_name}' not found in controller registry.")

    # Validate parameters using the specific action's Pydantic model
    try:
        validated_params = action_model_type(**params)
        # The controller expects the action wrapped in its dynamic model
        action_list_item = {action_name: validated_params}
        # Create the dynamic ActionModel instance expected by controller.act
        action_model_instance = controller.registry.create_action_model()(**action_list_item)
    except Exception as e:
        raise ToolExecutionError(f"Invalid parameters for action '{action_name}': {e}")

    try:
        # Use controller.act for single actions
        results: list[ActionResult] = await controller.act(
            action=action_model_instance,
            browser_context=browser_context,
            page_extraction_llm=context.page_extraction_llm,
            sensitive_data=context.sensitive_data,
            available_file_paths=context.available_file_paths,
            # Add check_break_if_paused if pause/stop is implemented in runner/context
            # check_break_if_paused=lambda: context.check_if_paused_or_stopped(),
        )
        if results:
            # Log success/failure based on result content
            if results[0].error:
                 logger.warning(f"Action '{action_name}' executed but resulted in error: {results[0].error}")
            elif results[0].extracted_content:
                 logger.info(f"Action '{action_name}' executed successfully with extracted content.")
            else:
                 logger.info(f"Action '{action_name}' executed successfully.")
            return results[0]
        else:
            logger.warning(f"Controller.act for '{action_name}' returned no results unexpectedly.")
            # Return an error result if no specific result came back
            return ActionResult(error=f"Action '{action_name}' completed but controller returned no result.", include_in_memory=True)

    except ActionException as e:
        # Log controller-level action exceptions
        logger.warning(f"Action '{action_name}' failed during execution: {e}")
        # Return error in ActionResult format so agent can see it
        return ActionResult(error=str(e), include_in_memory=True)
    except Exception as e:
        # Catch unexpected errors during action execution
        logger.error(f"Unexpected error during action '{action_name}': {e}", exc_info=True)
        # Return error in ActionResult format
        return ActionResult(error=f"Unexpected error executing {action_name}: {e}", include_in_memory=True)


# --- State Update Tool ---

@function_tool
async def update_agent_state(
    context: RunContextWrapper[BrowserAgentContext],
    page_summary: str,
    evaluation_previous_goal: str,
    memory: str,
    next_goal: str,
) -> AgentStateUpdate:
    """
    Call this tool FIRST in your response. Update the agent's internal analysis of the current state,
    evaluation of the previous action's success, task memory, and the goal for the next action(s).
    """
    state_update = AgentStateUpdate(
        page_summary=page_summary,
        evaluation_previous_goal=evaluation_previous_goal,
        memory=memory,
        next_goal=next_goal,
    )
    # Store the latest state update in the shared context
    context.context.current_agent_state = state_update
    logger.info(f"🧠 Agent State Updated: Eval='{evaluation_previous_goal}', Memory='{memory}', Next Goal='{next_goal}'")
    # Return the state update object itself - Agent SDK handles passing this back if needed
    return state_update


# --- Browser Action Tools ---
# Descriptions are now more concise as the main logic is in the system prompt.

@function_tool(use_docstring_info=False)
async def go_to_url(context: RunContextWrapper[BrowserAgentContext], params: GoToURLParams) -> ActionResult:
    """Navigates the current browser tab to the specified URL."""
    logger.info(f"🛠️ Executing: go_to_url(url='{params.url}')")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'go_to_url', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def click_element(context: RunContextWrapper[BrowserAgentContext], params: ClickParams) -> ActionResult:
    """Clicks the interactive element specified by its index."""
    logger.info(f"🛠️ Executing: click_element(index={params.index})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'click_element', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def input_text(context: RunContextWrapper[BrowserAgentContext], params: InputTextParams) -> ActionResult:
    """Inputs text into the form field element specified by its index. Optionally clears the field first."""
    # Mask sensitive data before logging
    log_text = params.text
    if context.context.sensitive_data:
        # Basic masking check
        for key, value in context.context.sensitive_data.items():
            if isinstance(value, str) and value == params.text and value: # Ensure value is comparable string
                log_text = f"*** SENSITIVE DATA ({key}) ***"
                break
    logger.info(f"🛠️ Executing: input_text(index={params.index}, text='{log_text}', clear={params.clear_before_input})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'input_text', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def scroll(context: RunContextWrapper[BrowserAgentContext], params: ScrollParams) -> ActionResult:
    """Scrolls the page 'up' or 'down', optionally by a specific number of pixels."""
    logger.info(f"🛠️ Executing: scroll(direction='{params.direction}', pixels={params.pixels})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'scroll', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def select_option(context: RunContextWrapper[BrowserAgentContext], params: SelectOptionParams) -> ActionResult:
    """Selects an option within a <select> element specified by their indices."""
    logger.info(f"🛠️ Executing: select_option(index={params.index}, option_index={params.option_index})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'select_option', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def checkbox_set(context: RunContextWrapper[BrowserAgentContext], params: CheckboxParams) -> ActionResult:
    """Checks (true) or unchecks (false) a checkbox element specified by its index."""
    logger.info(f"🛠️ Executing: checkbox_set(index={params.index}, checked={params.checked})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'checkbox_set', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def extract_page_content(context: RunContextWrapper[BrowserAgentContext], params: ExtractContentParams) -> ActionResult:
    """Extracts and summarizes the main textual content of the current page. Use an optional query to focus extraction on specific information."""
    logger.info(f"🛠️ Executing: extract_page_content(query='{params.query}')")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'extract_page_content', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def open_new_tab(context: RunContextWrapper[BrowserAgentContext], params: OpenNewTabParams) -> ActionResult:
    """Opens a new, empty browser tab and switches focus to it."""
    logger.info("🛠️ Executing: open_new_tab()")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'open_new_tab', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def close_tab(context: RunContextWrapper[BrowserAgentContext], params: CloseTabParams) -> ActionResult:
    """Closes the currently active browser tab."""
    logger.info("🛠️ Executing: close_tab()")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'close_tab', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def switch_tab(context: RunContextWrapper[BrowserAgentContext], params: SwitchTabParams) -> ActionResult:
    """Switches focus to the browser tab specified by its index (from 'Available Tabs' list in the input)."""
    logger.info(f"🛠️ Executing: switch_tab(tab_index={params.tab_index})")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'switch_tab', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def go_back(context: RunContextWrapper[BrowserAgentContext], params: GoBackParams) -> ActionResult:
    """Navigates back to the previous page in the current tab's history."""
    logger.info("🛠️ Executing: go_back()")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'go_back', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def upload_file(context: RunContextWrapper[BrowserAgentContext], params: UploadFileParams) -> ActionResult:
    """Uploads a local file using the file input element specified by its index. Requires the file path to be allowed."""
    logger.info(f"🛠️ Executing: upload_file(index={params.index}, file_path='{params.file_path}')")
    # Security check: Ensure file path is allowed if available_file_paths is set
    if context.context.available_file_paths and params.file_path not in context.context.available_file_paths:
         raise ToolExecutionError(f"File path '{params.file_path}' is not in the allowed list.")
    return await _execute_browser_action(
        context.context.controller, context.context.browser_context, 'upload_file', params.model_dump(), context.context
    )

@function_tool(use_docstring_info=False)
async def ask_human(context: RunContextWrapper[BrowserAgentContext], params: AskHumanParams) -> ActionResult:
    """Pauses execution and asks a human operator for help or input via an external mechanism. Returns the human's response."""
    # This is a placeholder. Real implementation would involve callbacks, pausing the run,
    # waiting for external input, and then resuming.
    logger.warning(f"⚠️ HUMAN INTERVENTION REQUESTED: {params.question}")
    # Simulate asking and getting a generic response or potentially raising a specific exception/status
    await asyncio.sleep(1) # Simulate delay for human response
    response = f"Human was asked: '{params.question}'. (Simulation: No response received)."
    # Return success, indicating the question was asked, include response/status in extracted_content
    # A real implementation might return a special status or require the runner to handle pausing.
    return ActionResult(extracted_content=response, include_in_memory=True)


@function_tool(use_docstring_info=False)
async def done(context: RunContextWrapper[BrowserAgentContext], params: DoneParams) -> ActionResult:
    """
    Call this tool ONLY when the entire task, including all sub-steps and repetitions, is fully complete.
    Provide the final answer or result required by the original task description in the 'final_answer' parameter.
    This signals the successful end of the agent's execution.
    """
    logger.info(f"✅ Task Complete: done(final_answer='{params.final_answer}')")
    context.context.is_done = True # Set flag in shared context
    # Return ActionResult indicating done status and the final answer
    return ActionResult(is_done=True, extracted_content=params.final_answer, include_in_memory=True)


# List of all tools for the agent
BROWSER_TOOLS = [
    update_agent_state,
    go_to_url,
    click_element,
    input_text,
    scroll,
    select_option,
    checkbox_set,
    extract_page_content,
    open_new_tab,
    close_tab,
    switch_tab,
    go_back,
    upload_file,
    ask_human,
    done,
]
