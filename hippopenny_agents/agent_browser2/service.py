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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

# Import from Agent SDK
from agents import (
    Agent,
    FunctionToolResult,
    ItemHelpers,
    MessageOutputItem,
    RunConfig,
    RunHooks,
    Runner,
    TResponseInputItem,
    ToolInputItem,
    ToolOutputItem,
    trace,
)
from agents.util._transforms import transform_string_function_style

# Imports from original browser_use components
from browser_use.agent.views import AgentStepInfo
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async

# Imports from this refactored package
from .agent import create_browser_agent
from .context import BrowserAgentContext
from .prompts import AgentMessagePrompt, PlannerPromptBuilder
from .tools import BROWSER_TOOLS # Import for potential direct use if needed
from .views import ActionResult, AgentHistory, AgentHistoryList, AgentStateUpdate, PlannerOutput

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BrowserAgentRunner:
    """
    Orchestrates the execution of the browser agent using the Agent SDK's Runner.
    Includes logic for initialization, step execution, history management,
    planner invocation, and GIF generation.
    """
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser: Browser | None = None,
        browser_context: BrowserContext | None = None,
        controller: Controller = Controller(),
        use_vision: bool = True,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        # max_failures_per_step: int = 3, # Handled within step logic/retries if needed
        # retry_delay: int = 10, # Handled within step logic/retries if needed
        # max_input_tokens: int = 128000, # Context management not explicitly added here
        generate_gif: bool | str = True,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        include_attributes: list[str] = [
            'title', 'type', 'name', 'role', 'tabindex', 'aria-label',
            'placeholder', 'value', 'alt', 'aria-expanded',
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10, # Passed to prompt builder
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Cloud Callbacks - Can be implemented using RunHooks if needed
        tool_calling_method: Optional[str] = 'auto',
        page_extraction_llm: Optional[BaseChatModel] = None,
        # Planner specific parameters
        planner_llm: Optional[BaseChatModel] = None,
        planning_interval: int = 1, # Run planner every N steps
        use_vision_for_planner: bool = False,
    ):
        self.agent_id = str(uuid.uuid4())
        self.task = task
        self.llm = llm
        self.use_vision = use_vision
        self.save_conversation_path = save_conversation_path
        self.save_conversation_path_encoding = save_conversation_path_encoding
        self.include_attributes = include_attributes
        self.max_error_length = max_error_length
        self.generate_gif = generate_gif
        self.sensitive_data = sensitive_data
        self.available_file_paths = available_file_paths
        self.initial_actions = initial_actions
        self.page_extraction_llm = page_extraction_llm or llm

        # Planner setup
        self.planner_llm = planner_llm
        self.planning_interval = planning_interval if planner_llm else 0 # Disable if no planner LLM
        self.use_vision_for_planner = use_vision_for_planner if planner_llm else False

        # Telemetry setup
        self.telemetry = ProductTelemetry()
        self._set_version_and_source()
        self._set_model_names() # Includes planner model name now

        # Browser setup
        self.injected_browser = browser is not None
        self.injected_browser_context = browser_context is not None
        self.browser = browser if browser is not None else (None if browser_context else Browser())
        if browser_context:
            self.browser_context_instance = browser_context
        elif self.browser:
            context_config = getattr(self.browser, 'config', {}).get('new_context_config', {})
            self.browser_context_instance = BrowserContext(browser=self.browser, config=context_config)
        else:
            # If neither browser nor context is provided, create both
            logger.info("No existing browser or context provided, creating new instances.")
            self.browser = Browser()
            self.browser_context_instance = BrowserContext(browser=self.browser)

        # Controller setup
        self.controller = controller

        # Create Agent instance
        self.agent = create_browser_agent(
            llm=self.llm,
            task=self.task,
            use_vision=self.use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
        )

        # Initialize Agent Context (passed to Runner.run)
        self.agent_context = BrowserAgentContext(
            task=self.task,
            browser_context=self.browser_context_instance,
            controller=self.controller,
            use_vision=self.use_vision,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            sensitive_data=self.sensitive_data,
            available_file_paths=self.available_file_paths,
            agent_id=self.agent_id,
            telemetry=self.telemetry,
            page_extraction_llm=self.page_extraction_llm,
            planner_llm=self.planner_llm,
            planning_interval=self.planning_interval,
            use_vision_for_planner=self.use_vision_for_planner,
            # history is managed externally by the loop/AgentHistoryList
            # last_plan is updated by _run_planner
        )

        # History tracking
        self.full_history = AgentHistoryList(history=[])
        self.current_conversation_messages: list[BaseMessage] = [] # Store LangChain messages

        # Control flags
        self._paused = False
        self._stopped = False

        if save_conversation_path:
            logger.info(f'Saving conversation steps to {save_conversation_path}_<step>.txt')


    def _set_version_and_source(self) -> None:
        # (Same as original)
        try:
            import pkg_resources
            self.version = pkg_resources.get_distribution('browser-use').version
            self.source = 'pip'
        except Exception:
            try:
                import subprocess
                self.version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
                self.source = 'git'
            except Exception:
                self.version = 'unknown'
                self.source = 'unknown'
        logger.debug(f'Version: {self.version}, Source: {self.source}')

    def _set_model_names(self) -> None:
         # (Adapted from original, includes planner)
        self.chat_model_library = self.llm.__class__.__name__
        if hasattr(self.llm, 'model_name'): self.model_name = self.llm.model_name
        elif hasattr(self.llm, 'model'): self.model_name = self.llm.model
        else: self.model_name = 'Unknown'

        self.planner_model_name = None
        if self.planner_llm:
            if hasattr(self.planner_llm, 'model_name'): self.planner_model_name = self.planner_llm.model_name
            elif hasattr(self.planner_llm, 'model'): self.planner_model_name = self.planner_llm.model
            else: self.planner_model_name = 'Unknown (Planner)'
        logger.debug(f"Main Model: {self.model_name}, Planner Model: {self.planner_model_name or 'N/A'}")


    def _log_agent_run_start(self) -> None:
        """Log the agent run start"""
        logger.info(f"🚀 Starting task: {self.task} (Agent ID: {self.agent_id})")
        logger.debug(f'Version: {self.version}, Source: {self.source}')
        self.telemetry.capture(
            AgentRunTelemetryEvent(
                agent_id=self.agent_id,
                use_vision=self.use_vision,
                task=self.task,
                model_name=self.model_name,
                chat_model_library=self.chat_model_library,
                version=self.version,
                source=self.source,
                # Add planner info if desired
                planner_model_name=self.planner_model_name,
            )
        )

    def _log_agent_run_end(self, steps_taken: int, max_steps: int) -> None:
        """Log the agent run end"""
        is_done = self.full_history.is_done()
        max_steps_reached = steps_taken >= max_steps and not is_done
        logger.info(f"🏁 Agent run finished. Success: {is_done}, Steps: {steps_taken}, Max steps reached: {max_steps_reached}")
        self.telemetry.capture(
            AgentEndTelemetryEvent(
                agent_id=self.agent_id,
                success=is_done,
                steps=steps_taken,
                max_steps_reached=max_steps_reached,
                errors=self.full_history.errors(),
            )
        )

    async def _execute_initial_actions(self) -> List[ActionResult]:
        """Executes predefined initial actions before the main loop."""
        if not self.initial_actions:
            return []

        logger.info("Executing initial actions...")
        results = []
        action_model_base = self.controller.registry.create_action_model() # Get the dynamic base model

        converted_actions = []
        for action_dict in self.initial_actions:
            action_name = next(iter(action_dict))
            params = action_dict[action_name]
            action_info = self.controller.registry.registry.actions.get(action_name)
            if not action_info:
                 logger.error(f"Initial action '{action_name}' not found in registry. Skipping.")
                 continue
            param_model = action_info.param_model
            try:
                validated_params = param_model(**params)
                # Create instance of the dynamic ActionModel
                action_instance = action_model_base(**{action_name: validated_params})
                converted_actions.append(action_instance)
            except ValidationError as e:
                 logger.error(f"Invalid parameters for initial action '{action_name}': {e}. Skipping.")
                 continue

        if not converted_actions:
             logger.warning("No valid initial actions to execute.")
             return []

        # Use multi_act for the sequence of initial actions
        try:
            initial_results = await self.controller.multi_act(
                converted_actions,
                self.browser_context_instance,
                check_for_new_elements=False, # As per original logic
                page_extraction_llm=self.page_extraction_llm,
                sensitive_data=self.sensitive_data,
                available_file_paths=self.available_file_paths,
                # check_break_if_paused - add if pause/stop needed here
            )
            results.extend(initial_results)
            logger.info(f"Initial actions completed with {len(results)} results.")
        except Exception as e:
            logger.error(f"Error executing initial actions sequence: {e}", exc_info=True)
            results.append(ActionResult(error=f"Initial actions failed: {e}", include_in_memory=True))

        return results

    @staticmethod
    def _convert_to_langchain_message(item: TResponseInputItem) -> BaseMessage:
        """Converts SDK input item dict to LangChain BaseMessage."""
        role = item.get("role")
        content = item.get("content")
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            # Handle potential tool calls within assistant message if needed,
            # though Runner usually handles this conversion internally.
            # For simplicity, treat as plain AIMessage for history tracking here.
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        # Add handling for tool messages if necessary, though Runner input usually doesn't require them directly.
        else:
            logger.warning(f"Unknown role '{role}' in input item, treating as AIMessage.")
            return AIMessage(content=content) # Default or raise error

    THINK_TAGS_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
    JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON block or parses the whole string as JSON."""
        # Remove think tags first (like in deepseek models)
        text = re.sub(self.THINK_TAGS_RE, '', text).strip()

        # Try extracting ```json ... ``` block
        match = self.JSON_BLOCK_RE.search(text)
        if match:
            json_str = match.group(1)
        else:
            # Assume the whole text might be JSON
            json_str = text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from response: {text[:200]}...")
            return None

    async def _run_planner(self, current_messages: list[BaseMessage]) -> PlannerOutput | str | None:
        """Invokes the planner LLM and returns the structured plan or raw output."""
        if not self.planner_llm:
            return None

        logger.info("🧠 Running Planner...")
        planner_prompt_builder = PlannerPromptBuilder()
        planner_system_message = SystemMessage(content=planner_prompt_builder.get_system_message_content())

        # Prepare messages for the planner
        # Use current conversation history, excluding the initial system prompt if present
        planner_input_messages = [planner_system_message]
        # Start from index 1 if the first message is the main agent's system prompt
        start_index = 1 if isinstance(current_messages[0], SystemMessage) else 0
        planner_input_messages.extend(current_messages[start_index:])

        # Handle vision for planner
        if not self.agent_context.use_vision_for_planner and self.use_vision:
            # Remove image content from the last message if it exists
            last_msg = planner_input_messages[-1]
            if isinstance(last_msg.content, list):
                new_content = [part for part in last_msg.content if part.get("type") != "image_url"]
                # If only text remains, convert content back to string
                if len(new_content) == 1 and new_content[0].get("type") == "text":
                     planner_input_messages[-1] = last_msg.__class__(content=new_content[0].get("text", ""))
                else:
                     planner_input_messages[-1] = last_msg.__class__(content=new_content)


        try:
            # Invoke the planner LLM
            # Using invoke for simplicity, assuming planner doesn't need streaming/tools
            response = await self.planner_llm.ainvoke(planner_input_messages)
            raw_plan_content = response.content
            logger.debug(f"Planner Raw Output: {raw_plan_content}")

            # Attempt to parse the JSON output
            parsed_json = self._extract_json_from_response(raw_plan_content)

            if parsed_json:
                try:
                    # Validate against the PlannerOutput model
                    plan_output = PlannerOutput.model_validate(parsed_json)
                    logger.info(f"Planner Output (Parsed): {plan_output.model_dump_json(indent=2)}")
                    return plan_output
                except ValidationError as e:
                    logger.warning(f"Planner output failed Pydantic validation: {e}. Returning raw content.")
                    return raw_plan_content # Return raw string on validation failure
            else:
                logger.warning("Planner output was not valid JSON. Returning raw content.")
                return raw_plan_content # Return raw string if JSON parsing failed

        except Exception as e:
            logger.error(f"Error running planner: {e}", exc_info=True)
            return f"Error during planning: {e}" # Return error string

    @time_execution_async('--agent_run')
    @trace("Browser Agent Run") # Wrap the entire run in a trace
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the browser task using the Agent SDK Runner."""
        self._log_agent_run_start()
        self.agent_context.max_steps = max_steps
        current_step = 0
        # Use LangChain messages for internal history tracking
        self.current_conversation_messages = []
        # Add initial system message for the main agent
        self.current_conversation_messages.append(SystemMessage(content=self.agent.instructions))

        try:
            # --- Initial Actions ---
            last_step_results = await self._execute_initial_actions()
            self.agent_context.last_results = last_step_results
            # Add initial results to conversation history? Maybe as a user message?
            # For now, they are passed via context to the first AgentMessagePrompt

            # --- Main Loop ---
            for step in range(max_steps):
                current_step = step
                self.agent_context.current_step = current_step
                logger.info(f"📍 Step {current_step + 1}/{max_steps}")

                # --- Check Control Flags ---
                if self._stopped: logger.info("Agent stopped."); break
                while self._paused:
                    logger.debug("Agent paused..."); await asyncio.sleep(0.2)
                    if self._stopped: logger.info("Agent stopped while paused."); break
                if self._stopped: break

                # --- Run Planner (if applicable) ---
                current_plan = None
                if self.planning_interval > 0 and (current_step % self.planning_interval == 0):
                    current_plan = await self._run_planner(self.current_conversation_messages)
                    self.agent_context.last_plan = current_plan # Update context

                # --- Prepare Input for Main Agent ---
                browser_state = await self.browser_context_instance.get_state()
                step_info = AgentStepInfo(step_number=current_step, max_steps=max_steps)
                message_prompt = AgentMessagePrompt(
                    state=browser_state,
                    result=last_step_results, # Results from the *previous* step
                    plan=self.agent_context.last_plan, # Use latest plan from context
                    include_attributes=self.include_attributes,
                    max_error_length=self.max_error_length,
                    step_info=step_info,
                )
                human_message_content = message_prompt.get_user_message_content(use_vision=self.use_vision)
                current_human_message = HumanMessage(content=human_message_content)

                # Prepare input list for Runner.run (includes history)
                runner_input: list[TResponseInputItem] = [
                    msg.dict() for msg in self.current_conversation_messages # Convert history to dicts
                ]
                runner_input.append(current_human_message.dict()) # Add current user message dict

                # --- Run Agent Step ---
                step_history = AgentHistory(
                    browser_state=BrowserStateHistory.from_browser_state(browser_state),
                    plan=current_plan # Record the plan generated *for* this step
                )
                tool_results_for_step: list[ActionResult] = []
                tool_calls_for_step: list[Dict[str, Any]] = []
                agent_state_update_for_step: AgentStateUpdate | None = None
                assistant_response_content = "" # To store final text response if any

                try:
                    run_result = await Runner.run(
                        starting_agent=self.agent,
                        input=runner_input, # Pass dict list
                        context=self.agent_context,
                        max_turns=3, # Allow for Agent -> Tools -> Agent response cycle if needed
                        # hooks=...
                        # run_config=...
                    )

                    # --- Process Step Output ---
                    # Add the human message that triggered this turn to internal history
                    self.current_conversation_messages.append(current_human_message)

                    # Process new items from the run result
                    for item in run_result.new_items:
                        if isinstance(item, ToolInputItem):
                            tool_name = transform_string_function_style(item.tool_name)
                            tool_calls_for_step.append({tool_name: item.tool_input})
                            # Add ToolInputItem representation to LangChain history? Optional.
                        elif isinstance(item, ToolOutputItem):
                            if isinstance(item.tool_output, ActionResult):
                                tool_results_for_step.append(item.tool_output)
                            elif isinstance(item.tool_output, AgentStateUpdate):
                                agent_state_update_for_step = item.tool_output
                                self.agent_context.current_agent_state = agent_state_update_for_step
                            else:
                                logger.warning(f"Unexpected tool output type: {type(item.tool_output)}")
                                tool_results_for_step.append(ActionResult(extracted_content=str(item.tool_output)))
                            # Add ToolOutputItem representation to LangChain history? Optional.
                        elif isinstance(item, MessageOutputItem):
                            # This is the final assistant message after tools (if any)
                            assistant_response_content = ItemHelpers.text_message_output(item)
                            logger.info(f"Assistant Response: {assistant_response_content}")
                            # Add the final AIMessage to LangChain history
                            self.current_conversation_messages.append(AIMessage(content=assistant_response_content))

                    # If no explicit AIMessage was the last item, create one from tool calls/state?
                    # Usually Runner ensures a final MessageOutputItem if max_turns allows.
                    if not isinstance(run_result.new_items[-1], MessageOutputItem):
                         logger.debug("Run ended with tool call/output, no final assistant message item.")
                         # Optionally add a placeholder AIMessage if needed for history consistency
                         # self.current_conversation_messages.append(AIMessage(content="[Agent completed turn with tool execution]"))


                    # Update step history record
                    step_history.agent_state_update = agent_state_update_for_step
                    step_history.tool_calls = tool_calls_for_step
                    step_history.tool_results = tool_results_for_step
                    self.full_history.history.append(step_history)

                    # Update results for the *next* step's prompt
                    last_step_results = tool_results_for_step

                    # --- Save Conversation Step ---
                    if self.save_conversation_path:
                        # Pass runner_input and run_result.new_items
                        self._save_conversation_step(current_step + 1, runner_input, run_result.new_items)

                    # --- Telemetry ---
                    self.telemetry.capture(
                        AgentStepTelemetryEvent(
                            agent_id=self.agent_id,
                            step=current_step + 1,
                            actions=[call for call in tool_calls_for_step],
                            consecutive_failures=0, # Reset/manage failure count if needed
                            step_error=[r.error for r in tool_results_for_step if r.error],
                        )
                    )

                    # --- Check for Done ---
                    if self.agent_context.is_done:
                        logger.info("✅ Task completed successfully (done tool called).")
                        done_result = next((r for r in tool_results_for_step if r.is_done), None)
                        self.agent_context.last_results = [done_result] if done_result else []
                        break

                except Exception as e:
                    logger.error(f"❌ Error during agent step {current_step + 1}: {e}", exc_info=True)
                    error_result = ActionResult(error=f"Step failed: {e}", include_in_memory=True)
                    step_history.tool_results.append(error_result)
                    if not any(h is step_history for h in self.full_history.history): # Avoid duplicates if already added
                         self.full_history.history.append(step_history)
                    last_step_results = [error_result]
                    # Add error message to conversation history
                    self.current_conversation_messages.append(current_human_message) # Add the input that caused error
                    self.current_conversation_messages.append(AIMessage(content=f"[ERROR] Step failed: {e}"))
                    break # Exit loop on error

            else: # Max steps reached
                 if not self.agent_context.is_done:
                    logger.warning(f"⚠️ Task not completed within maximum steps ({max_steps}).")

            return self.full_history

        finally:
            self._log_agent_run_end(current_step + 1, max_steps)
            await self._cleanup_resources() # Consolidate cleanup

            # --- Generate GIF ---
            if self.generate_gif:
                output_path: str = 'agent_history_sdk.gif'
                if isinstance(self.generate_gif, str): output_path = self.generate_gif
                try: self.create_history_gif(output_path=output_path)
                except Exception as gif_e: logger.error(f"Failed to generate GIF: {gif_e}", exc_info=True)

    async def _cleanup_resources(self):
        """Closes browser and context if they were created by this runner."""
        logger.debug("Cleaning up resources...")
        if not self.injected_browser_context and hasattr(self, 'browser_context_instance'):
            try:
                await self.browser_context_instance.close()
                logger.debug("Browser context closed.")
            except Exception as e:
                logger.error(f"Error closing browser context: {e}", exc_info=True)
        if not self.injected_browser and self.browser:
            try:
                await self.browser.close()
                logger.debug("Browser closed.")
            except Exception as e:
                logger.error(f"Error closing browser: {e}", exc_info=True)


    def _save_conversation_step(self, step_num: int, input_items: list[TResponseInputItem], output_items: list) -> None:
        """Saves the input and output of a single step to a file."""
        # (Identical to previous implementation)
        if not self.save_conversation_path: return
        filepath = Path(f"{self.save_conversation_path}_{step_num}.txt")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, 'w', encoding=self.save_conversation_path_encoding) as f:
                f.write("--- INPUT MESSAGES ---\n")
                for item in input_items:
                    role = item.get('role', 'unknown')
                    content = item.get('content', '')
                    f.write(f"Role: {role}\n")
                    if isinstance(content, list):
                        for part in content:
                            if part.get('type') == 'text': f.write(f"Content (text):\n{part.get('text', '')}\n")
                            elif part.get('type') == 'image_url': f.write("Content (image): [base64 data omitted]\n")
                    elif isinstance(content, str): f.write(f"Content:\n{content}\n")
                    f.write("-" * 10 + "\n")

                f.write("\n--- OUTPUT ITEMS ---\n")
                for item in output_items:
                     f.write(f"Type: {item.__class__.__name__}\n")
                     if isinstance(item, ToolInputItem):
                         f.write(f"Tool Name: {item.tool_name}\n")
                         try: f.write(f"Input: {json.dumps(item.tool_input, indent=2)}\n")
                         except TypeError: f.write(f"Input: {item.tool_input!s} (non-serializable)\n")
                     elif isinstance(item, ToolOutputItem):
                         f.write(f"Tool Name: {item.tool_name}\n")
                         f.write(f"Output: {item.tool_output!s}\n") # Simple string representation
                     elif isinstance(item, MessageOutputItem):
                         f.write(f"Content: {ItemHelpers.text_message_output(item)}\n")
                     f.write("-" * 10 + "\n")
        except Exception as e:
            logger.error(f"Failed to save conversation step {step_num} to {filepath}: {e}")


    # --- GIF Generation (Identical to previous implementation) ---
    def create_history_gif(
        self, output_path: str = 'agent_history_sdk.gif', duration: int = 3000,
        show_goals: bool = True, show_task: bool = True, show_logo: bool = False,
        font_size: int = 40, title_font_size: int = 56, goal_font_size: int = 44,
        margin: int = 40, line_spacing: float = 1.5,
    ) -> None:
        if not self.full_history.history: logger.warning('No history to create GIF from'); return
        images = []
        first_screenshot = self.full_history.screenshots()[0] if self.full_history.screenshots() else None
        if not first_screenshot: logger.warning('No screenshots found in history to create GIF from'); return

        try:
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana', 'Calibri', 'Tahoma'] # Added more options
            font_loaded = False
            regular_font, title_font, goal_font = None, None, None
            for font_name in font_options:
                try:
                    font_path = self._find_font_path(font_name)
                    if font_path:
                        regular_font = ImageFont.truetype(font_path, font_size)
                        title_font = ImageFont.truetype(font_path, title_font_size)
                        goal_font = ImageFont.truetype(font_path, goal_font_size)
                        font_loaded = True
                        logger.debug(f"Loaded font: {font_path}")
                        break
                except OSError: logger.debug(f"Font '{font_name}' not found or failed to load at path '{font_path}'.")
                except Exception as e: logger.debug(f"Error loading font '{font_name}': {e}")

            if not font_loaded:
                logger.warning('No preferred fonts found, using default.')
                regular_font, title_font = ImageFont.load_default(), ImageFont.load_default()
                goal_font = regular_font
        except Exception as font_e:
             logger.error(f"Error loading fonts: {font_e}", exc_info=True)
             regular_font, title_font = ImageFont.load_default(), ImageFont.load_default()
             goal_font = regular_font

        logo = None
        if show_logo:
            try:
                logo_path = Path(__file__).parent.parent / 'static/browser-use.png' # Adjust path if needed
                if logo_path.exists():
                    logo = Image.open(logo_path); logo_height = 150; aspect_ratio = logo.width / logo.height
                    logo_width = int(logo_height * aspect_ratio); logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                else: logger.warning(f"Logo file not found at {logo_path}")
            except Exception as e: logger.warning(f'Could not load logo: {e}')

        if show_task and self.task:
            try:
                task_frame = self._create_task_frame(self.task, first_screenshot, title_font, regular_font, logo, line_spacing)
                images.append(task_frame)
            except Exception as e: logger.error(f"Failed to create task frame: {e}", exc_info=True)

        for i, item in enumerate(self.full_history.history, 1):
            if not item.browser_state or not item.browser_state.screenshot: continue
            try:
                img_data = base64.b64decode(item.browser_state.screenshot); image = Image.open(io.BytesIO(img_data))
                goal_text = "No goal recorded"
                if item.agent_state_update: goal_text = item.agent_state_update.next_goal
                elif item.plan and isinstance(item.plan, PlannerOutput): goal_text = "; ".join(item.plan.next_steps) # Use planner steps if no agent goal

                image = self._add_overlay_to_image(
                    image=image, step_number=i, goal_text=goal_text if show_goals else "",
                    regular_font=regular_font, title_font=title_font, goal_font=goal_font,
                    margin=margin, logo=logo, display_step=True, display_goal=show_goals,
                )
                images.append(image)
            except Exception as e: logger.error(f"Failed to process history item {i} for GIF: {e}", exc_info=True)

        if images:
            try:
                images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
                logger.info(f'Created GIF at {output_path}')
            except Exception as e: logger.error(f"Failed to save GIF to {output_path}: {e}", exc_info=True)
        else: logger.warning('No valid images generated from history to create GIF')

    def _find_font_path(self, font_name: str) -> Optional[str]:
         """Tries to find a path for a given font name based on OS."""
         system = platform.system()
         font_name_ttf = font_name + ".ttf"
         font_name_otf = font_name + ".otf" # Also check otf

         if system == "Windows":
             font_dir = os.getenv('WINDIR', 'C:\\Windows') + '\\Fonts'
             paths_to_check = [os.path.join(font_dir, font_name_ttf), os.path.join(font_dir, font_name_otf)]
         elif system == "Linux":
             # Common Linux font directories
             dirs = ["/usr/share/fonts/truetype", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
             paths_to_check = []
             for d in dirs:
                 # Simple check, could use find command for more robust search
                 if os.path.isdir(d):
                     for root, _, files in os.walk(d):
                         if font_name_ttf in files: paths_to_check.append(os.path.join(root, font_name_ttf))
                         if font_name_otf in files: paths_to_check.append(os.path.join(root, font_name_otf))
             # Add common package paths directly
             paths_to_check.extend([
                 f"/usr/share/fonts/truetype/msttcorefonts/{font_name}.ttf", # Example for corefonts
                 f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Example for DejaVu
             ])
         elif system == "Darwin": # macOS
             paths_to_check = [
                 f"/Library/Fonts/{font_name_ttf}", f"/Library/Fonts/{font_name_otf}",
                 f"/System/Library/Fonts/Supplemental/{font_name_ttf}", f"/System/Library/Fonts/Supplemental/{font_name_otf}", # Big Sur+
                 os.path.expanduser(f"~/Library/Fonts/{font_name_ttf}"), os.path.expanduser(f"~/Library/Fonts/{font_name_otf}")
             ]
         else: # Unsupported OS
             return None

         for path in paths_to_check:
             if os.path.exists(path):
                 return path
         return None # Font not found in common locations


    def _create_task_frame(self, task: str, first_screenshot: str, title_font, regular_font, logo, line_spacing) -> Image.Image:
        # (Identical to previous implementation)
        img_data = base64.b64decode(first_screenshot); template = Image.open(io.BytesIO(img_data))
        image = Image.new('RGB', template.size, (0, 0, 0)); draw = ImageDraw.Draw(image)
        center_y = image.height // 2; margin = 140; max_width = image.width - (2 * margin)
        larger_font = regular_font
        if hasattr(regular_font, 'path') and hasattr(regular_font, 'size'):
             try: larger_font = ImageFont.truetype(regular_font.path, regular_font.size + 16)
             except Exception: pass
        wrapped_text = self._wrap_text(task, larger_font, max_width)
        line_height = (larger_font.size if hasattr(larger_font, 'size') else 20) * line_spacing
        lines = wrapped_text.split('\n'); total_height = line_height * len(lines)
        text_y = center_y - (total_height / 2) + 50
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=larger_font)
            text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2
            draw.text((text_x, text_y), line, font=larger_font, fill=(255, 255, 255))
            text_y += line_height
        if logo:
            logo_margin = 20; logo_x = image.width - logo.width - logo_margin
            image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
        return image

    def _add_overlay_to_image(
        self, image: Image.Image, step_number: int, goal_text: str, regular_font, title_font, goal_font,
        margin: int, logo: Optional[Image.Image] = None, display_step: bool = True, display_goal: bool = True,
        text_color: tuple = (255, 255, 255, 255), text_box_color: tuple = (0, 0, 0, 200),
    ) -> Image.Image:
        # (Identical to previous implementation)
        image = image.convert('RGBA'); txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(txt_layer)
        y_step = 0; padding = 20
        if display_step:
            step_text = str(step_number); step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
            step_width = step_bbox[2] - step_bbox[0]; step_height = step_bbox[3] - step_bbox[1]
            x_step = margin + 10; y_step = image.height - margin - step_height - 10
            step_bg_bbox = (x_step - padding, y_step - padding, x_step + step_width + padding, y_step + step_height + padding)
            draw.rounded_rectangle(step_bg_bbox, radius=15, fill=text_box_color)
            draw.text((x_step, y_step), step_text, font=title_font, fill=text_color)
        if display_goal and goal_text:
            max_width = image.width - (4 * margin); wrapped_goal = self._wrap_text(goal_text, goal_font, max_width)
            goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=goal_font)
            goal_width = goal_bbox[2] - goal_bbox[0]; goal_height = goal_bbox[3] - goal_bbox[1]
            x_goal = (image.width - goal_width) // 2
            y_goal_base = (y_step - padding * 4) if display_step else (image.height - margin - 10)
            y_goal = y_goal_base - goal_height; padding_goal = 25
            goal_bg_bbox = (x_goal - padding_goal, y_goal - padding_goal, x_goal + goal_width + padding_goal, y_goal + goal_height + padding_goal)
            draw.rounded_rectangle(goal_bg_bbox, radius=15, fill=text_box_color)
            draw.multiline_text((x_goal, y_goal), wrapped_goal, font=goal_font, fill=text_color, align='center')
        if logo:
            logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0)); logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
            image = Image.alpha_composite(image, logo_layer) # Composite logo first
        result = Image.alpha_composite(image, txt_layer)
        return result.convert('RGB')

    def _wrap_text(self, text: str, font, max_width: int) -> str:
        # (Identical to previous implementation)
        if not hasattr(font, 'getbbox'):
             import textwrap; avg_char_width = 10; wrap_width = max(10, int(max_width / avg_char_width))
             return textwrap.fill(text, width=wrap_width)
        words = text.split(); lines = []; current_line = []
        for word in words:
            current_line.append(word); line = ' '.join(current_line)
            try: bbox = font.getbbox(line); line_width = bbox[2] - bbox[0]
            except AttributeError: line_width = len(line) * 10
            if line_width > max_width:
                if len(current_line) == 1:
                    split_point = max(1, int(len(word) * max_width / line_width) -1)
                    lines.append(word[:split_point] + '-'); current_line = [word[split_point:]]
                else:
                    current_line.pop(); lines.append(' '.join(current_line)); current_line = [word]
        if current_line: lines.append(' '.join(current_line))
        return '\n'.join(lines)

    # --- Control Methods ---
    def pause(self) -> None: logger.info('🔄 Pausing Agent Runner'); self._paused = True
    def resume(self) -> None: logger.info('▶️ Resuming Agent Runner'); self._paused = False
    def stop(self) -> None: logger.info('⏹️ Stopping Agent Runner'); self._stopped = True

