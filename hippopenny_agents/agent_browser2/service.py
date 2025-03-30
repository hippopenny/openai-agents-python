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
# Removed imports from langchain_core, pydantic (BaseModel used directly below)
from pydantic import BaseModel, ValidationError # Keep pydantic for internal models if needed
from PIL import Image, ImageDraw, ImageFont

# Removed imports from agents SDK (Agent, Runner, trace, ItemHelpers, etc.)
# Removed imports from browser_use (Browser, Controller, Telemetry, etc.)

# Imports from this refactored package
from .context import BrowserAgentContext # Keep context
from .prompts import AgentMessagePrompt, PlannerPromptBuilder # Keep prompts
# from .tools import BROWSER_TOOLS_SIGNATURES # Keep tool signatures if needed
from .views import ActionResult, AgentHistory, AgentHistoryList, AgentStateUpdate, PlannerOutput # Keep views

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BrowserAgentRunner:
    """
    Orchestrates the execution of the browser agent.
    NOTE: Dependencies on agents SDK, langchain, and browser_use have been removed.
    The core execution logic (`run` method) has been removed and needs reimplementation.
    Kept __init__ structure, GIF generation, and control methods as placeholders.
    """
    def __init__(
        self,
        task: str,
        llm: Any, # Changed BaseChatModel to Any
        # Removed browser, browser_context, controller parameters
        use_vision: bool = True,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        generate_gif: bool | str = True,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        include_attributes: list[str] = [
            'title', 'type', 'name', 'role', 'tabindex', 'aria-label',
            'placeholder', 'value', 'alt', 'aria-expanded',
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None, # Structure kept, execution removed
        page_extraction_llm: Optional[Any] = None, # Changed BaseChatModel to Any
        planner_llm: Optional[Any] = None, # Changed BaseChatModel to Any
        planning_interval: int = 1,
        use_vision_for_planner: bool = False,
        # Add placeholders for browser/controller interfaces if needed
        browser_interface: Any | None = None,
        controller_interface: Any | None = None,
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
        self.initial_actions = initial_actions # Store definition, execution logic removed
        self.page_extraction_llm = page_extraction_llm or llm

        # Planner setup
        self.planner_llm = planner_llm
        self.planning_interval = planning_interval if planner_llm else 0
        self.use_vision_for_planner = use_vision_for_planner if planner_llm else False

        # Telemetry setup - Placeholder, original ProductTelemetry removed
        self.telemetry = None # Replace with actual telemetry implementation if needed
        self._set_version_and_source()
        self._set_model_names()

        # Browser/Controller setup - Placeholders
        self.browser_interface = browser_interface # Needs implementation
        self.controller_interface = controller_interface # Needs implementation
        logger.warning("Browser and Controller interfaces are placeholders and require implementation.")

        # Agent setup - Placeholder
        # self.agent = StandaloneBrowserAgent(...) # Instantiate custom agent here if defined
        self.agent = None # Placeholder
        logger.warning("Agent definition and execution logic require implementation.")


        # Initialize Agent Context (structure remains, usage depends on reimplementation)
        self.agent_context = BrowserAgentContext(
            task=self.task,
            # Removed browser_context, controller dependencies
            browser_interface=self.browser_interface, # Pass placeholder interfaces
            controller_interface=self.controller_interface,
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
        )

        # History tracking
        self.full_history = AgentHistoryList(history=[])
        self.current_conversation_messages: list[Dict] = [] # Store simple dict messages

        # Control flags
        self._paused = False
        self._stopped = False

        if save_conversation_path:
            logger.info(f'Saving conversation steps to {save_conversation_path}_<step>.txt')


    def _set_version_and_source(self) -> None:
        # (Same as original)
        try:
            # Try pkg_resources first
            import pkg_resources
            self.version = pkg_resources.get_distribution('hippopenny_agents').version # Adjust package name if needed
            self.source = 'pip'
        except Exception:
            try:
                # Fallback to git describe
                import subprocess
                self.version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
                self.source = 'git'
            except Exception:
                self.version = 'unknown'
                self.source = 'unknown'
        logger.debug(f'Version: {self.version}, Source: {self.source}')

    def _set_model_names(self) -> None:
         # Attempt to get model names if possible, otherwise 'Unknown'
         # This depends heavily on the structure of the passed LLM objects (now Any)
        self.model_name = 'Unknown'
        self.planner_model_name = None
        try:
            if hasattr(self.llm, 'model_name'): self.model_name = self.llm.model_name
            elif hasattr(self.llm, 'model'): self.model_name = self.llm.model
        except Exception: pass # Ignore errors if attributes don't exist

        if self.planner_llm:
            try:
                if hasattr(self.planner_llm, 'model_name'): self.planner_model_name = self.planner_llm.model_name
                elif hasattr(self.planner_llm, 'model'): self.planner_model_name = self.planner_llm.model
                else: self.planner_model_name = 'Unknown (Planner)'
            except Exception:
                 self.planner_model_name = 'Unknown (Planner)'
        logger.debug(f"Main Model: {self.model_name}, Planner Model: {self.planner_model_name or 'N/A'}")


    def _log_agent_run_start(self) -> None:
        """Log the agent run start"""
        logger.info(f"🚀 Starting task: {self.task} (Agent ID: {self.agent_id})")
        # Add telemetry call here if implemented

    def _log_agent_run_end(self, steps_taken: int, max_steps: int) -> None:
        """Log the agent run end"""
        is_done = self.full_history.is_done()
        max_steps_reached = steps_taken >= max_steps and not is_done
        logger.info(f"🏁 Agent run finished. Success: {is_done}, Steps: {steps_taken}, Max steps reached: {max_steps_reached}")
         # Add telemetry call here if implemented

    async def _execute_initial_actions(self) -> List[ActionResult]:
        """Placeholder for executing initial actions."""
        if not self.initial_actions:
            return []
        logger.info("Executing initial actions (Placeholder - No execution)")
        # Requires reimplementation using self.controller_interface
        # For now, return an empty success result
        return [ActionResult()] # Placeholder result

    # Removed _run_planner method as it depended on LangChain messages and LLM structure

    # Removed run method as it depended heavily on agents.Runner

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
         """Placeholder for the main execution loop."""
         self._log_agent_run_start()
         logger.error("BrowserAgentRunner.run() requires reimplementation without agents SDK.")
         # Basic structure:
         # 1. Execute initial actions (placeholder)
         # await self._execute_initial_actions()
         # 2. Loop for max_steps:
         #    a. Check control flags
         #    b. Run planner (placeholder/reimplement)
         #    c. Get browser state (placeholder/reimplement)
         #    d. Prepare input message (using AgentMessagePrompt)
         #    e. Call LLM (using self.llm)
         #    f. Parse LLM response (expecting tool calls)
         #    g. Execute tools (placeholder/reimplement using controller_interface)
         #    h. Update history (using AgentHistory/AgentHistoryList)
         #    i. Check if done
         # 3. Cleanup (placeholder)
         # await self._cleanup_resources()
         # 4. Log end
         self._log_agent_run_end(0, max_steps) # Log end with 0 steps
         # 5. Generate GIF (kept)
         if self.generate_gif:
             output_path: str = 'agent_history_sdk.gif'
             if isinstance(self.generate_gif, str): output_path = self.generate_gif
             try: self.create_history_gif(output_path=output_path)
             except Exception as gif_e: logger.error(f"Failed to generate GIF: {gif_e}", exc_info=True)

         return self.full_history # Return potentially empty history


    async def _cleanup_resources(self):
        """Placeholder for cleaning up resources like browser connections."""
        logger.debug("Cleaning up resources (Placeholder)...")
        # Add logic here to close browser connection via self.browser_interface if needed

    # Removed _save_conversation_step as it relied on specific input/output item structures

    # --- GIF Generation (Kept, relies on AgentHistoryList structure and PIL) ---
    # Assumes AgentHistoryList structure provides necessary data (screenshots, goals/plans)
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
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana', 'Calibri', 'Tahoma']
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
                # Assuming static assets might be relative to this file's location
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
            # Use placeholder BrowserStateHistory
            if not item.browser_state or not item.browser_state.screenshot: continue
            try:
                img_data = base64.b64decode(item.browser_state.screenshot); image = Image.open(io.BytesIO(img_data))
                goal_text = "No goal/plan recorded" # Default text
                if item.agent_state_update: goal_text = item.agent_state_update.next_goal
                elif item.plan and isinstance(item.plan, PlannerOutput): goal_text = "; ".join(item.plan.next_steps)
                elif item.plan and isinstance(item.plan, str): goal_text = item.plan # Show raw plan string if not parsed

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
         # (Identical to previous implementation)
         system = platform.system()
         font_name_ttf = font_name + ".ttf"
         font_name_otf = font_name + ".otf" # Also check otf

         if system == "Windows":
             font_dir = os.getenv('WINDIR', 'C:\\Windows') + '\\Fonts'
             paths_to_check = [os.path.join(font_dir, font_name_ttf), os.path.join(font_dir, font_name_otf)]
         elif system == "Linux":
             dirs = ["/usr/share/fonts/truetype", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
             paths_to_check = []
             for d in dirs:
                 if os.path.isdir(d):
                     for root, _, files in os.walk(d):
                         if font_name_ttf in files: paths_to_check.append(os.path.join(root, font_name_ttf))
                         if font_name_otf in files: paths_to_check.append(os.path.join(root, font_name_otf))
             paths_to_check.extend([
                 f"/usr/share/fonts/truetype/msttcorefonts/{font_name}.ttf",
                 f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             ])
         elif system == "Darwin": # macOS
             paths_to_check = [
                 f"/Library/Fonts/{font_name_ttf}", f"/Library/Fonts/{font_name_otf}",
                 f"/System/Library/Fonts/Supplemental/{font_name_ttf}", f"/System/Library/Fonts/Supplemental/{font_name_otf}",
                 os.path.expanduser(f"~/Library/Fonts/{font_name_ttf}"), os.path.expanduser(f"~/Library/Fonts/{font_name_otf}")
             ]
         else: return None

         for path in paths_to_check:
             if os.path.exists(path): return path
         return None


    def _create_task_frame(self, task: str, first_screenshot: str, title_font, regular_font, logo, line_spacing) -> Image.Image:
        # (Identical to previous implementation)
        img_data = base64.b64decode(first_screenshot); template = Image.open(io.BytesIO(img_data))
        image = Image.new('RGB', template.size, (0, 0, 0)); draw = ImageDraw.Draw(image)
        center_y = image.height // 2; margin = 140; max_width = image.width - (2 * margin)
        larger_font = regular_font
        if hasattr(regular_font, 'path') and hasattr(regular_font, 'size'):
             try: larger_font = ImageFont.truetype(regular_font.path, regular_font.size + 16)
             except Exception: pass # Use regular_font if increasing size fails
        wrapped_text = self._wrap_text(task, larger_font, max_width)
        line_height = (larger_font.size if hasattr(larger_font, 'size') else 20) * line_spacing
        lines = wrapped_text.split('\n'); total_height = line_height * len(lines)
        text_y = center_y - (total_height / 2) + 50
        for line in lines:
            try:
                 # Use textbbox for better centering
                 line_bbox = draw.textbbox((0, 0), line, font=larger_font)
                 text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2
            except AttributeError: # Fallback for older PIL/Pillow or default font
                 line_width, _ = draw.textsize(line, font=larger_font)
                 text_x = (image.width - line_width) // 2

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
        # (Identical to previous implementation, check font methods)
        image = image.convert('RGBA'); txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(txt_layer)
        y_step = 0; padding = 20
        if display_step:
            step_text = str(step_number)
            try: step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
            except AttributeError: step_bbox = (0,0) + draw.textsize(step_text, font=title_font) # Fallback
            step_width = step_bbox[2] - step_bbox[0]; step_height = step_bbox[3] - step_bbox[1]
            x_step = margin + 10; y_step = image.height - margin - step_height - 10
            step_bg_bbox = (x_step - padding, y_step - padding, x_step + step_width + padding, y_step + step_height + padding)
            draw.rounded_rectangle(step_bg_bbox, radius=15, fill=text_box_color)
            draw.text((x_step, y_step), step_text, font=title_font, fill=text_color)
        if display_goal and goal_text:
            max_width = image.width - (4 * margin); wrapped_goal = self._wrap_text(goal_text, goal_font, max_width)
            try: goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=goal_font)
            except AttributeError: goal_bbox = (0,0) + draw.multiline_textsize(wrapped_goal, font=goal_font) # Fallback
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
        # (Identical to previous implementation, check font methods)
        if not hasattr(font, 'getbbox') or not hasattr(font, 'getlength'): # Check for methods needed
             import textwrap; avg_char_width = 10; wrap_width = max(10, int(max_width / avg_char_width))
             logger.debug("Using textwrap for text wrapping due to missing font methods.")
             return textwrap.fill(text, width=wrap_width)

        words = text.split(); lines = []; current_line = []
        for word in words:
            current_line.append(word); line = ' '.join(current_line)
            try:
                 # Use getlength if available (more efficient for width check)
                 line_width = font.getlength(line)
            except AttributeError:
                 # Fallback to getbbox if getlength is not available
                 bbox = font.getbbox(line); line_width = bbox[2] - bbox[0]

            if line_width > max_width:
                if len(current_line) == 1: # Word itself is too long
                    # Basic split, might break words awkwardly
                    split_point = max(1, int(len(word) * max_width / line_width) -1)
                    lines.append(word[:split_point] + '-')
                    current_line = [word[split_point:]]
                    logger.debug(f"Splitting long word: {word}")
                else:
                    current_line.pop() # Remove the word that made it too long
                    lines.append(' '.join(current_line))
                    current_line = [word] # Start new line with the current word
        if current_line: lines.append(' '.join(current_line))
        return '\n'.join(lines)

    # --- Control Methods ---
    def pause(self) -> None: logger.info('🔄 Pausing Agent Runner'); self._paused = True
    def resume(self) -> None: logger.info('▶️ Resuming Agent Runner'); self._paused = False
    def stop(self) -> None: logger.info('⏹️ Stopping Agent Runner'); self._stopped = True

