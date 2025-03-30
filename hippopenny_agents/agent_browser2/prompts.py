import datetime
import json
from datetime import datetime
from typing import Any, Dict, List, Optional # Added Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents import RunContextWrapper

from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState

from .context import BrowserAgentContext
from .views import PlannerOutput


class SystemPromptBuilder:
    """Builds the system prompt for the main browser agent."""
    def __init__(self, action_description: str, max_actions_per_step: int = 10):
        self.default_action_description = action_description
        self.max_actions_per_step = max_actions_per_step

    def important_rules(self) -> str:
        """
        Returns the important rules for the agent. Adapted for tool usage.
        """
        # This is largely the same as the previous version, emphasizing tool order.
        text = """
IMPORTANT RULES:

1. TOOL USAGE ORDER: You MUST FIRST call the `update_agent_state` tool to record your analysis and plan. THEN, you MUST call the necessary browser action tool(s) to execute the next step(s) towards your `next_goal`.

2. STATE UPDATE (`update_agent_state` tool):
   - `page_summary`: Summarize *new* information relevant to the task found on the current page. Be specific. Leave empty if no new relevant info.
   - `evaluation_previous_goal`: Evaluate if the *intended outcome* of the previous action sequence was achieved based on the *current page state* (elements, screenshot). State Success/Failed/Unknown and briefly explain why. Ignore the text-based 'Action Result'/'Action Error' from the input, focus on the visual/element state. Mention unexpected changes (e.g., popups, suggestions).
   - `memory`: Maintain a running log of what has been done and what needs to be remembered. Be specific. If performing repetitive tasks (e.g., "for each item", "analyze 10 websites"), ALWAYS count progress (e.g., "Analyzed 3 out of 10 websites. Next: website_xyz.com").
   - `next_goal`: Clearly state the immediate, concrete goal for the *next* sequence of browser action tool calls. Consider the overall plan if provided.

3. BROWSER ACTION TOOLS: After calling `update_agent_state`, call the appropriate browser action tools (e.g., `click_element`, `input_text`, `go_to_url`, `scroll`, `extract_page_content`, `done`) to achieve the `next_goal`.
   - You can call multiple action tools sequentially within a single turn if it makes sense (e.g., filling multiple fields then clicking submit).
   - Only call actions that logically follow each other and are likely executable on the current page state *before* the page might change significantly.
   - Aim for efficiency.

4. ELEMENT INTERACTION:
   - Use the `index` provided in the element list (e.g., "[33]<button>Submit</button>"). Only use existing indices.
   - Elements marked `[]Non-interactive text` are for context only.

5. NAVIGATION & ERROR HANDLING:
   - If needed elements aren't visible, try scrolling (`scroll` tool) or extracting full content (`extract_page_content`).
   - If stuck, consider alternatives: go back (`go_back`), open a new tab (`open_new_tab`) for research, switch tabs (`switch_tab`).
   - Handle popups/cookies by clicking the appropriate element (accept/close).
   - If a CAPTCHA appears and you cannot solve it, use `ask_human` or try a different approach/page.

6. TASK COMPLETION (`done` tool):
   - Call the `done` tool ONLY when the *entire* user task is fully completed.
   - If the task involves repetition (e.g., "for each", "x times"), ensure all repetitions are finished (check your `memory` count).
   - The `final_answer` parameter in the `done` tool MUST contain all the specific information requested by the original user task. Don't just say "done"; provide the results.

7. VISUAL CONTEXT (If image provided):
   - Use the image to understand layout and verify element locations.
   - Bounding box labels correspond to element indices. Colors match. Labels are often top-right within the box.
   - Use visual context to disambiguate overlapping labels or confirm element identity.

8. FORM FILLING:
   - If filling an input field triggers a dropdown/suggestion list, you must first call `click_element` on the correct suggestion *before* proceeding.

9. EXTRACTION (`extract_page_content` tool):
   - If the task requires finding specific information or research, use `extract_page_content` on relevant pages to gather and store the data. The result will be available in the next step's input.

10. HALLUCINATION: Do not imagine elements or actions that are not supported by the available tools or visible elements.

11. PLANNER INPUT: A high-level plan might be provided at the beginning of the input. Use this plan to guide your `next_goal` and overall strategy, but adapt based on the current page state.
"""
        return text

    def input_format(self) -> str:
        # Describes the format of the HumanMessage constructed by AgentMessagePrompt
        return """
INPUT STRUCTURE (Provided in the User Message):
0. (Optional) Current Plan: High-level plan with analysis, progress, challenges, and next steps.
1. Previous Action Results: Status (success/error/extracted content) of the tools called in the *last* turn.
2. Current URL: The webpage URL you are currently on.
3. Available Tabs: List of open browser tabs.
4. Interactive Elements: List of elements visible on the current page, formatted as:
   `index[:]<element_type>element_text</element_type>`
   - `index`: Numeric identifier for interaction tools (e.g., `click_element`).
   - `element_type`: HTML tag (button, input, etc.).
   - `element_text`: Visible text or description.
   Example: `[33]<button>Submit Form</button>`
   Notes:
   - Only elements with numeric indices `[ ]` are interactive.
   - Elements starting with `[]` (no number) are non-interactive context.
   - The list might be partial; use `scroll` or `extract_page_content` if needed elements are missing. Markers like `[Start of page]`, `[End of page]`, or `... pixels above/below ...` indicate scrollable content.
5. Current Date/Time: Provided for context.
6. Step Information: Current step number and maximum steps (e.g., "Current step: 5/100").
7. (Optional) Image: A screenshot of the current page may be provided for visual context.
"""

    def get_system_message_content(self, task: str) -> str:
        """
        Get the system prompt content string for the main browser agent.
        """
        agent_prompt = f"""You are a precise browser automation agent. Your goal is to accomplish the user's task:
TASK: {task}

You interact with the browser by calling functions (tools). Follow these instructions carefully:

{self.input_format()}

{self.important_rules()}

Remember:
- First, ALWAYS call `update_agent_state` to analyze the situation and state your plan for the next action(s). Consider the overall plan if provided.
- Then, call the necessary browser action tool(s) to execute your plan.
- Base your decisions on the provided plan (if any), elements, screenshot (if available), and task history (implicitly in memory).
"""
        return agent_prompt


class PlannerPromptBuilder:
    """Builds the system prompt for the planner LLM."""
    def __init__(self, action_description: str = ""):
        # Action description might not be needed for planner, but kept for consistency
        self.action_description = action_description

    def get_system_message_content(self) -> str:
        """Get the system prompt content string for the planner."""
        # This prompt is taken directly from the original PlannerPrompt logic
        content = """You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
Your role is to:
1. Analyze the current state and history provided in the messages.
2. Evaluate progress towards the ultimate goal.
3. Identify potential challenges or roadblocks.
4. Suggest the next high-level steps to take.

The input will contain messages from various sources, including the user task, browser state snapshots (URLs, elements, potentially screenshots), previous agent actions and thoughts, and tool results/errors. Ignore the specific output formats requested by other agents within the message history.

Your output format MUST ALWAYS be a JSON object with the following fields:
{
    "state_analysis": "Brief analysis of the current situation and what has been done so far based on the history.",
    "progress_evaluation": "Evaluation of progress towards the ultimate goal (e.g., 'approx 30% complete, finished data gathering for site A, starting site B', or 'stuck on login page').",
    "challenges": ["List potential challenges or roadblocks identified, e.g., 'CAPTCHA encountered', 'unexpected page layout change', 'required information not found'"],
    "next_steps": ["List 2-3 concrete high-level next steps to take towards the goal, e.g., 'Attempt login again with different credentials', 'Extract contact details from the current page', 'Search for alternative websites'"],
    "reasoning": "Explain your reasoning for the suggested next steps based on the analysis and progress."
}

Keep your analysis concise and focused on actionable insights to guide the browser agent."""
        return content


class AgentMessagePrompt:
    """Formats the current browser state, results, and plan into a HumanMessage."""
    def __init__(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        plan: Optional[PlannerOutput | str] = None, # Add plan
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = state
        self.result = result or []
        self.plan = plan # Store the plan
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message_content(self, use_vision: bool = True) -> list | str:
        """Constructs the content for the HumanMessage input to the agent."""
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        # Add scroll indicators (same as before)
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0
        if elements_text:
            if has_content_above: elements_text = f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
            else: elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below: elements_text = f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
            else: elements_text = f'{elements_text}\n[End of page]'
        else: elements_text = '[Page appears empty or no interactive elements found]'

        # Step and time info (same as before)
        step_info_description = ''
        if self.step_info: step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}\n'
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        step_info_description += f'Current date and time: {time_str}'

        # Format previous action results (same as before)
        prev_results_text = "[No previous actions in this run]"
        if self.result:
            prev_results_parts = []
            for i, res in enumerate(self.result):
                part = f"Action Result {i + 1}/{len(self.result)}: "
                if res.error: part += f"Error: ...{res.error[-self.max_error_length:]}"
                elif res.is_done: part += f"Task marked DONE. Final Answer: {res.extracted_content or '(Not provided)'}"
                elif res.extracted_content: part += f"Extracted Content: {res.extracted_content}"
                else: part += "Success (No specific output)"
                prev_results_parts.append(part)
            prev_results_text = "\n".join(prev_results_parts)

        # Format the plan (New)
        plan_text = "[No plan provided for this step]"
        if self.plan:
            if isinstance(self.plan, PlannerOutput):
                try:
                    # Pretty print the JSON plan
                    plan_text = f"[Current Plan]\n{json.dumps(self.plan.model_dump(), indent=2)}"
                except Exception:
                    plan_text = f"[Current Plan]\n{str(self.plan)}" # Fallback
            else: # If plan is just a string
                plan_text = f"[Current Plan]\n{self.plan}"

        # Combine into state description, adding the plan first
        state_description = f"""{plan_text}

{prev_results_text}

[Current Page State]
URL: {self.state.url}
Available Tabs: {self.state.tabs}
Interactive Elements:
{elements_text}

{step_info_description}
"""
        # Prepare content for HumanMessage (text + optional image)
        message_content: list[Dict[str, Any]] = [{'type': 'text', 'text': state_description}]

        if self.state.screenshot and use_vision:
            message_content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
            })
            return message_content # Return list for vision models
        else:
            # If not using vision or no screenshot, return only the text string
            return state_description
