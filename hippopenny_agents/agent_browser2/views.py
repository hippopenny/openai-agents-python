from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.dom.history_tree_processor.service import (
    DOMElementNode,
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from browser_use.dom.views import SelectorMap


# --- Tool Input Models ---
# These models define the parameters for each browser action tool.
# They mirror the structure previously expected within the 'action' list.

class GoToURLParams(BaseModel):
    url: str = Field(..., description='The URL to navigate to.')

class ClickParams(BaseModel):
    index: int = Field(..., description='The index of the element to click.')

class InputTextParams(BaseModel):
    index: int = Field(..., description='The index of the input element.')
    text: str = Field(..., description='The text to input.')
    clear_before_input: bool = Field(default=True, description='Whether to clear the field before inputting text.')

class ScrollParams(BaseModel):
    direction: str = Field(..., description="Direction to scroll ('up' or 'down').")
    pixels: Optional[int] = Field(default=None, description='Number of pixels to scroll (optional).')

class SelectOptionParams(BaseModel):
    index: int = Field(..., description='The index of the select element.')
    option_index: int = Field(..., description='The index of the option to select.')

class CheckboxParams(BaseModel):
    index: int = Field(..., description='The index of the checkbox element.')
    checked: bool = Field(..., description='Whether the checkbox should be checked or unchecked.')

class ExtractContentParams(BaseModel):
    query: Optional[str] = Field(default=None, description='Optional query to guide content extraction.')

class OpenNewTabParams(BaseModel):
    pass # No parameters needed

class CloseTabParams(BaseModel):
    pass # No parameters needed

class SwitchTabParams(BaseModel):
    tab_index: int = Field(..., description='The index of the tab to switch to.')

class GoBackParams(BaseModel):
    pass # No parameters needed

class UploadFileParams(BaseModel):
    index: int = Field(..., description='The index of the file input element.')
    file_path: str = Field(..., description='The path to the file to upload.')

class AskHumanParams(BaseModel):
    question: str = Field(..., description='The question to ask the human.')

class DoneParams(BaseModel):
    final_answer: str = Field(..., description='The final answer or result of the task.')


# --- Agent State Update Model ---

class AgentStateUpdate(BaseModel):
    """Model for the LLM to update its internal state analysis via the update_agent_state tool."""
    page_summary: str = Field(description="Quick detailed summary of new information from the current page which is not yet in the task history memory. Be specific with details which are important for the task. This is not on the meta level, but should be facts. If all the information is already in the task history memory, leave this empty.")
    evaluation_previous_goal: str = Field(description="Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not")
    memory: str = Field(description="Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz")
    next_goal: str = Field(description="What needs to be done with the next actions")


# --- Planner Output Model ---

class PlannerOutput(BaseModel):
    """Model for the structured output expected from the Planner LLM."""
    state_analysis: str = Field(description="Brief analysis of the current state and what has been done so far based on history.")
    progress_evaluation: str = Field(description="Evaluation of progress towards the ultimate goal (e.g., percentage, description).")
    challenges: List[str] = Field(description="List any potential challenges or roadblocks identified.")
    next_steps: List[str] = Field(description="List 2-3 concrete high-level next steps to take towards the goal.")
    reasoning: str = Field(description="Explain the reasoning for the suggested next steps.")


# --- Action Result ---
# Reusing the original ActionResult for simplicity.
@dataclass
class ActionResult:
    """Result of executing an action tool."""
    is_done: Optional[bool] = False
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False  # whether to include in past messages as context or not


# --- History Tracking ---

class AgentHistory(BaseModel):
    """History item for a single step of the agent's execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # Store the state update and the sequence of tool calls/results for this step
    agent_state_update: Optional[AgentStateUpdate] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list) # Store tool call details {tool_name: params}
    tool_results: List[ActionResult] = Field(default_factory=list) # Store results corresponding to tool_calls
    browser_state: BrowserStateHistory # The browser state *before* the actions were taken
    plan: Optional[PlannerOutput | str] = None # Store the plan generated during this step (if any)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization handling"""
        plan_dump = None
        if isinstance(self.plan, PlannerOutput):
            plan_dump = self.plan.model_dump()
        elif isinstance(self.plan, str):
            plan_dump = self.plan # Store raw string if not parsed

        return {
            'agent_state_update': self.agent_state_update.model_dump() if self.agent_state_update else None,
            'tool_calls': self.tool_calls, # Already dicts
            'tool_results': [r.__dict__ for r in self.tool_results], # Simple dict conversion for dataclass
            'browser_state': self.browser_state.to_dict(),
            'plan': plan_dump,
        }


class AgentHistoryList(BaseModel):
    """List of agent history items, representing the full run."""
    history: list[AgentHistory] = Field(default_factory=list)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save history to JSON file with proper serialization"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise e

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization that properly uses AgentHistory's model_dump"""
        return {
            'history': [h.model_dump(**kwargs) for h in self.history],
        }

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> 'AgentHistoryList':
        """Load history from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Basic validation/parsing during load
        history_items = []
        for h_data in data.get('history', []):
            state_update = AgentStateUpdate.model_validate(h_data['agent_state_update']) if h_data.get('agent_state_update') else None
            browser_state_hist = BrowserStateHistory(**h_data['browser_state'])
            tool_results = [ActionResult(**res_data) for res_data in h_data.get('tool_results', [])]

            # Load plan
            plan_data = h_data.get('plan')
            plan = None
            if isinstance(plan_data, dict):
                try:
                    plan = PlannerOutput.model_validate(plan_data)
                except Exception: # Fallback to string if validation fails
                    plan = json.dumps(plan_data)
            elif isinstance(plan_data, str):
                plan = plan_data

            history_items.append(AgentHistory(
                agent_state_update=state_update,
                tool_calls=h_data.get('tool_calls', []),
                tool_results=tool_results,
                browser_state=browser_state_hist,
                plan=plan
            ))
        return cls(history=history_items)

    # --- Helper methods to query history ---

    def errors(self) -> list[str]:
        """Get all errors from history"""
        errors = []
        for h in self.history:
            errors.extend([r.error for r in h.tool_results if r.error])
        return errors

    def final_result(self) -> None | str:
        """Final result from history (assuming 'done' tool provides it)"""
        if self.history:
            last_history = self.history[-1]
            if last_history.tool_calls and last_history.tool_calls[-1].get('done'):
                 if len(last_history.tool_results) == len(last_history.tool_calls):
                     last_result = last_history.tool_results[-1]
                     return last_result.extracted_content
        return None

    def is_done(self) -> bool:
        """Check if the agent called the 'done' tool in the last step"""
        if self.history:
            last_history = self.history[-1]
            if last_history.tool_calls and last_history.tool_calls[-1].get('done'):
                if len(last_history.tool_results) == len(last_history.tool_calls):
                    return last_history.tool_results[-1].is_done or False
        return False

    def has_errors(self) -> bool:
        """Check if the agent has any errors"""
        return len(self.errors()) > 0

    def urls(self) -> list[str]:
        """Get all unique URLs from history"""
        return list(set(h.browser_state.url for h in self.history if h.browser_state and h.browser_state.url))

    def screenshots(self) -> list[str]:
        """Get all screenshots from history"""
        return [h.browser_state.screenshot for h in self.history if h.browser_state and h.browser_state.screenshot]

    def action_names(self) -> list[str]:
        """Get all action (tool) names from history"""
        action_names = []
        for h in self.history:
            for tool_call in h.tool_calls:
                action_names.extend(list(tool_call.keys()))
        return action_names

    def model_thoughts(self) -> list[AgentStateUpdate]:
        """Get all agent state updates from history"""
        return [h.agent_state_update for h in self.history if h.agent_state_update]

    def extracted_content(self) -> list[str]:
        """Get all extracted content from history"""
        content = []
        for h in self.history:
            content.extend([r.extracted_content for r in h.tool_results if r.extracted_content])
        return content

    def plans(self) -> list[PlannerOutput | str | None]:
        """Get all plans generated during the run."""
        return [h.plan for h in self.history]
