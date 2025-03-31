from __future__ import annotations

import json
import logging # Import logging
import warnings # Import warnings module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

# Removed imports from browser_use.*

logger = logging.getLogger(__name__) # Initialize logger

# --- Placeholder definition for ActionResult ---
# Kept simple as the placeholder context returns basic dicts now.
# A real implementation might use a more detailed model from browser_use or similar.
@dataclass
class ActionResult:
    """Placeholder for action results."""
    is_done: Optional[bool] = False
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    # Removed include_in_memory as its usage depends on the runner logic

# Removed placeholder BrowserStateHistory as state is handled as dict by placeholder context

# --- Tool Input Models ---
# Removed as tools.py is cleared/changed pattern

# --- Agent State Update Model ---
# Kept as it might be used by a potential update_agent_state tool
class AgentStateUpdate(BaseModel):
    """Model for the LLM to update its internal state analysis via the update_agent_state tool."""
    page_summary: str = Field(description="Quick detailed summary of new information from the current page which is not yet in the task history memory. Be specific with details which are important for the task. This is not on the meta level, but should be facts. If all the information is already in the task history memory, leave this empty.")
    evaluation_previous_goal: str = Field(description="Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not")
    memory: str = Field(description="Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz")
    next_goal: str = Field(description="What needs to be done with the next actions")


# --- Planner Output Model ---
# Kept as it's used by the Planner class
class PlannerOutput(BaseModel):
    """Model for the structured output expected from the Planner LLM."""
    state_analysis: str = Field(description="Brief analysis of the current state and what has been done so far based on history.")
    progress_evaluation: str = Field(description="Evaluation of progress towards the ultimate goal (e.g., percentage, description).")
    challenges: List[str] = Field(description="List any potential challenges or roadblocks identified.")
    next_steps: List[str] = Field(description="List 2-3 concrete high-level next steps to take towards the goal.")
    reasoning: str = Field(description="Explain the reasoning for the suggested next steps.")


# --- History Tracking ---
# Kept AgentHistory and AgentHistoryList structure, but note that the new
# service.py logic does not populate these in detail. This would require
# mapping the output from agents.Runner back into this structure if needed.

class AgentHistory(BaseModel):
    """History item for a single step of the agent's execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # Fields to store information about a step
    step_number: int
    state_before: Optional[Dict[str, Any]] = None # Store raw state dict
    plan: Optional[PlannerOutput | str] = None # Store plan used for this step
    orchestrator_input: Optional[str] = None # Store formatted input to orchestrator
    orchestrator_output_items: Optional[List[Dict[str, Any]]] = None # Store raw output items from Runner
    # Simplified results storage
    action_results: List[Dict[str, Any]] = Field(default_factory=list) # Store raw action result dicts

    # Removed agent_state_update, tool_calls, tool_results, browser_state
    # as the new orchestration doesn't map directly to these easily.

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization handling"""
        plan_dump = None
        if isinstance(self.plan, PlannerOutput):
            plan_dump = self.plan.model_dump()
        elif isinstance(self.plan, str):
            plan_dump = self.plan

        # Basic dump, might need refinement based on actual content
        return {
            'step_number': self.step_number,
            'state_before': self.state_before,
            'plan': plan_dump,
            'orchestrator_input': self.orchestrator_input,
            'orchestrator_output_items': self.orchestrator_output_items,
            'action_results': self.action_results,
        }


class AgentHistoryList(BaseModel):
    """List of agent history items, representing the full run."""
    history: list[AgentHistory] = Field(default_factory=list)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save history to JSON file with proper serialization"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            # Use the custom model_dump from AgentHistory
            data = {'history': [h.model_dump() for h in self.history]}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Add logging for save errors
            logger.error(f"Failed to save history to {filepath}: {e}", exc_info=True)
            raise e # Re-raise the exception

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> 'AgentHistoryList':
        """Load history from JSON file"""
        # This needs careful reimplementation based on the new AgentHistory structure
        # For now, return an empty list as parsing the new structure isn't defined.
        # Use warnings.warn instead of logger.warning for testability with pytest.warns
        # warnings.warn(
        #     f"AgentHistoryList.load_from_file needs reimplementation for new AgentHistory structure. Returning empty history from {filepath}.",
        #     UserWarning,
        #     stacklevel=2
        # )
        # Example parsing logic would go here if needed:
        try:
            # Actually open the file now
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Basic parsing attempt - might fail if structure is complex/changed
            history_items = [AgentHistory(**h_data) for h_data in data.get('history', [])]
            return cls(history=history_items)
        except FileNotFoundError:
             logger.error(f"History file not found: {filepath}")
             warnings.warn(f"History file not found: {filepath}", UserWarning, stacklevel=2)
             return cls(history=[])
        except Exception as e:
             logger.error(f"Failed to load or parse history from {filepath}: {e}", exc_info=True)
             warnings.warn(f"Failed to load or parse history from {filepath}: {e}", UserWarning, stacklevel=2)
             return cls(history=[])


    # --- Helper methods might need adjustment based on new AgentHistory structure ---

    def errors(self) -> list[str]:
        """Get all errors from action results history"""
        errors = []
        for h in self.history:
            errors.extend([str(r.get('error')) for r in h.action_results if r.get('error')])
        return errors

    def final_result(self) -> None | str:
        """Attempt to find a final result (e.g., from a 'done' action). Needs refinement."""
        if not self.history: return None
        last_step = self.history[-1]
        # Check last action result in the last step
        if last_step.action_results:
            last_action_result = last_step.action_results[-1]
            # Heuristic: check if action was 'done' or if result indicates completion
            action_executed = last_action_result.get('action_executed', {})
            action_name = next(iter(action_executed), None)
            if action_name == 'done' or last_action_result.get('is_done'):
                 # Extract content if available
                 return str(last_action_result.get('result', 'Task marked done.'))
        # Check orchestrator text output as fallback
        if last_step.orchestrator_output_items:
             last_msg = last_step.orchestrator_output_items[-1]
             if last_msg.get("type") == "message" and last_msg.get("role") == "assistant":
                 content = last_msg.get("content", "")
                 if isinstance(content, list): # Handle vision format
                     text_parts = [part.get("text","") for part in content if part.get("type")=="text"]
                     return "\n".join(text_parts)
                 elif isinstance(content, str):
                     return content
        return None

    def is_done(self) -> bool:
        """Check if the task was marked done. Needs refinement."""
        if not self.history: return False
        last_step = self.history[-1]
        if last_step.action_results:
            last_action_result = last_step.action_results[-1]
            action_executed = last_action_result.get('action_executed', {})
            action_name = next(iter(action_executed), None)
            if action_name == 'done' or last_action_result.get('is_done'):
                return True
        return False

    def has_errors(self) -> bool:
        """Check if any step has errors"""
        return len(self.errors()) > 0

    def urls(self) -> list[str]:
        """Get all unique URLs from state history"""
        urls = set()
        for h in self.history:
            if h.state_before and h.state_before.get('url'):
                urls.add(h.state_before['url'])
        return list(urls)

    def screenshots(self) -> list[str]:
        """Get all screenshots from state history"""
        screenshots = []
        for h in self.history:
             if h.state_before and h.state_before.get('screenshot'):
                 screenshots.append(h.state_before['screenshot'])
        return screenshots

    # Other helper methods (action_names, model_thoughts, extracted_content, plans)
    # would need significant rework based on how data is stored in the new AgentHistory.
    # Omitting them for now.
