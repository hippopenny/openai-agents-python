from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Removed imports from langchain_core, browser_use

if TYPE_CHECKING:
    # Removed type hints for BaseChatModel, BrowserContext, Controller
    from .views import AgentHistoryList, AgentStateUpdate, PlannerOutput, ActionResult


@dataclass
class BrowserAgentContext:
    """Context object holding state for the browser agent run.
    NOTE: Dependencies on browser_use and langchain/agents have been removed.
    Browser/Controller interaction logic needs to be reimplemented.
    """

    task: str
    # browser_context: BrowserContext # Removed dependency
    # controller: Controller # Removed dependency
    history: AgentHistoryList = field(default_factory=lambda: AgentHistoryList(history=[]))
    last_results: List[ActionResult] = field(default_factory=list) # Uses placeholder ActionResult
    current_agent_state: AgentStateUpdate | None = None
    max_steps: int = 100
    current_step: int = 0
    use_vision: bool = True
    include_attributes: list[str] = field(
        default_factory=lambda: [
            'title',
            'type',
            'name',
            'role',
            'tabindex',
            'aria-label',
            'placeholder',
            'value',
            'alt',
            'aria-expanded',
        ]
    )
    max_error_length: int = 400
    sensitive_data: Optional[Dict[str, str]] = None
    available_file_paths: Optional[list[str]] = None
    agent_id: str | None = None # To be set by the runner/orchestrator
    telemetry: Any | None = None # Placeholder for telemetry client
    page_extraction_llm: Any | None = None # Placeholder for extraction LLM (needs type)

    # Planner specific fields
    planner_llm: Any | None = None # Placeholder for planner LLM (needs type)
    planning_interval: int = 1 # How often to run the planner (e.g., every 1 step)
    last_plan: PlannerOutput | str | None = None # Store the last output from the planner
    use_vision_for_planner: bool = False # Whether planner should receive images

    # Flag to indicate if the 'done' tool has been called
    is_done: bool = False

    # Placeholders for browser/controller instances if needed by reimplemented logic
    browser_interface: Any | None = None
    controller_interface: Any | None = None
