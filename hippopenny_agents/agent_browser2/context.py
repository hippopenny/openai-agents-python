from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from browser_use.agent.views import ActionResult
    from browser_use.browser.context import BrowserContext
    from browser_use.controller.service import Controller

    from .views import AgentHistoryList, AgentStateUpdate, PlannerOutput


@dataclass
class BrowserAgentContext:
    """Context object holding state for the browser agent run."""

    task: str
    browser_context: BrowserContext
    controller: Controller
    history: AgentHistoryList = field(default_factory=lambda: AgentHistoryList(history=[]))
    last_results: List[ActionResult] = field(default_factory=list)
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
    agent_id: str | None = None # To be set by the runner
    telemetry: Any | None = None # Placeholder for telemetry client
    page_extraction_llm: Any | None = None # Placeholder for extraction LLM

    # Planner specific fields
    planner_llm: Optional[BaseChatModel] = None
    planning_interval: int = 1 # How often to run the planner (e.g., every 1 step)
    last_plan: PlannerOutput | str | None = None # Store the last output from the planner
    use_vision_for_planner: bool = False # Whether planner should receive images

    # Flag to indicate if the 'done' tool has been called
    is_done: bool = False
