from __future__ import annotations

import logging

# Re-introduce agents SDK dependency
from agents import Agent

from .prompts import PlannerPromptBuilder
from .models import PlannerOutput

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 3. Planner: Defined as an agents.Agent
# (Replaces the previous Planner class)
# ----------------------------------------------------------

# Create an instance of the prompt builder to get the system message
_planner_prompt_builder = PlannerPromptBuilder()
_planner_system_prompt = _planner_prompt_builder.get_system_message_content()

# Define the planner as an agent
planner_agent = Agent(
    name="planner_agent",
    instructions=_planner_system_prompt,
    output_type=PlannerOutput,
    # Add model settings if needed, e.g., specific model for planning
    # model_settings=ModelSettings(model="gpt-4-turbo"),
    handoff_description="analyzes history and generates a plan with state analysis, progress, challenges, and next steps.",
    model="gpt-4o-mini"
)

logger.info(f"Defined planner_agent using agents.Agent with output type {PlannerOutput.__name__}")

