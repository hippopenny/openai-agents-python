import pytest
from agents import Agent

# Import the agents defined in agent.py
from hippopenny_agents.agent_browser2.agent import (
    browser_agent,
    api_agent,
    orchestrator_agent,
)
# Import the planner agent
from hippopenny_agents.agent_browser2.planner import planner_agent
from hippopenny_agents.agent_browser2.models import PlannerOutput


def test_agents_are_instances_of_agent_class():
    """Verify that all defined agents are instances of agents.Agent."""
    assert isinstance(browser_agent, Agent)
    assert isinstance(api_agent, Agent)
    assert isinstance(orchestrator_agent, Agent)
    assert isinstance(planner_agent, Agent)

def test_orchestrator_agent_has_tools():
    """Verify the orchestrator agent has the browser and api tools."""
    assert hasattr(orchestrator_agent, 'tools')
    assert isinstance(orchestrator_agent.tools, list)
    assert len(orchestrator_agent.tools) >= 2 # At least browser and api tools

    tool_names = [tool.name for tool in orchestrator_agent.tools]
    assert "browser_tool" in tool_names
    assert "api_tool" in tool_names

    # Check descriptions (optional but good)
    browser_tool = next(t for t in orchestrator_agent.tools if t.name == "browser_tool")
    api_tool = next(t for t in orchestrator_agent.tools if t.name == "api_tool")

    assert "web browser interaction" in browser_tool.description
    assert "API interactions" in api_tool.description

def test_planner_agent_output_type():
    """Verify the planner agent has the correct output type."""
    assert hasattr(planner_agent, 'output_type')
    assert planner_agent.output_type == PlannerOutput

# Add more specific tests if needed, e.g., checking instructions content partially

