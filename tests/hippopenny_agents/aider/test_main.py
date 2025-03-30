import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from agents import (
    Runner, RunResult, FunctionTool, MessageOutputItem, ToolCallItem,
    ToolCallOutputItem, ResponseOutputText
)

from hippopenny_agents.aider.main import run_aider
from hippopenny_agents.aider.models import ProjectContext, CoderOutput, TaskStatus
from hippopenny_agents.aider.agents import PlannerAgent # Import the actual agent


# --- Helper to create RunResult ---
def create_planner_run_result(
    input_msg: str,
    planner_response: str | None = None,
    tool_call_item: ToolCallItem | None = None,
    tool_output_item: ToolCallOutputItem | None = None,
    context: ProjectContext | None = None # Context state *after* this run
) -> RunResult:
    items = []
    final_output = None
    if planner_response:
        msg_item = MessageOutputItem(
            agent_name="PlannerAgent",
            model_name="mock_planner_model",
            response_id="planner_resp",
            content=[ResponseOutputText(text=planner_response, type="text")]
        )
        items.append(msg_item)
        final_output = planner_response # Simplified final output for testing
    if tool_call_item:
        items.append(tool_call_item)
    if tool_output_item:
        items.append(tool_output_item)
        # If tool output exists, it's often considered the 'final' actionable output of the turn
        final_output = tool_output_item.output

    # Simulate the structure Runner.run returns
    return RunResult(
        input=input_msg,
        final_output=final_output, # May not be accurate representation, depends on items
        usage=AsyncMock(),
        new_items=items, # Items generated *in this specific run*
        all_items=items, # Simplified for testing (usually includes history)
        last_agent=PlannerAgent,
        model_name="mock_planner_model",
        response_id="planner_resp_id",
        context=context # Include context if needed for assertions later
    )

# --- Test ---

@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_run_aider_full_loop(mock_runner_run: AsyncMock):
    """Test the main loop from goal to completion using the revised structure."""

    project_goal = "Create hello.py"
    max_turns = 5

    # --- Mocking Strategy ---
    # Mock `Runner.run` which is called repeatedly for the PlannerAgent.
    # Simulate the Planner calling different tools over several turns.
    # The *tools themselves* modify the context, so the context passed
    # back to the mock `Runner.run` on the next iteration should reflect those changes.

    # We need to manage the state of the context across mock calls.
    test_context = ProjectContext(project_goal=project_goal)

    # Define the sequence of mock results from Planner's Runner.run calls
    async def mock_run_side_effect(*args, **kwargs):
        agent = args[0]
        input_msg = args[1]
        # IMPORTANT: The context passed to Runner.run is the *mutable* context object
        context: ProjectContext = kwargs['context']
        call_count = mock_runner_run.call_count # 1-based call count

        assert agent == PlannerAgent # Ensure only planner is called by main loop

        if call_count == 1:
            # Planner receives initial goal, decides to plan initial tasks
            tool_call = ToolCallItem(tool_name="plan_initial_tasks", tool_input={'task_descriptions': ['Create file', 'Write code', 'Test script']}, tool_call_id="call_1")
            # Simulate tool execution: update context *before* returning result
            # In real execution, the tool function modifies the context. We mimic that here.
            context.add_tasks(['Create file', 'Write code', 'Test script'])
            tool_output = ToolCallOutputItem(tool_name="plan_initial_tasks", tool_call_id="call_1", output="Successfully planned 3 initial tasks.")
            return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)

        elif call_count == 2:
            # Planner receives result of planning, decides to implement task 0 (ID 0)
            assert context.tasks[0].status == "pending"
            task_id = context.tasks[0].id
            tool_call = ToolCallItem(tool_name="implement_task", tool_input={'task_id': task_id}, tool_call_id="call_2")
            # Simulate tool execution (including internal Coder run)
            context.update_task_status(task_id, "done") # Simulate coder success
            tool_output = ToolCallOutputItem(tool_name="implement_task", tool_call_id="call_2", output="Coder completed task 0.")
            return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)

        elif call_count == 3:
            # Planner receives result of task 0, implements task 1 (ID 1)
            assert context.tasks[0].status == "done"
            assert context.tasks[1].status == "pending"
            task_id = context.tasks[1].id
            tool_call = ToolCallItem(tool_name="implement_task", tool_input={'task_id': task_id}, tool_call_id="call_3")
            context.update_task_status(task_id, "done")
            tool_output = ToolCallOutputItem(tool_name="implement_task", tool_call_id="call_3", output="Coder completed task 1.")
            return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)

        elif call_count == 4:
            # Planner receives result of task 1, implements task 2 (ID 2)
            assert context.tasks[1].status == "done"
            assert context.tasks[2].status == "pending"
            task_id = context.tasks[2].id
            tool_call = ToolCallItem(tool_name="implement_task", tool_input={'task_id': task_id}, tool_call_id="call_4")
            context.update_task_status(task_id, "done")
            tool_output = ToolCallOutputItem(tool_name="implement_task", tool_call_id="call_4", output="Coder completed task 2.")
            return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)

        elif call_count == 5:
            # Planner receives result of task 2, sees all tasks done, responds with completion message
            assert context.are_all_tasks_done()
            completion_message = "Project finished successfully."
            return create_planner_run_result(input_msg, planner_response=completion_message, context=context)

        else:
            # Should not be called more times in this scenario
            pytest.fail(f"Planner called too many times ({call_count})")

    mock_runner_run.side_effect = mock_run_side_effect

    # --- Run the aider ---
    # Pass the *mutable* test_context
    final_context, final_message = await run_aider(project_goal, max_turns=max_turns)

    # --- Assertions ---
    assert mock_runner_run.call_count == 5 # Called 5 times as simulated
    assert final_context.are_all_tasks_done()
    assert len(final_context.tasks) == 3
    assert final_context.tasks[0].status == "done"
    assert final_context.tasks[1].status == "done"
    assert final_context.tasks[2].status == "done"
    assert final_message == "Project finished successfully." # Check final message from planner

    # Verify calls to Runner.run had the correct context progression (simplified check)
    expected_inputs = [
        f"Project Goal: {project_goal}. The task list is currently empty. Please plan the initial tasks.",
        "Previous action: Called tool 'plan_initial_tasks'. Result: Successfully planned 3 initial tasks.. Current task status is updated in the context. Decide the next step based on the project goal and task list.",
        "Previous action: Called tool 'implement_task'. Result: Coder completed task 0.. Current task status is updated in the context. Decide the next step based on the project goal and task list.",
        "Previous action: Called tool 'implement_task'. Result: Coder completed task 1.. Current task status is updated in the context. Decide the next step based on the project goal and task list.",
        "Previous action: Called tool 'implement_task'. Result: Coder completed task 2.. Current task status is updated in the context. Decide the next step based on the project goal and task list.",
    ]
    for i, expected_input in enumerate(expected_inputs):
         call_args, call_kwargs = mock_runner_run.call_args_list[i]
         assert call_args[1] == expected_input # Check input message for each turn
         # Ensure the *same* context object instance was passed each time
         assert call_kwargs['context'] is test_context


@pytest.mark.asyncio
@patch('agents.Runner.run', new_callable=AsyncMock)
async def test_run_aider_max_turns(mock_runner_run: AsyncMock):
    """Test that the loop terminates due to max_turns."""

    project_goal = "Max turns test"
    max_turns = 3
    test_context = ProjectContext(project_goal=project_goal)

    # Mock Planner to always call implement_task but never finish
    async def mock_run_side_effect(*args, **kwargs):
        context: ProjectContext = kwargs['context']
        input_msg = args[1]
        call_count = mock_runner_run.call_count

        if not context.tasks:
             # Simulate planning on first turn
             tool_call = ToolCallItem(tool_name="plan_initial_tasks", tool_input={'task_descriptions': ['Task 0']}, tool_call_id=f"call_{call_count}")
             context.add_tasks(['Task 0'])
             tool_output = ToolCallOutputItem(tool_name="plan_initial_tasks", tool_call_id=f"call_{call_count}", output="Planned task 0.")
             return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)
        else:
             # Simulate calling implement but never marking done
             task_id = context.tasks[0].id
             tool_call = ToolCallItem(tool_name="implement_task", tool_input={'task_id': task_id}, tool_call_id=f"call_{call_count}")
             # Don't update context status to done
             context.update_task_status(task_id, "in_progress") # Or failed
             tool_output = ToolCallOutputItem(tool_name="implement_task", tool_call_id=f"call_{call_count}", output="Coder still working...")
             return create_planner_run_result(input_msg, tool_call_item=tool_call, tool_output_item=tool_output, context=context)

    mock_runner_run.side_effect = mock_run_side_effect

    final_context, final_message = await run_aider(project_goal, max_turns=max_turns)

    assert mock_runner_run.call_count == max_turns # Called exactly max_turns times
    assert f"Max turns ({max_turns}) reached" in final_message
    assert not final_context.are_all_tasks_done()
