import pytest

from hippopenny_agents.aider.models import ProjectContext, Task


def test_project_context_initialization():
    goal = "Create a calculator app"
    context = ProjectContext(project_goal=goal)
    assert context.project_goal == goal
    assert context.tasks == []
    assert context.coder_error is None


def test_add_tasks():
    context = ProjectContext(project_goal="Test")
    context.add_tasks(["Task 1", "Task 2"])
    assert len(context.tasks) == 2
    assert context.tasks[0].id == 0
    assert context.tasks[0].description == "Task 1"
    assert context.tasks[0].status == "pending"
    assert context.tasks[1].id == 1
    assert context.tasks[1].description == "Task 2"
    assert context.tasks[1].status == "pending"

    context.add_tasks(["Task 3"])
    assert len(context.tasks) == 3
    assert context.tasks[2].id == 2
    assert context.tasks[2].description == "Task 3"


def test_get_next_pending_task():
    context = ProjectContext(project_goal="Test")
    assert context.get_next_pending_task() is None  # No tasks

    context.add_tasks(["Task 1", "Task 2"])
    next_task = context.get_next_pending_task()
    assert next_task is not None
    assert next_task.id == 0
    assert next_task.description == "Task 1"

    context.tasks[0].status = "done"
    next_task = context.get_next_pending_task()
    assert next_task is not None
    assert next_task.id == 1
    assert next_task.description == "Task 2"

    context.tasks[1].status = "in_progress"
    assert context.get_next_pending_task() is None # No pending tasks left

    context.tasks[1].status = "done"
    assert context.get_next_pending_task() is None # All done


def test_update_task_status():
    context = ProjectContext(project_goal="Test")
    context.add_tasks(["Task 1"])
    task_id = context.tasks[0].id

    context.update_task_status(task_id, "in_progress")
    assert context.tasks[0].status == "in_progress"
    assert context.coder_error is None

    context.update_task_status(task_id, "done")
    assert context.tasks[0].status == "done"
    assert context.coder_error is None

    context.update_task_status(task_id, "failed", error="Syntax error")
    assert context.tasks[0].status == "failed"
    assert context.coder_error == "Syntax error"

    # Test updating non-existent task
    with pytest.raises(ValueError):
        context.update_task_status(999, "done")


def test_are_all_tasks_done():
    context = ProjectContext(project_goal="Test")
    assert not context.are_all_tasks_done() # No tasks

    context.add_tasks(["Task 1", "Task 2"])
    assert not context.are_all_tasks_done() # Tasks are pending

    context.tasks[0].status = "done"
    assert not context.are_all_tasks_done() # One task still pending

    context.tasks[1].status = "in_progress"
    assert not context.are_all_tasks_done() # One task in progress

    context.tasks[1].status = "failed"
    assert not context.are_all_tasks_done() # One task failed

    context.tasks[1].status = "done"
    assert context.are_all_tasks_done() # All tasks done
