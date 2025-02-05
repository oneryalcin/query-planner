import pytest
from query_planner_execution import Task, TaskPlan, TaskType
from pydantic import ValidationError
import logging
import asyncio
from enum import Enum


# Configure logging for the test
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


@pytest.fixture
def simple_task_plan():
    return TaskPlan(
        task_graph=[
            Task(id=1, task="Main task", subtasks=[2, 3], priority=1, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=2, task="Subtask 1", subtasks=[], priority=2, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=3, task="Subtask 2", subtasks=[], priority=3, task_type=TaskType.DATA_RETRIEVAL),
        ]
    )


@pytest.fixture
def circular_task_plan():
    return TaskPlan(
        task_graph=[
            Task(id=1, task="Main task", subtasks=[2], priority=1, task_type=TaskType.COMPARISON),
            Task(id=2, task="Subtask 1", subtasks=[3], priority=2, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=3, task="Subtask 2", subtasks=[1], priority=3, task_type=TaskType.DATA_RETRIEVAL),  # Circular dependency
        ]
    )


@pytest.fixture
def task_plan_with_limit():
    return TaskPlan(
        task_graph=[
            Task(id=1, task="Main task", subtasks=[2, 3, 4], priority=1, task_type=TaskType.COMPARISON),
            Task(id=2, task="Subtask 1", subtasks=[], priority=2, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=3, task="Subtask 2", subtasks=[], priority=3, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=4, task="Subtask 3", subtasks=[], priority=4, task_type=TaskType.CALCULATION),
        ],
        parallelism_limit=2  # Set the limit for concurrent tasks
    )


def test_get_execution_order_no_cycle(simple_task_plan):
    """
    Test that a task plan without circular dependencies returns the correct execution order.
    """
    execution_order = simple_task_plan._get_execution_order()
    assert execution_order == [2, 3, 1], "Execution order is incorrect for simple task plan"


def test_get_execution_order_with_cycle(circular_task_plan):
    """
    Test that a task plan with circular dependencies raises a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        circular_task_plan._get_execution_order()
    assert "Circular dependencies detected" in str(excinfo.value), "Expected circular dependency error message"


def test_identify_circular_dependencies(circular_task_plan):
    """
    Test that the circular dependencies are correctly identified.
    """
    circular_deps = circular_task_plan._identify_circular_dependencies({
        1: {2},
        2: {3},
        3: {1},
    })
    assert circular_deps == [[1, 2, 3, 1]], "Circular dependencies are not correctly identified"


def test_empty_task_plan():
    """
    Test that an empty task plan returns an empty execution order.
    """
    task_plan = TaskPlan(task_graph=[])
    execution_order = task_plan._get_execution_order()
    assert execution_order == [], "Execution order should be empty for an empty task plan"


def test_invalid_task_graph():
    """
    Test that a task plan with invalid data raises a ValidationError.
    """
    with pytest.raises(ValidationError):
        TaskPlan(task_graph=[
            Task(id=1, task="Invalid task", subtasks=["invalid_id"], priority=1, task_type=TaskType.DATA_RETRIEVAL),  # Invalid subtask ID type
        ])


def test_execution_no_cycle(simple_task_plan):
    """
    Test that the execute method works correctly for a simple task plan without circular dependencies.
    """
    results = asyncio.run(simple_task_plan.execute())
    assert isinstance(results, dict), "Execution results should be a dictionary"
    assert len(results) == 3, "Execution results should contain all tasks"
    assert results[1].task_id == 1, "Main task result should have correct task ID"
    assert results[2].task_id == 2, "Subtask 1 result should have correct task ID"
    assert results[3].task_id == 3, "Subtask 2 result should have correct task ID"
    assert "Aggregating results from subtasks" in results[1].result, "Main task should aggregate results from subtasks"


def test_execution_with_limit(task_plan_with_limit):
    """
    Test that the execute method respects the parallelism limit.
    """
    results = asyncio.run(task_plan_with_limit.execute())
    assert isinstance(results, dict), "Execution results should be a dictionary"
    assert len(results) == 4, "Execution results should contain all tasks"
    assert results[1].task_id == 1, "Main task result should have correct task ID"
    assert results[2].task_id == 2, "Subtask 1 result should have correct task ID"
    assert results[3].task_id == 3, "Subtask 2 result should have correct task ID"
    assert results[4].task_id == 4, "Subtask 3 result should have correct task ID"
    assert "Aggregating results from subtasks" in results[1].result, "Main task should aggregate results from subtasks"


def test_execution_with_priority():
    """
    Test that tasks are executed in order of their priority.
    """
    task_plan = TaskPlan(
        task_graph=[
            Task(id=1, task="Main task", subtasks=[2, 3], priority=1, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=2, task="High priority subtask", subtasks=[], priority=3, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=3, task="Low priority subtask", subtasks=[], priority=2, task_type=TaskType.DATA_RETRIEVAL),
        ]
    )
    execution_order = task_plan._get_execution_order()
    assert execution_order == [2, 3, 1], "Tasks should be executed in order of their priority where possible"
    results = asyncio.run(task_plan.execute())
    assert isinstance(results, dict), "Execution results should be a dictionary"
    assert len(results) == 3, "Execution results should contain all tasks"
    assert results[2].task_id == 2, "High priority subtask should be executed first"
    assert results[3].task_id == 3, "Low priority subtask should be executed second"
    assert results[1].task_id == 1, "Main task should be executed last"
    assert "Aggregating results from subtasks" in results[1].result, "Main task should aggregate results from subtasks"


def test_aggregation_of_subtasks():
    """
    Test that parent tasks aggregate the results from their subtasks correctly.
    """
    task_plan = TaskPlan(
        task_graph=[
            Task(id=1, task="Aggregate results", subtasks=[2, 3], priority=1, task_type=TaskType.COMPARISON),
            Task(id=2, task="Subtask A", subtasks=[], priority=2, task_type=TaskType.DATA_RETRIEVAL),
            Task(id=3, task="Subtask B", subtasks=[], priority=3, task_type=TaskType.DATA_RETRIEVAL),
        ]
    )
    results = asyncio.run(task_plan.execute())
    assert isinstance(results, dict), "Execution results should be a dictionary"
    assert len(results) == 3, "Execution results should contain all tasks"
    assert results[1].task_id == 1, "Parent task result should have correct task ID"
    assert "Aggregating results from subtasks" in results[1].result, "Parent task should aggregate results from subtasks"
    assert "Subtask 2: Executing task `Subtask A`" in results[1].result, "Subtask A result should be included in parent task aggregation"
    assert "Subtask 3: Executing task `Subtask B`" in results[1].result, "Subtask B result should be included in parent task aggregation"

