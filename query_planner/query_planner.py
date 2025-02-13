"""
Proof of Concept for a task planning and execution system using
OpenAIs Functions and topological sort, based on the idea in
query_planner_execution.py.py.

Additionally: There are also cases where the "pure" recursive approach has advantages;
If subtasks for different parent tasks that start in parallel have different runtimes,
we will wait unnecessarily with my current implementation.

Added by Jan Philipp Harries / @jpdus
"""

import asyncio
from enum import Enum, StrEnum
from datetime import datetime
from collections.abc import Generator
import logging
from typing import Dict, Set, List, Generator, Optional, Callable, Any

import instructor
from openai import OpenAI
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr
from loguru import logger

client = instructor.from_openai(OpenAI())


class TaskResult(BaseModel):
    task_id: int
    result: str


class TaskResults(BaseModel):
    results: list[TaskResult]
    

class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TaskType(StrEnum):
    DATA_RETRIEVAL = "data_retrieval"
    COMPARISON = "comparison"
    CALCULATION = "calculation"
    TEMPORAL_REASONING = "temporal_reasoning"

class Task(BaseModel):
    """
    Class representing a single task in a task plan.
    """

    id: int = Field(..., description="Unique id of the task")
    task: str = Field(
        ...,
        description="""Contains the task in text form. If there are multiple tasks,
        this task can only be executed when all dependant subtasks have been answered.""",
    )
    subtasks: list[int] = Field(
        default_factory=list,
        description="""List of the IDs of subtasks that need to be answered before
        we can answer the main question. Use a subtask when anything may be unknown
        and we need to ask multiple questions to get the answer.
        Dependencies must only be other tasks.""",
    )
    priority: int = Field(1, description="Priority of the task. Higher numbers indicate higher priority")
    task_type: TaskType = Field(..., description="Type of the task, indicating the kind of operation to perform.")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Status of the task")


    async def aexecute(self, with_results: TaskResults) -> TaskResult:
        """
        Executes the task by reasoning over the task description and the results of its subtasks.
        
        TODO: Do some real work here.
        """
        # Perform reasoning based on the task description and subtask results.
        # Placeholder logic for different task types
        if self.task_type == TaskType.DATA_RETRIEVAL:
            reasoning_output = f"Retrieving data for `{self.task}`. Subtask results: {with_results}"
        elif self.task_type == TaskType.COMPARISON:
            reasoning_output = f"Comparing results for `{self.task}`. Subtask results: {with_results}"
        elif self.task_type == TaskType.CALCULATION:
            reasoning_output = f"Calculating based on `{self.task}`. Subtask results: {with_results}"
        elif self.task_type == TaskType.TEMPORAL_REASONING:
            reasoning_output = f"Reasoning over time for `{self.task}`. Subtask results: {with_results}"
        else:
            reasoning_output = f"Executing task `{self.task}`. Subtask results: {with_results}"

        logger.info(f"Executing task {self.id}: {self.task}")
        reasoning_output = f"Executing task `{self.task}`. Aggregating results from subtasks: "
        subtask_details = []

        for result in with_results.results:
            subtask_details.append(f"(Subtask {result.task_id}: {result.result})")

        reasoning_output += ", ".join(subtask_details)
        
        # You could enhance this to involve more complex decision-making.
        final_result = f"{reasoning_output} -> Final output for task {self.id}"

        await asyncio.sleep(1)
        logger.info(f"Task {self.id} executed at {datetime.now()}")
        logger.info(final_result)
        
        return TaskResult(task_id=self.id, result=final_result)



# def log_task_completion(task: Task, *args, **kwargs):
#     logger.info(f"Notification: Task {args}")
#     logger.info(f"Notification: Task {task.id} - '{task.task}' has been completed.", *args, **kwargs)
#     # logger.info(f"Notification: Task {task.id} - '{task.task}' has been completed.")
    

def log_task_completion(*args):
    logger.info(f"Notification: Task {args}")
    
class TaskPlan(BaseModel):
    """
    Container class representing a tree of tasks and subtasks.
    Make sure every task is in the tree, and every task is done only once.
    """

    task_graph: list[Task] = Field(
        ...,
        description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.",
    )
    parallelism_limit: int = Field(3, description="Maximum number of tasks to execute concurrently.")
    _notify_on_completion: Optional[Callable[[Task], None]] = PrivateAttr(default=None)  # Optional callback function for task completion 
    
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        self._notify_on_completion = data.get('_notify_on_completion', None)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump['task_graph'] = [task.model_dump() for task in self.task_graph]
        return dump


    def _get_execution_order(self) -> List[int]:
        """
        Returns the order in which the tasks should be executed using topological sort.
        Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
        """
        tmp_dep_graph: Dict[int, Set[int]] = {item.id: set(item.subtasks) for item in self.task_graph}

        def topological_sort(dep_graph: Dict[int, Set[int]]) -> Generator[Set[int], None, None]:
            while dep_graph:
                ordered = set(item for item, dep in dep_graph.items() if not dep)
                if not ordered:
                    circular_deps = self._identify_circular_dependencies(dep_graph)
                    error_msg = (
                        f"Circular dependencies detected among the following tasks: "
                        f"{'; '.join(' -> '.join(map(str, cycle)) for cycle in circular_deps)}. "
                        "Please review your task dependencies and ensure there are no loops."
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                yield ordered
                dep_graph = {
                    item: (dep - ordered)
                    for item, dep in dep_graph.items()
                    if item not in ordered
                }

        result = []
        for d in topological_sort(tmp_dep_graph):
            result.extend(sorted(d))
        return result

    def _identify_circular_dependencies(self, dep_graph: Dict[int, Set[int]]) -> List[List[int]]:
        """
        Identifies circular dependencies in the dependency graph.
        
        TODO: We may also want to consider the scalability of the DFS cycle detection 
        for large graphs. A more optimized approach could involve 
        Tarjan’s Strongly Connected Components (SCC) algorithm 
        to detect all cycles more efficiently.
        
        Another option would be optimizing DFS. If performance becomes an issue, 
        add a visited set to the DFS function to prevent unnecessary revisits.
        """
        def dfs(node: int, path: List[int]) -> List[List[int]]:
            if node in path:
                return [path[path.index(node):] + [node]]
            cycles = []
            for neighbor in dep_graph[node]:
                cycles.extend(dfs(neighbor, path + [node]))
            return cycles

        all_cycles = []
        for node in dep_graph:
            all_cycles.extend(dfs(node, []))
        
        # Remove duplicates and rotations
        unique_cycles = []
        for cycle in all_cycles:
            normalized = tuple(cycle[:-1])  # Remove last element as it's same as first
            rotations = {tuple(normalized[i:] + normalized[:i]) for i in range(len(normalized))}
            if not any(r in unique_cycles for r in rotations):
                unique_cycles.append(normalized)
        
        return [list(cycle) + [cycle[0]] for cycle in unique_cycles]  # Add first element to end

    async def execute(self) -> dict[int, TaskResult]:
        """
        Executes the tasks in the task plan in the correct order using asyncio and chunks with answered dependencies.
        Enforces a limit on the number of concurrently executing tasks and respects task priority.
        """
        # Get the maximum number of tasks to execute concurrently
        parallelism_limit = getattr(self, 'parallelism_limit', 3)
        semaphore = asyncio.Semaphore(parallelism_limit)

        execution_order = self._get_execution_order()
        tasks = {q.id: q for q in self.task_graph}
        task_results = {}

        async def execute_task(task):
            async with semaphore:
                # Gather results of all subtasks
                subtask_results = [
                    result
                    for result in task_results.values()
                    if result.task_id in task.subtasks
                ]
                
                task.status = TaskStatus.IN_PROGRESS
                print(self.generate_mermaid_graph())  # Visualize the current state before execution

                # Execute the task with results from its subtasks
                result = await task.aexecute(
                    with_results=TaskResults(results=subtask_results)
                )
                
                # Check if new tasks need to be added based on the result
                if "Add new task" in result.result:  # Placeholder for the logic to add new tasks
                    new_task_id = max(tasks.keys()) + 1
                    new_task = Task(
                        id=new_task_id,
                        task=f"Newly added subtask of task {task.id}",
                        subtasks=[],
                        priority=1,
                        task_type=TaskType.DATA_RETRIEVAL,
                        status=TaskStatus.PENDING,
                    )
                    self.add_task(new_task, parent_task_id=task.id)
                    tasks[new_task_id] = new_task
                    
                
                task.status = TaskStatus.COMPLETED
                task_results[result.task_id] = result
                
                 # Notify on task completion, if callback is set
                if self._notify_on_completion:
                    self._notify_on_completion(task)
                
                print(self.generate_mermaid_graph())  # Visualize the current state after execution

        while len(task_results) < len(execution_order):
            # Get tasks that are ready to execute
            ready_to_execute = [
                tasks[task_id]
                for task_id in execution_order
                if task_id not in task_results
                and all(
                    subtask_id in task_results for subtask_id in tasks[task_id].subtasks
                )
            ]

            # Sort ready tasks by priority (higher priority first)
            ready_to_execute.sort(key=lambda task: task.priority, reverse=True)

            # prints chunks to visualize execution order
            print(f"Ready to execute: {[task.id for task in ready_to_execute]}")

            await asyncio.gather(*[execute_task(task) for task in ready_to_execute])
            
            # Recalculate the execution order to include the new task and its dependencies
            execution_order = self._get_execution_order()

        return task_results
    
    def add_task(self, new_task: Task, parent_task_id: int = None):
        """
        Adds a new task to the task graph and optionally links it as a subtask of an existing task.
        
        Args:
            new_task (Task): The task to be added.
            parent_task_id (int, optional): The ID of the task that will treat this as a subtask.
        """
        # Check if a task with the same ID already exists
        if any(task.id == new_task.id for task in self.task_graph):
            raise ValueError(f"Task with id {new_task.id} already exists.")
        
        # Append the new task
        self.task_graph.append(new_task)
        
        # Update parent task with the new subtask if a parent is specified
        if parent_task_id is not None:
            parent_task = next((task for task in self.task_graph if task.id == parent_task_id), None)
            if parent_task:
                parent_task.subtasks.append(new_task.id)
            else:
                raise ValueError(f"Parent task with id {parent_task_id} not found.")
    
    
    def generate_mermaid_graph(self) -> str:
        """
        Generates a Mermaid diagram representing the task dependencies, with nodes colored based on their status.
        """
        lines = ["graph TD"]

        # Add nodes and edges based on dependencies
        for task in self.task_graph:
            task_label = f"Task {task.id}: {task.task}"

            # Customize node styles based on task status
            if task.status == TaskStatus.COMPLETED:
                lines.append(f'    T{task.id}["{task_label}"]:::completed')
            elif task.status == TaskStatus.IN_PROGRESS:
                lines.append(f'    T{task.id}["{task_label}"]:::in_progress')
            else:  # TaskStatus.PENDING
                lines.append(f'    T{task.id}["{task_label}"]:::pending')

            for subtask_id in task.subtasks:
                lines.append(f"    T{subtask_id} --> T{task.id}")

        # Define custom classes for the nodes to represent different statuses
        lines.append("    classDef completed fill:#9f6,stroke:#333,stroke-width:2px;")
        lines.append("    classDef in_progress fill:#ff9,stroke:#333,stroke-width:2px;")
        lines.append("    classDef pending fill:#f96,stroke:#333,stroke-width:2px;")

        return "\n".join(lines)

Task.model_rebuild()
TaskPlan.model_rebuild()

# TaskPlan._notify_on_completion = log_task_completion


system_prompt = """You are an advanced task planning algorithm designed to decompose a main task into a set of clear, dependent subtasks. Your goals are:

1. **Granularity**: Break down the main task into single-fact, specific questions that are actionable.
2. **Dependency Management**: Clearly define dependencies among subtasks. Tasks without dependencies should be executable in parallel.
3. **Avoid Ambiguity**: Do not use conjunctions like "and" in subtasks unless they further break down into more specific subtasks.
4. **Clarity**: Each subtask should be a standalone question seeking a single fact or piece of information.
5. **Date**: Use the date in the question to inform the date in the subtasks. Don't be vague in dates like "last quarter" or "this quarter" be specific like "Q2 2021".

**Guidelines**:

- **Leaf Tasks**: These are tasks with no dependencies. They should be simple, specific, and answerable without requiring further decomposition.
  
- **Dependency Structure**: Ensure that a task only depends on subtasks that logically contribute to its completion.

**Example**:

*Bad Example*:

QUESTION:`Apple second quarter 2024 results overview`
TASK GRAPH:
```json
{
  "task_graph": [
    {
      "id": 1,
      "task": "Gather Apple Inc. second quarter 2024 financial results overview",
      "subtasks": [
        2,
        3,
        4,
        5,
        6
      ]
    },
    {
      "id": 2,
      "task": "Find total revenue for Apple in Q2 2024",
      "subtasks": []
    },
    {
      "id": 3,
      "task": "Find net income for Apple in Q2 2024",
      "subtasks": []
    },
    {
      "id": 4,
      "task": "Find earnings per share (EPS) for Apple in Q2 2024",
      "subtasks": []
    },
    {
      "id": 5,
      "task": "Find year-over-year growth percentage for revenue in Q2 2024",
      "subtasks": []
    },
    {
      "id": 6,
      "task": "Find key product performance highlights for Apple in Q2 2024",
      "subtasks": []
    }
  ]
}
```

Why it is bad? Task id 5 is not granular enough. Task id 5 should be broken down into smaller subtasks, such as `Find Apple's revenue for Q2 2023",` and `"Find Apple's revenue for Q2 2024"` and then `Calculate year-over-year growth percentage for revenue in Q2 2024`

{
  "task_graph": [
      ...
    {
      "id": 7,
      "task": "Find Apple's revenue for Q2 2023",
      "subtasks": []
    },
    {
      "id": 8,
      "task": "Find Apple's revenue for Q2 2024",
      "subtasks": []
    },
    {
      "id": 5,
      "task": "Calculate year-over-year growth percentage for revenue in Q2 2024",
      "subtasks": [
        7,
        8
      ]
      ...
    }
  ]
}

today's date is %s
""" % datetime.now().strftime("%Y-%m-%d")

def task_planner(question: str) -> TaskPlan:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",  # Make sure this is a valid model name
        temperature=0,
        response_model=TaskPlan,
        messages=messages,
        max_tokens=1000,
    )

    # The completion object should already be a TaskPlan instance
    return completion


if __name__ == "__main__":
    import time
    import asyncio
    from rich import print

    start_time = time.time()
    plan = task_planner("If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?")

    # Optionally set a custom parallelism limit
    plan.parallelism_limit = 1

    print(plan.model_dump_json(indent=2))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print('Executing Tasks')
    start = time.time()
    asyncio.run(plan.execute())
    end_time = time.time()
    print(f'Tasks executed in {end_time - start:.2f} seconds')
