"""
Aisupea Autonomous Agent System

Full AI agent architecture with planning, reasoning, reflection, and task management.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from ..memory import ContextMemory, TaskMemory, SessionMemory, VectorSimilarityMemory
from ..inference import InferenceEngine
from ..tools import ToolSystem


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    LEARNING = "learning"


class Goal:
    """Represents an agent goal."""

    def __init__(self, description: str, priority: int = 1, deadline: Optional[float] = None):
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.achieved = False
        self.created_at = self._get_timestamp()

    def _get_timestamp(self) -> float:
        import time
        return time.time()

    def __repr__(self) -> str:
        return f"Goal(description='{self.description}', priority={self.priority}, achieved={self.achieved})"


class Task:
    """Represents a task to be executed."""

    def __init__(self, description: str, goal_id: Optional[str] = None, dependencies: List[str] = None):
        self.id = self._generate_id()
        self.description = description
        self.goal_id = goal_id
        self.dependencies = dependencies or []
        self.status = "pending"  # pending, in_progress, completed, failed
        self.result = None
        self.created_at = self._get_timestamp()
        self.completed_at = None

    def _generate_id(self) -> str:
        import time
        return f"task_{int(time.time() * 1000)}"

    def _get_timestamp(self) -> float:
        import time
        return time.time()

    def complete(self, result: Any = None):
        """Mark task as completed."""
        self.status = "completed"
        self.result = result
        self.completed_at = self._get_timestamp()

    def fail(self, error: str):
        """Mark task as failed."""
        self.status = "failed"
        self.result = error
        self.completed_at = self._get_timestamp()

    def __repr__(self) -> str:
        return f"Task(id='{self.id}', description='{self.description}', status='{self.status}')"


class PlanningModule:
    """Handles task planning and goal decomposition."""

    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine

    def create_plan(self, goal: Goal, context: str) -> List[Task]:
        """
        Create a plan to achieve a goal.

        Args:
            goal: Goal to achieve
            context: Current context information

        Returns:
            List of tasks to execute
        """
        # Use inference engine to generate plan
        prompt = f"""
        Goal: {goal.description}
        Context: {context}

        Create a step-by-step plan to achieve this goal. Break it down into specific, actionable tasks.
        Format your response as a numbered list of tasks.
        """

        plan_text = self.inference_engine.generate(prompt, max_length=200)

        # Parse plan into tasks
        tasks = []
        lines = plan_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Extract task description
                task_desc = line.lstrip('0123456789.- ').strip()
                if task_desc:
                    task = Task(task_desc, goal_id=id(goal))
                    tasks.append(task)

        return tasks

    def prioritize_tasks(self, tasks: List[Task], goals: List[Goal]) -> List[Task]:
        """Prioritize tasks based on goal priorities and dependencies."""
        # Simple prioritization: sort by goal priority, then by creation time
        goal_priorities = {id(goal): goal.priority for goal in goals}

        def task_key(task):
            goal_priority = goal_priorities.get(task.goal_id, 0)
            return (-goal_priority, task.created_at)

        return sorted(tasks, key=task_key)


class ReasoningEngine:
    """Handles logical reasoning and decision making."""

    def __init__(self, inference_engine: InferenceEngine, task_memory: TaskMemory):
        self.inference_engine = inference_engine
        self.task_memory = task_memory

    def analyze_situation(self, context: str, available_actions: List[str]) -> Dict[str, Any]:
        """
        Analyze current situation and recommend actions.

        Args:
            context: Current context
            available_actions: List of possible actions

        Returns:
            Analysis with recommended actions and reasoning
        """
        prompt = f"""
        Current situation: {context}
        Available actions: {', '.join(available_actions)}

        Analyze this situation and recommend the best course of action.
        Consider:
        1. What is the current goal or problem?
        2. What actions are most likely to succeed?
        3. What are the potential risks?
        4. What information is missing?

        Provide your analysis and recommendation.
        """

        analysis = self.inference_engine.generate(prompt, max_length=300)

        return {
            'analysis': analysis,
            'recommended_actions': available_actions[:3],  # Simplified
            'confidence': 0.8  # Placeholder
        }

    def learn_from_experience(self, task: Task, success: bool, outcome: str):
        """Learn from task execution experience."""
        self.task_memory.add_task(
            task_description=task.description,
            outcome=outcome,
            success=success,
            metadata={'task_id': task.id}
        )

    def predict_outcome(self, action: str, context: str) -> Tuple[str, float]:
        """Predict outcome of an action."""
        # Search similar past experiences
        similar_tasks = self.task_memory.search_similar_tasks(action)

        if similar_tasks:
            # Calculate success rate from similar tasks
            success_count = sum(1 for task in similar_tasks if task['success'])
            success_rate = success_count / len(similar_tasks)

            # Generate prediction based on past outcomes
            outcomes = [task['outcome'] for task in similar_tasks if task['success']]
            predicted_outcome = outcomes[0] if outcomes else "Unknown outcome"

            return predicted_outcome, success_rate

        return "No similar experiences found", 0.5


class ReflectionModule:
    """Handles self-reflection and improvement."""

    def __init__(self, inference_engine: InferenceEngine, session_memory: SessionMemory):
        self.inference_engine = inference_engine
        self.session_memory = session_memory

    def reflect_on_performance(self, recent_tasks: List[Task]) -> Dict[str, Any]:
        """
        Reflect on recent performance and identify improvements.

        Args:
            recent_tasks: List of recently completed tasks

        Returns:
            Reflection analysis
        """
        if not recent_tasks:
            return {'insights': [], 'improvements': []}

        # Summarize recent performance
        successful_tasks = [t for t in recent_tasks if t.status == 'completed']
        failed_tasks = [t for t in recent_tasks if t.status == 'failed']

        success_rate = len(successful_tasks) / len(recent_tasks) if recent_tasks else 0

        prompt = f"""
        Recent performance summary:
        - Total tasks: {len(recent_tasks)}
        - Successful: {len(successful_tasks)}
        - Failed: {len(failed_tasks)}
        - Success rate: {success_rate:.2%}

        Successful tasks:
        {chr(10).join(f"- {t.description}" for t in successful_tasks[:5])}

        Failed tasks:
        {chr(10).join(f"- {t.description}: {t.result}" for t in failed_tasks[:5])}

        Reflect on this performance and suggest improvements.
        What patterns do you notice? What could be done better?
        """

        reflection = self.inference_engine.generate(prompt, max_length=300)

        return {
            'success_rate': success_rate,
            'reflection': reflection,
            'insights': self._extract_insights(reflection),
            'improvements': self._extract_improvements(reflection)
        }

    def _extract_insights(self, reflection: str) -> List[str]:
        """Extract insights from reflection text."""
        # Simple extraction - look for sentences with insight keywords
        sentences = reflection.split('.')
        insights = []
        insight_keywords = ['notice', 'pattern', 'trend', 'learn', 'realize']

        for sentence in sentences:
            if any(kw in sentence.lower() for kw in insight_keywords):
                insights.append(sentence.strip())

        return insights

    def _extract_improvements(self, reflection: str) -> List[str]:
        """Extract improvement suggestions from reflection text."""
        sentences = reflection.split('.')
        improvements = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in ['should', 'could', 'improve', 'better', 'change']):
                improvements.append(sentence.strip())

        return improvements


class GoalManager:
    """Manages agent goals and priorities."""

    def __init__(self):
        self.goals: List[Goal] = []
        self.completed_goals: List[Goal] = []

    def add_goal(self, description: str, priority: int = 1, deadline: Optional[float] = None) -> Goal:
        """Add a new goal."""
        goal = Goal(description, priority, deadline)
        self.goals.append(goal)
        return goal

    def get_active_goals(self) -> List[Goal]:
        """Get currently active goals."""
        return [g for g in self.goals if not g.achieved]

    def complete_goal(self, goal: Goal):
        """Mark goal as completed."""
        goal.achieved = True
        self.goals.remove(goal)
        self.completed_goals.append(goal)

    def prioritize_goals(self) -> List[Goal]:
        """Return goals sorted by priority and deadline."""
        def goal_key(goal):
            # Higher priority first, then earlier deadline
            deadline_factor = goal.deadline or float('inf')
            return (-goal.priority, deadline_factor)

        return sorted(self.get_active_goals(), key=goal_key)


class TaskManager:
    """Manages task execution and dependencies."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: List[Task] = []

    def add_task(self, task: Task):
        """Add a task to the manager."""
        self.tasks[task.id] = task

    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute (respecting dependencies)."""
        # Find pending tasks with satisfied dependencies
        available_tasks = []

        for task in self.tasks.values():
            if task.status == 'pending' and self._dependencies_satisfied(task):
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Return highest priority task (simplified - just first available)
        return available_tasks[0]

    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != 'completed':
                return False
        return True

    def execute_task(self, task: Task, executor: Callable[[Task], Any]) -> bool:
        """
        Execute a task using the provided executor function.

        Args:
            task: Task to execute
            executor: Function that takes a task and returns result

        Returns:
            True if successful, False otherwise
        """
        try:
            task.status = 'in_progress'
            result = executor(task)
            task.complete(result)
            return True
        except Exception as e:
            task.fail(str(e))
            return False

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [t for t in self.tasks.values() if t.status == 'completed']

    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [t for t in self.tasks.values() if t.status == 'failed']


class DecisionEngine:
    """Makes decisions about what actions to take."""

    def __init__(self, reasoning_engine: ReasoningEngine, goal_manager: GoalManager):
        self.reasoning_engine = reasoning_engine
        self.goal_manager = goal_manager

    def decide_next_action(self, context: str, available_actions: List[str]) -> str:
        """
        Decide what action to take next.

        Args:
            context: Current context
            available_actions: List of possible actions

        Returns:
            Chosen action
        """
        if not available_actions:
            return "wait"

        # Get current goals
        active_goals = self.goal_manager.get_active_goals()

        if not active_goals:
            return "explore"  # Default action when no goals

        # Analyze situation
        analysis = self.reasoning_engine.analyze_situation(context, available_actions)

        # Choose action based on analysis
        recommended_actions = analysis.get('recommended_actions', available_actions)

        # Return first recommended action
        return recommended_actions[0] if recommended_actions else available_actions[0]


class AutonomousAgent:
    """
    Main autonomous agent class that coordinates all modules.
    """

    def __init__(self, inference_engine: InferenceEngine, tool_system: ToolSystem):
        # Core modules
        self.inference_engine = inference_engine
        self.tool_system = tool_system

        # Memory systems
        self.context_memory = ContextMemory()
        self.task_memory = TaskMemory()
        self.session_memory = SessionMemory()
        self.vector_memory = VectorSimilarityMemory(embed_dim=768)  # Assuming 768-dim embeddings

        # Agent modules
        self.planning_module = PlanningModule(inference_engine)
        self.reasoning_engine = ReasoningEngine(inference_engine, self.task_memory)
        self.reflection_module = ReflectionModule(inference_engine, self.session_memory)
        self.goal_manager = GoalManager()
        self.task_manager = TaskManager()
        self.decision_engine = DecisionEngine(self.reasoning_engine, self.goal_manager)

        # State
        self.state = AgentState.IDLE
        self.current_task = None

    def set_goal(self, description: str, priority: int = 1) -> Goal:
        """Set a new goal for the agent."""
        goal = self.goal_manager.add_goal(description, priority)

        # Create plan for the goal
        context = self.context_memory.get_context()
        tasks = self.planning_module.create_plan(goal, context)

        # Add tasks to task manager
        for task in tasks:
            self.task_manager.add_task(task)

        return goal

    def run_step(self) -> Dict[str, Any]:
        """
        Execute one step of the agent loop.

        Returns:
            Step result information
        """
        result = {
            'state': self.state.value,
            'action_taken': None,
            'task_completed': None,
            'reflection': None
        }

        # Get current context
        context = self._get_current_context()

        # Decision making
        available_actions = self._get_available_actions()
        next_action = self.decision_engine.decide_next_action(context, available_actions)

        result['action_taken'] = next_action

        # Execute action
        if next_action == "execute_task":
            self._execute_next_task()
        elif next_action == "reflect":
            reflection = self.reflection_module.reflect_on_performance(
                self.task_manager.get_completed_tasks()[-10:]  # Last 10 tasks
            )
            result['reflection'] = reflection
        elif next_action == "plan":
            # Re-plan based on current situation
            active_goals = self.goal_manager.get_active_goals()
            if active_goals:
                goal = active_goals[0]  # Focus on highest priority goal
                new_tasks = self.planning_module.create_plan(goal, context)
                for task in new_tasks:
                    self.task_manager.add_task(task)

        # Update state
        self._update_state()

        return result

    def _get_current_context(self) -> str:
        """Get current context from memory systems."""
        context_parts = []

        # Conversation context
        conv_context = self.context_memory.get_context(max_tokens=500)
        if conv_context:
            context_parts.append(f"Conversation:\n{conv_context}")

        # Current goals
        active_goals = self.goal_manager.get_active_goals()
        if active_goals:
            goals_text = "\n".join(f"- {g.description} (priority: {g.priority})" for g in active_goals)
            context_parts.append(f"Active Goals:\n{goals_text}")

        # Current task
        if self.current_task:
            context_parts.append(f"Current Task: {self.current_task.description}")

        return "\n\n".join(context_parts)

    def _get_available_actions(self) -> List[str]:
        """Get list of available actions."""
        actions = ["execute_task", "reflect", "plan", "wait"]

        # Add tool-based actions
        tool_actions = [f"use_tool_{tool_name}" for tool_name in self.tool_system.get_available_tools()]
        actions.extend(tool_actions)

        return actions

    def _execute_next_task(self):
        """Execute the next available task."""
        task = self.task_manager.get_next_task()
        if task:
            self.current_task = task

            # Execute task using tools
            success = self.task_manager.execute_task(task, self._execute_task_function)

            # Learn from experience
            self.reasoning_engine.learn_from_experience(
                task, success, task.result or "Task completed successfully"
            )

            self.current_task = None

    def _execute_task_function(self, task: Task) -> Any:
        """Execute a task using available tools."""
        # Use inference to determine which tool to use
        prompt = f"""
        Task: {task.description}

        Available tools: {', '.join(self.tool_system.get_available_tools())}

        Which tool should be used to complete this task? Provide the tool name and any required parameters.
        """

        tool_decision = self.inference_engine.generate(prompt, max_length=100)

        # Parse tool decision (simplified)
        # In practice, this would parse the response to extract tool name and parameters

        # For now, assume we can execute the task
        return f"Executed task: {task.description}"

    def _update_state(self):
        """Update agent state based on current situation."""
        if self.current_task:
            self.state = AgentState.EXECUTING
        elif self.task_manager.get_next_task():
            self.state = AgentState.PLANNING
        else:
            self.state = AgentState.IDLE

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'state': self.state.value,
            'active_goals': len(self.goal_manager.get_active_goals()),
            'pending_tasks': len([t for t in self.task_manager.tasks.values() if t.status == 'pending']),
            'completed_tasks': len(self.task_manager.get_completed_tasks()),
            'current_task': self.current_task.description if self.current_task else None
        }