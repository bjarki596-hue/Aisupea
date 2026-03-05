"""
Aisupea Brain Module

Central cognitive architecture coordinating all AI modules.
"""

from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import time
import threading

try:
    from ..reasoning import ReasoningEngine
    from ..thinking import ThinkingEngine
    from ..agent import Agent
    from ..memory import ContextMemory, TaskMemory, SessionMemory
    from ..interface import Logger
except ImportError:
    # Fallback for missing modules
    ReasoningEngine = None
    ThinkingEngine = None
    Agent = None
    ContextMemory = None
    TaskMemory = None
    SessionMemory = None
    Logger = None

class Consciousness:
    """Simulates basic consciousness and self-awareness."""

    def __init__(self):
        self.awareness_level = 0.0
        self.thoughts = []
        self.emotions = defaultdict(float)
        self.attention_focus = None

    def update_awareness(self, input_stimuli: Any):
        """Update awareness based on input."""
        self.awareness_level = min(1.0, self.awareness_level + 0.1)
        self.thoughts.append(f"Processing: {input_stimuli}")
        if len(self.thoughts) > 100:  # Limit memory
            self.thoughts = self.thoughts[-50:]

    def set_attention(self, focus: str):
        """Set attention focus."""
        self.attention_focus = focus
        self.emotions['focus'] = 0.8

    def reflect(self) -> str:
        """Generate self-reflection."""
        return f"Current awareness: {self.awareness_level:.2f}, Focus: {self.attention_focus}, Recent thoughts: {len(self.thoughts)}"

class MetaReasoning:
    """Handles higher-order reasoning about reasoning processes."""

    def __init__(self):
        self.reasoning_history = []
        self.confidence_levels = []
        self.biases_detected = []

    def analyze_reasoning(self, reasoning_output: Any) -> Dict[str, Any]:
        """Analyze the quality of reasoning."""
        confidence = self._assess_confidence(reasoning_output)
        self.reasoning_history.append(reasoning_output)
        self.confidence_levels.append(confidence)

        analysis = {
            'quality_score': confidence,
            'suggestions': self._generate_improvements(reasoning_output),
            'biases': self._detect_biases(reasoning_output)
        }

        return analysis

    def _assess_confidence(self, output: Any) -> float:
        """Assess confidence in reasoning output."""
        if isinstance(output, dict):
            # Check for logical consistency, evidence strength, etc.
            confidence = 0.5
            if 'logical_inferences' in output and output['logical_inferences']:
                confidence += 0.2
            if 'probabilistic_assessments' in output:
                confidence += 0.1
            if output.get('confidence', 0) > 0.7:
                confidence += 0.2
            return min(1.0, confidence)
        return 0.5

    def _generate_improvements(self, output: Any) -> List[str]:
        """Generate suggestions for improving reasoning."""
        suggestions = []
        if isinstance(output, dict):
            if not output.get('logical_inferences'):
                suggestions.append("Consider adding more logical inference steps")
            if 'best_decision' not in output:
                suggestions.append("Evaluate decision options more thoroughly")
            if output.get('confidence', 1.0) < 0.6:
                suggestions.append("Gather more evidence to increase confidence")
        return suggestions

    def _detect_biases(self, output: Any) -> List[str]:
        """Detect potential cognitive biases."""
        biases = []
        # Simple bias detection
        if isinstance(output, dict) and 'best_decision' in output:
            if len(self.reasoning_history) > 5:
                recent_decisions = [h.get('best_decision', '') for h in self.reasoning_history[-5:]]
                if len(set(recent_decisions)) == 1:
                    biases.append("Confirmation bias detected - considering same decisions repeatedly")
        return biases

    def improve_reasoning(self) -> str:
        """Suggest reasoning improvements."""
        avg_confidence = sum(self.confidence_levels) / len(self.confidence_levels) if self.confidence_levels else 0.5
        return f"Reasoning improvement plan: Average confidence {avg_confidence:.2f}. Focus on {', '.join(set(self._generate_improvements({})))}"

class Brain:
    """Central cognitive architecture coordinating all AI modules."""

    def __init__(self):
        self.consciousness = Consciousness()
        self.meta_reasoning = MetaReasoning()
        self.modules = {}
        self.thinking_processes = []
        self.reasoning_engine = ReasoningEngine() if ReasoningEngine else None
        self.thinking_engine = ThinkingEngine() if ThinkingEngine else None
        self.agent = Agent() if Agent else None
        self.decision_maker = None
        self.knowledge_base = {}
        self.goals = []

        # Memory systems
        self.context_memory = ContextMemory() if ContextMemory else None
        self.task_memory = TaskMemory() if TaskMemory else None
        self.session_memory = SessionMemory() if SessionMemory else None

        # Logging
        self.logger = Logger("brain.log") if Logger else None

    def register_module(self, name: str, module: Any):
        """Register an AI module for coordination."""
        self.modules[name] = module
        if self.logger:
            self.logger.info(f"Registered module: {name}")
        else:
            print(f"Registered module: {name}")

    def think(self, prompt: str) -> Dict[str, Any]:
        """Main thinking process coordinating modules."""
        # Update consciousness
        self.consciousness.update_awareness(prompt)
        self.consciousness.set_attention("processing_input")

        # Coordinate with modules
        thoughts = {}
        for name, module in self.modules.items():
            if hasattr(module, 'process'):
                try:
                    thoughts[name] = module.process(prompt)
                except Exception as e:
                    thoughts[name] = f"Error: {str(e)}"

        # Use thinking engine if available
        if self.thinking_engine:
            thinking_result = self.thinking_engine.think({
                'type': 'general',
                'content': prompt
            })
            thoughts['cognitive'] = thinking_result

        # Meta-reasoning
        analysis = self.meta_reasoning.analyze_reasoning(thoughts)

        # Store thinking process
        self.thinking_processes.append({
            'prompt': prompt,
            'thoughts': thoughts,
            'analysis': analysis,
            'timestamp': time.time()
        })

        # Update consciousness with results
        self.consciousness.update_awareness(f"Thought process completed with {len(thoughts)} module responses")

        return {
            'thoughts': thoughts,
            'meta_analysis': analysis,
            'consciousness_state': self.consciousness.reflect()
        }

    def reason(self, problem: str) -> str:
        """Reasoning process with meta-analysis."""
        if self.reasoning_engine:
            reasoning_result = self.reasoning_engine.reason({
                'facts': [problem],
                'rules': [],
                'uncertainties': {},
                'causal_links': [],
                'decisions': []
            })
            meta_analysis = self.meta_reasoning.analyze_reasoning(reasoning_result)
            explanation = self.reasoning_engine.explain_reasoning(reasoning_result)
            return f"{explanation} | Meta-analysis: {meta_analysis}"
        else:
            base_reasoning = f"Basic reasoning on: {problem}"
            meta_analysis = self.meta_reasoning.analyze_reasoning(base_reasoning)
            return f"{base_reasoning} | Meta-analysis: {meta_analysis}"

    def decide(self, options: List[str]) -> str:
        """Decision making process."""
        if not options:
            return "No options provided"

        if self.reasoning_engine and hasattr(self.reasoning_engine, 'decision'):
            # Create decision problem
            decisions = [{
                'name': f'option_{i}',
                'outcomes': [{'name': opt, 'probability': 1.0/len(options), 'utility': 0.5} for opt in options]
            } for i in range(len(options))]

            problem = {
                'decisions': decisions
            }
            result = self.reasoning_engine.reason(problem)
            best_decision = result.get('best_decision', '')
            if best_decision and best_decision.startswith('option_'):
                idx = int(best_decision.split('_')[1])
                return options[idx] if idx < len(options) else options[0]

        # Fallback: random choice with consciousness update
        choice = options[0]  # Default to first
        self.consciousness.update_awareness(f"Made decision: {choice}")
        return choice

    def set_goal(self, goal: str):
        """Set a cognitive goal."""
        self.goals.append({
            'description': goal,
            'status': 'active',
            'created': time.time()
        })
        self.consciousness.set_attention(goal)

    def process_goal(self):
        """Process active goals."""
        for goal in self.goals:
            if goal['status'] == 'active':
                # Use reasoning to work towards goal
                if self.reasoning_engine:
                    reasoning = self.reasoning_engine.reason({
                        'facts': [f"Goal: {goal['description']}"],
                        'rules': [],
                        'uncertainties': {},
                        'causal_links': [],
                        'decisions': []
                    })
                    goal['progress'] = reasoning.get('confidence', 0.0)

    def learn(self, experience: Dict[str, Any]):
        """Learn from experience."""
        if self.thinking_engine:
            self.thinking_engine.learn_from_experience(experience)

        # Update knowledge base
        if 'lesson' in experience:
            key = experience.get('topic', 'general')
            if key not in self.knowledge_base:
                self.knowledge_base[key] = []
            self.knowledge_base[key].append(experience['lesson'])

        self.consciousness.update_awareness(f"Learned: {experience}")

    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        return {
            'awareness': self.consciousness.awareness_level,
            'attention': self.consciousness.attention_focus,
            'recent_thoughts': len(self.thinking_processes),
            'modules_registered': list(self.modules.keys()),
            'reasoning_confidence': sum(self.meta_reasoning.confidence_levels) / len(self.meta_reasoning.confidence_levels) if self.meta_reasoning.confidence_levels else 0.0,
            'active_goals': len([g for g in self.goals if g['status'] == 'active']),
            'knowledge_areas': len(self.knowledge_base)
        }

    def shutdown(self):
        """Graceful shutdown of cognitive processes."""
        self.consciousness.update_awareness("System shutdown initiated")
        if self.logger:
            self.logger.info("Brain shutdown complete")
        else:
            print("Brain shutdown complete")

__all__ = ['Brain', 'Consciousness', 'MetaReasoning']