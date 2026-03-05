"""
Aisupea Reasoning Module

Advanced reasoning engine with logical inference, probabilistic reasoning,
causal analysis, and decision theory.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
import math
import random
import time

class LogicalReasoner:
    """Handles logical inference and deduction."""

    def __init__(self):
        self.facts = set()
        self.rules = []
        self.inferences = []

    def add_fact(self, fact: str):
        """Add a factual statement."""
        self.facts.add(fact)

    def add_rule(self, premise: List[str], conclusion: str):
        """Add a logical rule (if premises then conclusion)."""
        self.rules.append((premise, conclusion))

    def infer(self, max_steps: int = 10) -> List[str]:
        """Perform logical inference."""
        new_inferences = []
        for _ in range(max_steps):
            for premises, conclusion in self.rules:
                if all(p in self.facts for p in premises) and conclusion not in self.facts:
                    self.facts.add(conclusion)
                    new_inferences.append(conclusion)
                    self.inferences.append(f"From {premises} inferred {conclusion}")
        return new_inferences

    def query(self, statement: str) -> bool:
        """Check if a statement can be proven."""
        return statement in self.facts

class ProbabilisticReasoner:
    """Handles probabilistic reasoning and uncertainty."""

    def __init__(self):
        self.probabilities = {}
        self.dependencies = defaultdict(list)

    def set_probability(self, event: str, prob: float):
        """Set probability of an event."""
        self.probabilities[event] = prob

    def add_dependency(self, cause: str, effect: str, strength: float):
        """Add causal dependency."""
        self.dependencies[cause].append((effect, strength))

    def compute_probability(self, event: str) -> float:
        """Compute probability considering dependencies."""
        if event in self.probabilities:
            return self.probabilities[event]

        # Simple Bayesian inference
        prob = 0.5  # Default
        causes = [c for c, _ in self.dependencies.items() if any(e == event for e, _ in self.dependencies[c])]
        if causes:
            prob = sum(self.probabilities.get(c, 0.5) for c in causes) / len(causes)

        self.probabilities[event] = prob
        return prob

    def update_beliefs(self, evidence: Dict[str, bool]):
        """Update probabilities based on evidence."""
        for event, observed in evidence.items():
            if observed:
                self.probabilities[event] = min(1.0, self.probabilities.get(event, 0.5) * 1.2)
            else:
                self.probabilities[event] = max(0.0, self.probabilities.get(event, 0.5) * 0.8)

class CausalAnalyzer:
    """Analyzes cause-effect relationships."""

    def __init__(self):
        self.causal_graph = defaultdict(list)
        self.correlations = {}

    def add_causal_link(self, cause: str, effect: str, strength: float = 1.0):
        """Add a causal relationship."""
        self.causal_graph[cause].append((effect, strength))

    def find_causes(self, effect: str) -> List[Tuple[str, float]]:
        """Find potential causes of an effect."""
        causes = []
        for cause, effects in self.causal_graph.items():
            for eff, strength in effects:
                if eff == effect:
                    causes.append((cause, strength))
        return sorted(causes, key=lambda x: x[1], reverse=True)

    def simulate_outcome(self, initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate causal chain from initial conditions."""
        state = initial_conditions.copy()
        visited = set()

        def propagate(cause: str):
            if cause in visited:
                return
            visited.add(cause)
            for effect, strength in self.causal_graph[cause]:
                if effect not in state:
                    state[effect] = strength * state.get(cause, 1.0)
                    propagate(effect)

        for condition in initial_conditions:
            propagate(condition)

        return state

class DecisionEngine:
    """Makes decisions using decision theory."""

    def __init__(self):
        self.utilities = {}
        self.options = defaultdict(list)

    def set_utility(self, outcome: str, value: float):
        """Set utility value for an outcome."""
        self.utilities[outcome] = value

    def add_option(self, decision: str, outcome: str, probability: float):
        """Add possible outcome for a decision."""
        self.options[decision].append((outcome, probability))

    def expected_utility(self, decision: str) -> float:
        """Calculate expected utility of a decision."""
        if decision not in self.options:
            return 0.0

        eu = 0.0
        for outcome, prob in self.options[decision]:
            eu += prob * self.utilities.get(outcome, 0.0)
        return eu

    def best_decision(self, decisions: List[str]) -> str:
        """Find the decision with highest expected utility."""
        if not decisions:
            return ""

        best_dec = max(decisions, key=self.expected_utility)
        return best_dec

class ReasoningEngine:
    """Main reasoning coordinator."""

    def __init__(self):
        self.logical = LogicalReasoner()
        self.probabilistic = ProbabilisticReasoner()
        self.causal = CausalAnalyzer()
        self.decision = DecisionEngine()
        self.reasoning_history = []

    def reason(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive reasoning process."""
        start_time = time.time()

        # Extract problem components
        facts = problem.get('facts', [])
        rules = problem.get('rules', [])
        uncertainties = problem.get('uncertainties', {})
        causal_links = problem.get('causal_links', [])
        decisions = problem.get('decisions', [])

        # Add facts and rules
        for fact in facts:
            self.logical.add_fact(fact)

        for rule in rules:
            self.logical.add_rule(rule['premises'], rule['conclusion'])

        # Perform logical inference
        inferences = self.logical.infer()

        # Handle uncertainties
        for event, prob in uncertainties.items():
            self.probabilistic.set_probability(event, prob)

        # Add causal links
        for link in causal_links:
            self.causal.add_causal_link(link['cause'], link['effect'], link.get('strength', 1.0))

        # Analyze decisions
        best_decision = ""
        if decisions:
            for dec in decisions:
                for outcome in dec.get('outcomes', []):
                    self.decision.add_option(dec['name'], outcome['name'], outcome['probability'])
                    self.decision.set_utility(outcome['name'], outcome['utility'])

            best_decision = self.decision.best_decision([d['name'] for d in decisions])

        # Compile reasoning result
        result = {
            'logical_inferences': inferences,
            'probabilistic_assessments': {k: self.probabilistic.compute_probability(k) for k in uncertainties.keys()},
            'causal_analysis': {eff: self.causal.find_causes(eff) for eff in set(e['effect'] for e in causal_links)},
            'best_decision': best_decision,
            'confidence': self._assess_confidence(inferences, uncertainties),
            'reasoning_time': time.time() - start_time
        }

        self.reasoning_history.append({
            'problem': problem,
            'result': result,
            'timestamp': time.time()
        })

        return result

    def _assess_confidence(self, inferences: List[str], uncertainties: Dict[str, float]) -> float:
        """Assess overall confidence in reasoning."""
        logical_conf = min(1.0, len(inferences) / 10.0)  # More inferences = higher confidence
        prob_conf = 1.0 - (sum(uncertainties.values()) / len(uncertainties)) if uncertainties else 1.0
        return (logical_conf + prob_conf) / 2.0

    def explain_reasoning(self, result: Dict[str, Any]) -> str:
        """Generate explanation of reasoning process."""
        explanation = "Reasoning Analysis:\n"

        if result['logical_inferences']:
            explanation += f"- Logical inferences: {', '.join(result['logical_inferences'])}\n"

        if result['probabilistic_assessments']:
            explanation += f"- Probabilistic assessments: {result['probabilistic_assessments']}\n"

        if result['best_decision']:
            explanation += f"- Recommended decision: {result['best_decision']}\n"

        explanation += f"- Overall confidence: {result['confidence']:.2f}\n"
        explanation += f"- Reasoning time: {result['reasoning_time']:.3f}s\n"

        return explanation

    def learn_from_feedback(self, feedback: Dict[str, Any]):
        """Learn from feedback to improve reasoning."""
        # Simple learning: adjust utilities based on feedback
        if 'outcome_utility' in feedback:
            for outcome, utility in feedback['outcome_utility'].items():
                current = self.decision.utilities.get(outcome, 0.0)
                self.decision.utilities[outcome] = (current + utility) / 2.0

__all__ = ['LogicalReasoner', 'ProbabilisticReasoner', 'CausalAnalyzer', 'DecisionEngine', 'ReasoningEngine']