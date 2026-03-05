"""
Aisupea Brain Integration Example

Demonstrates the brain module coordinating reasoning and thinking.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from brain import Brain

def main():
    print("Aisupea Brain Integration Demo")
    print("=" * 40)

    # Initialize brain
    brain = Brain()

    # Register some mock modules
    class MockModule:
        def __init__(self, name):
            self.name = name

        def process(self, prompt):
            return f"{self.name} processed: {prompt[:50]}..."

    brain.register_module("memory", MockModule("Memory"))
    brain.register_module("tools", MockModule("Tools"))
    brain.register_module("interface", MockModule("Interface"))

    # Test thinking
    print("\n1. Testing Thinking Process:")
    thought_result = brain.think("What is the meaning of artificial intelligence?")
    print(f"Thoughts generated: {len(thought_result['thoughts'])}")
    print(f"Meta-analysis: {thought_result['meta_analysis']['quality_score']:.2f}")
    print(f"Consciousness: {thought_result['consciousness_state']}")

    # Test reasoning
    print("\n2. Testing Reasoning Process:")
    reasoning_result = brain.reason("If all humans are mortal and Socrates is human, then Socrates is mortal")
    print(f"Reasoning result: {reasoning_result[:100]}...")

    # Test decision making
    print("\n3. Testing Decision Making:")
    decision = brain.decide(["Option A", "Option B", "Option C"])
    print(f"Decision: {decision}")

    # Set a goal
    print("\n4. Setting and Processing Goals:")
    brain.set_goal("Learn about machine learning")
    brain.process_goal()
    print("Goal processed")

    # Learn from experience
    print("\n5. Learning from Experience:")
    brain.learn({
        "lesson": "Practice makes perfect",
        "topic": "learning",
        "outcome": "positive"
    })

    # Get cognitive state
    print("\n6. Cognitive State:")
    state = brain.get_cognitive_state()
    print(f"Awareness: {state['awareness']:.2f}")
    print(f"Active goals: {state['active_goals']}")
    print(f"Knowledge areas: {state['knowledge_areas']}")

    # Shutdown
    brain.shutdown()
    print("\nBrain shutdown complete")

if __name__ == "__main__":
    main()