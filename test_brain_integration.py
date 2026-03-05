#!/usr/bin/env python3
"""
Test script to verify the brain module integration works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_brain_integration():
    """Test the brain module with reasoning, thinking, and knowledge integration."""
    try:
        from brain import Brain
        from reasoning import ReasoningEngine
        from thinking import ThinkingEngine
        from knowledge import KnowledgeBase

        print("✓ Successfully imported all modules")

        # Create brain instance
        brain = Brain()
        print("✓ Brain initialized")

        # Register modules
        reasoning_engine = ReasoningEngine()
        thinking_engine = ThinkingEngine()
        knowledge_base = KnowledgeBase()

        brain.register_module("reasoning", reasoning_engine)
        brain.register_module("thinking", thinking_engine)
        brain.register_module("knowledge", knowledge_base)
        print("✓ Modules registered")

        # Test thinking
        result = brain.think("What is the meaning of life?")
        print(f"✓ Thinking result: {result['thoughts'].keys()}")

        # Test reasoning
        reasoning_result = brain.reason("If all men are mortal and Socrates is a man, then Socrates is mortal")
        print(f"✓ Reasoning result: {reasoning_result[:50]}...")

        # Test decision making
        decision = brain.decide(["Study philosophy", "Study science", "Study art"])
        print(f"✓ Decision made: {decision}")

        # Test cognitive state
        state = brain.get_cognitive_state()
        print(f"✓ Cognitive state: awareness={state['awareness']:.2f}, modules={len(state['modules_registered'])}")

        print("\n🎉 All tests passed! The brain module is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_brain_integration()
    sys.exit(0 if success else 1)