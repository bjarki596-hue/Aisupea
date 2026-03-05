"""
Aisupea Example: Transformer Model

Demonstrates how to create and use a transformer model for text generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import Transformer
from tokenization.tokenizer import Tokenizer
from inference.engine import InferenceEngine


def main():
    print("🤖 Aisupea Transformer Example")
    print("=" * 50)

    # Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = Tokenizer(vocab_size=1000)

    # Train on sample text
    sample_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Natural language processing with transformers.",
        "This is a simple example of text tokenization."
    ]

    print("Training tokenizer on sample texts...")
    tokenizer.train(sample_texts)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Create transformer model
    print("\n2. Creating transformer model...")
    model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_seq_len=50
    )

    print("Model created with:")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Embedding dimension: {model.embed_dim}")
    print(f"  - Number of heads: {model.num_heads}")
    print(f"  - Number of layers: {model.num_layers}")

    # Create inference engine
    print("\n3. Creating inference engine...")
    inference_engine = InferenceEngine(model, tokenizer)

    # Test tokenization
    print("\n4. Testing tokenization:")
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")

    # Test generation (this will be random since model is untrained)
    print("\n5. Testing text generation:")
    prompt = "Hello"

    print(f"Prompt: '{prompt}'")
    print("Generating response...")

    try:
        response = inference_engine.generate(prompt, max_length=20, temperature=0.8)
        print(f"Generated: '{response}'")
    except Exception as e:
        print(f"Generation failed (expected for untrained model): {e}")

    print("\n✅ Transformer example completed!")
    print("\nNote: The model is untrained, so generation will be random.")
    print("To train the model, use the training utilities in aisupea.training")


if __name__ == "__main__":
    main()