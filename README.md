# 🤖 Aisupea

A modular, lightweight AI framework written entirely in pure Python with minimal dependencies. Designed for building full autonomous AI systems that can run in constrained environments.

## 🌟 Features

- **Pure Python**: No external dependencies beyond the standard library
- **Modular Architecture**: Clean separation of concerns with well-defined modules
- **Constrained Environment Ready**: Minimal memory footprint and computational requirements
- **Full AI Pipeline**: From tensor operations to autonomous agents
- **Extensible Design**: Easy to add new components and customize behavior

## 🏗️ Architecture

```
aisupea/
├── core/           # Core tensor engine and fundamental operations
├── torch_api/      # Torch-style API with nn.Module system
├── numpy_compat/   # Lightweight NumPy replacement
├── models/         # Transformer architecture and other models
├── tokenization/   # Text tokenization system
├── inference/      # Text generation engine
├── memory/         # Multiple memory systems for agents
├── agent/          # Autonomous agent architecture
├── tools/          # Tool system for environment interaction
├── debugging/      # Self-debugging and error analysis
├── interface/      # CLI interfaces and session management
├── training/       # Training utilities and loops
├── brain/          # Central cognitive architecture and consciousness
├── reasoning/      # Advanced reasoning engines (logical, probabilistic, causal)
├── thinking/       # Higher-level cognitive processes (abstract, creative, intuitive)
├── knowledge/      # Knowledge representation and graph-based storage
└── utils/          # General utilities and helpers
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aisupea.git
cd aisupea

# No installation required - just import and use!
```

### Basic Usage

```python
from aisupea.core import Tensor
from aisupea.torch_api import matmul, softmax

# Create tensors
a = Tensor([[1, 2], [3, 4]], 'float')
b = Tensor([[5, 6], [7, 8]], 'float')

# Matrix multiplication
c = matmul(a, b)

# Apply softmax
d = softmax(c, dim=1)

print(d)
```

### Text Generation Example

```python
from aisupea.models.transformer import Transformer
from aisupea.tokenization.tokenizer import Tokenizer
from aisupea.inference.engine import InferenceEngine

# Create tokenizer and train on some text
tokenizer = Tokenizer(vocab_size=1000)
tokenizer.train(["Hello world", "This is a test"])

# Create transformer model
model = Transformer(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    hidden_dim=256,
    max_seq_len=50
)

# Create inference engine
engine = InferenceEngine(model, tokenizer)

# Generate text
response = engine.generate("Hello", max_length=20)
print(response)
```

### Brain Integration Example

```python
from aisupea.brain import Brain

# Create brain instance
brain = Brain()

# Register modules
brain.register_module("memory", memory_system)
brain.register_module("tools", tool_system)

# Think about a problem
result = brain.think("How can I improve my problem-solving skills?")
print(f"Generated {len(result['thoughts'])} thoughts")
print(f"Meta-analysis confidence: {result['meta_analysis']['quality_score']}")

# Make a decision
decision = brain.decide(["Study logic", "Practice puzzles", "Learn algorithms"])
print(f"Brain decided: {decision}")

# Set and work toward goals
brain.set_goal("Master advanced reasoning techniques")
brain.process_goal()
```

## 📚 Core Components

### 1. Core Math Engine (`core/`)

Custom tensor implementation with:
- Multidimensional arrays with efficient nested list storage
- Shape tracking and validation
- Reshape, transpose, and view operations
- Matrix multiplication with broadcasting
- Element-wise operations and reductions
- Softmax, argmax, and other activation functions

### 2. Torch-style API (`torch_api/`)

PyTorch-compatible interface:
- Tensor creation functions (`zeros`, `ones`, `arange`)
- Neural network modules (`Linear`, `LayerNorm`, `Embedding`)
- Functional API for operations
- `nn.Module` base class with parameter management

### 3. Lightweight NumPy (`numpy_compat/`)

Minimal ndarray replacement:
- Array creation and manipulation
- Vector/matrix operations
- Dot product and normalization
- Small linear algebra utilities

### 4. Transformer Architecture (`models/`)

Complete transformer implementation:
- Rotary Positional Embeddings (RoPE)
- Multi-head self-attention
- Feed-forward networks with GELU
- Layer normalization and residual connections
- KV cache for efficient inference

### 5. Tokenization System (`tokenization/`)

Text processing utilities:
- Word and punctuation tokenization
- Vocabulary training and management
- Encoding/decoding between text and token IDs
- Configurable vocabulary size

### 6. Inference Engine (`inference/`)

Text generation capabilities:
- Greedy decoding
- Temperature sampling
- Top-k and top-p sampling
- Token streaming
- Incremental inference with KV cache

### 7. Memory Systems (`memory/`)

Multiple memory types for agents:
- **Vector Similarity Memory**: Embedding-based retrieval
- **Context Memory**: Conversation history with sliding window
- **Task Memory**: Completed tasks and outcomes for learning
- **Session Memory**: Temporary storage during sessions

### 8. Autonomous Agent (`agent/`)

Full agent architecture:
- **Planning Module**: Goal decomposition and task planning
- **Reasoning Engine**: Logical reasoning and decision making
- **Reflection Module**: Self-improvement through performance analysis
- **Goal Manager**: Goal setting and prioritization
- **Task Manager**: Task execution and dependency management
- **Decision Engine**: Action selection based on context

### 9. Tool System (`tools/`)

Environment interaction tools:
- **Python Execution Tool**: Run Python code safely
- **Bash Execution Tool**: Execute shell commands
- **Filesystem Tool**: File and directory operations
- **Project Analyzer**: Code structure and dependency analysis
- **Code Search Tool**: Text search across codebase
- **Command Router**: Natural language command interpretation

### 10. Debugging System (`debugging/`)

Self-debugging capabilities:
- Python traceback parsing
- Missing import detection
- Automatic error fixes when safe
- Code analysis and suggestions

### 11. Interface System (`interface/`)

User interaction interfaces:
- **CLI Chat Interface**: Interactive conversation mode
- **Command Interface**: Direct tool execution
- **Session Manager**: Conversation persistence
- **Logger**: Structured logging system
- **Progress Tracker**: Long-running operation monitoring

### 12. Training System (`training/`)

Model training utilities:
- Training loops and data loading
- Learning rate scheduling
- Checkpoint management
- Configuration management

### 13. Brain System (`brain/`)

Central cognitive architecture:
- **Consciousness Simulation**: Awareness levels and self-reflection
- **Meta-Reasoning**: Higher-order reasoning about reasoning processes
- **Module Coordination**: Orchestrates all AI components
- **Cognitive State Management**: Tracks thinking processes and goals
- **Learning Integration**: Incorporates experiences into knowledge

### 14. Reasoning Engine (`reasoning/`)

Advanced reasoning capabilities:
- **Logical Reasoner**: Deduction and inference from facts and rules
- **Probabilistic Reasoner**: Uncertainty handling and Bayesian updates
- **Causal Analyzer**: Cause-effect relationship modeling
- **Decision Engine**: Expected utility maximization
- **Comprehensive Reasoning**: Multi-paradigm problem solving

### 15. Thinking System (`thinking/`)

Higher-level cognitive processes:
- **Abstract Thinker**: Concept formation and analogy generation
- **Creative Engine**: Idea generation and brainstorming
- **Intuitive Processor**: Pattern recognition and gut feelings
- **Metacognitive Monitor**: Self-regulation and strategy selection
- **Parallel Thinking**: Multi-threaded cognitive processing

### 16. Knowledge Base (`knowledge/`)

Knowledge representation and management:
- **Knowledge Graph**: Graph-based storage with relationships
- **Concept Storage**: Hierarchical concept organization
- **Query System**: Semantic search and retrieval
- **Learning from Text**: Automatic knowledge extraction
- **Persistence**: Save/load knowledge bases

## 🎯 Use Cases

- **Edge AI**: Run AI models on resource-constrained devices
- **Educational AI**: Learn AI concepts without complex dependencies
- **Research Prototyping**: Rapid experimentation with AI architectures
- **Autonomous Systems**: Build AI agents for various applications
- **Offline AI**: AI systems that work without internet connectivity

## 🔧 Development

### Running Examples

```bash
# Basic tensor operations
python examples/tensor_operations.py

# Transformer model usage
python examples/transformer_example.py

# Brain integration demo
python examples/brain_demo.py
```

### Running Tests

```bash
# Basic functionality tests
python -c "from aisupea.core import Tensor; print('Core test passed')"

# Import all modules
python -c "import aisupea; print('All imports successful')"
```

### Project Structure

The framework is organized into focused modules:

- Each module is self-contained with clear interfaces
- Extensive documentation and type hints
- Comprehensive error handling
- Modular design allows for easy extension

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Performance optimizations
- Additional model architectures
- More tool types
- Enhanced debugging capabilities
- Additional interface options

## 📄 License

This project is open source. See LICENSE file for details.

## ⚠️ Disclaimer

This is a research and educational framework. While designed for production use in constrained environments, it may not be suitable for all production applications without further testing and optimization.

---

**Built with ❤️ in pure Python**