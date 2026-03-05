# Aisupea Data Generator - Integration Guide

## Quick Integration

### Step 1: Generate Knowledge Base

```bash
# Initialize system
python -m data_generator setup

# Generate all data (2-4 hours)
python -m data_generator generate

# Verify completion
python -m data_generator stats
```

### Step 2: Use in Your AI Module

```python
# In any Aisupea module (brain, reasoning, thinking, etc.)
from data_generator.loader import DataPipeline

class YourModule:
    def __init__(self):
        # Load knowledge dataset for your module
        self.pipeline = DataPipeline()
        self.dataset = self.pipeline.get_module_dataset('your_module_name')
        
    def train(self):
        # Use data from different sources
        for source_name, source_data in self.dataset['sources'].items():
            items = source_data.get('items', [])
            for item in items:
                self.process_item(item)
```

## Detailed Integration Examples

### Example 1: Brain Module Integration

```python
# /workspaces/Aisupea/brain/__init__.py
from data_generator.loader import DataPipeline

class BrainModule:
    """Brain module with integrated knowledge base."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.knowledge = self.pipeline.get_module_dataset('brain')
        self.index_knowledge()
    
    def index_knowledge(self):
        """Index knowledge from all sources."""
        self.knowledge_index = {}
        
        for source_name, source_data in self.knowledge['sources'].items():
            print(f"Indexing {source_name}...")
            
            if 'items' in source_data:
                for item in source_data['items']:
                    item_id = item.get('id', str(hash(item)))
                    self.knowledge_index[item_id] = item
    
    def query_knowledge(self, query: str):
        """Query the knowledge base."""
        results = []
        for item_id, item in self.knowledge_index.items():
            if query.lower() in str(item).lower():
                results.append(item)
        return results
    
    def think(self, prompt: str):
        """Think using integrated knowledge."""
        relevant = self.query_knowledge(prompt)
        # Use relevant knowledge in thinking process
        return self.process_with_knowledge(prompt, relevant)
    
    def process_with_knowledge(self, prompt: str, relevant_data):
        """Process prompt using relevant knowledge."""
        context = self._build_context(relevant_data)
        # Your thinking logic here
        return {"thought": "...", "context_used": len(relevant_data)}
    
    def _build_context(self, relevant_data):
        """Build context from relevant data."""
        return "\n".join([str(item) for item in relevant_data[:10]])


# Usage
if __name__ == "__main__":
    brain = BrainModule()
    thought = brain.think("What is artificial intelligence?")
    print(thought)
```

### Example 2: Reasoning Module Integration

```python
# /workspaces/Aisupea/reasoning/__init__.py
from data_generator.loader import DataPipeline, DataProcessor

class ReasoningModule:
    """Reasoning module with scientific knowledge integration."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.processor = DataProcessor()
        
        # Get reasoning dataset
        self.dataset = self.pipeline.get_module_dataset('reasoning')
        
        # Specifically use ArXiv papers for scientific reasoning
        self.papers = self.dataset['sources'].get('arxiv', {}).get('items', [])
        
        # Also use GitHub for logical examples
        self.logic_examples = self.dataset['sources'].get('github', {}).get('items', [])
    
    def infer(self, premises: list, hypothesis: str) -> dict:
        """Infer using integrated knowledge."""
        # Find relevant papers
        relevant_papers = self._find_relevant_papers(premises)
        
        # Find logical examples
        relevant_examples = self._find_logical_examples(hypothesis)
        
        # Perform inference
        result = self._perform_inference(
            premises, 
            hypothesis, 
            relevant_papers, 
            relevant_examples
        )
        
        return result
    
    def _find_relevant_papers(self, premises):
        """Find relevant scientific papers."""
        relevant = []
        for paper in self.papers:
            keywords = str(paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            if any(p.lower() in keywords for p in premises):
                relevant.append(paper)
        return relevant[:5]  # Top 5 papers
    
    def _find_logical_examples(self, hypothesis):
        """Find relevant code examples."""
        relevant = []
        for example in self.logic_examples:
            if hypothesis.lower() in str(example).lower():
                relevant.append(example)
        return relevant[:3]  # Top 3 examples
    
    def _perform_inference(self, premises, hypothesis, papers, examples):
        """Perform the actual reasoning."""
        return {
            "premises": premises,
            "hypothesis": hypothesis,
            "supported_by_papers": len(papers),
            "supported_by_examples": len(examples),
            "confidence": 0.8  # Example confidence
        }


# Usage
if __name__ == "__main__":
    reasoning = ReasoningModule()
    result = reasoning.infer(
        premises=["All birds have wings", "Penguins are birds"],
        hypothesis="Penguins have wings"
    )
    print(result)
```

### Example 3: Knowledge Module Integration

```python
# /workspaces/Aisupea/knowledge/__init__.py
from data_generator.loader import DataPipeline
from data_generator.loader import DataProcessor
import json

class KnowledgeModule:
    """Knowledge module with fact base from Wikipedia and Open Library."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.processor = DataProcessor()
        
        # Get knowledge dataset
        self.dataset = self.pipeline.get_module_dataset('knowledge')
        
        # Build knowledge graphs
        self.build_knowledge_graph()
    
    def build_knowledge_graph(self):
        """Build knowledge graph from sources."""
        self.facts = {}
        self.entities = {}
        
        # Extract facts from Wikipedia
        wiki_items = self.dataset['sources'].get('wikipedia', {}).get('items', [])
        for item in wiki_items:
            entity_name = item.get('title', '')
            entity_description = item.get('content', '')
            
            if entity_name not in self.entities:
                self.entities[entity_name] = {
                    'description': entity_description[:500],  # First 500 chars
                    'url': item.get('url', ''),
                    'source': 'wikipedia',
                    'related': []
                }
        
        # Extract facts from Open Library
        lib_items = self.dataset['sources'].get('open_library', {}).get('items', [])
        for item in lib_items:
            book_title = item.get('title', '')
            self.facts[book_title] = {
                'author': item.get('author', ''),
                'year': item.get('year', ''),
                'description': item.get('description', ''),
                'source': 'open_library'
            }
    
    def query_entity(self, entity_name: str) -> dict:
        """Query information about an entity."""
        return self.entities.get(entity_name, None)
    
    def query_fact(self, fact_name: str) -> dict:
        """Query a specific fact."""
        return self.facts.get(fact_name, None)
    
    def find_related_entities(self, entity_name: str) -> list:
        """Find related entities."""
        if entity_name not in self.entities:
            return []
        
        entity = self.entities[entity_name]
        related = []
        
        # Simple keyword-based relation finder
        entity_keywords = set(entity['description'].lower().split()[:20])
        
        for other_name, other_entity in self.entities.items():
            if other_name == entity_name:
                continue
            
            other_keywords = set(other_entity['description'].lower().split()[:20])
            similarity = len(entity_keywords & other_keywords) / max(len(entity_keywords | other_keywords), 1)
            
            if similarity > 0.3:
                related.append((other_name, similarity))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def export_knowledge_base(self, filepath: str):
        """Export knowledge base to JSON."""
        kb = {
            'entities': self.entities,
            'facts': self.facts,
            'total_entities': len(self.entities),
            'total_facts': len(self.facts)
        }
        
        with open(filepath, 'w') as f:
            json.dump(kb, f, indent=2)


# Usage
if __name__ == "__main__":
    kb = KnowledgeModule()
    print(f"Loaded {len(kb.entities)} entities and {len(kb.facts)} facts")
    kb.export_knowledge_base('./knowledge_base.json')
```

### Example 4: Training Module Integration

```python
# /workspaces/Aisupea/training/__init__.py
from data_generator.loader import DataPipeline

class TrainingDataset:
    """Training dataset from generated knowledge."""
    
    def __init__(self, module_name: str = 'training'):
        self.pipeline = DataPipeline()
        self.dataset = self.pipeline.get_module_dataset(module_name)
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self):
        """Prepare training examples from knowledge sources."""
        examples = []
        
        # Gather examples from all sources
        for source_name, source_data in self.dataset['sources'].items():
            items = source_data.get('items', [])
            
            for item in items:
                example = {
                    'id': item.get('id', ''),
                    'input': item.get('title', ''),
                    'output': item.get('content', ''),
                    'source': source_name,
                    'metadata': {k: v for k, v in item.items() 
                               if k not in ['id', 'title', 'content']}
                }
                examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_batch(self, batch_size: int, start_idx: int = 0):
        """Get a batch of examples."""
        end_idx = min(start_idx + batch_size, len(self.examples))
        return self.examples[start_idx:end_idx]
    
    def statistics(self):
        """Get dataset statistics."""
        return {
            'total_examples': len(self.examples),
            'sources': len(self.dataset['sources']),
            'avg_input_length': sum(len(e['input']) for e in self.examples) / max(len(self.examples), 1),
            'avg_output_length': sum(len(e['output']) for e in self.examples) / max(len(self.examples), 1),
        }


class Trainer:
    """Trainer using generated knowledge data."""
    
    def __init__(self, model, module_name: str = 'training'):
        self.model = model
        self.dataset = TrainingDataset(module_name)
    
    def train(self, epochs: int = 1, batch_size: int = 32):
        """Train model on knowledge data."""
        total_examples = len(self.dataset)
        batches_per_epoch = (total_examples + batch_size - 1) // batch_size
        
        print(f"Training on {total_examples} examples")
        print(f"Batches per epoch: {batches_per_epoch}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            for batch_idx in range(0, total_examples, batch_size):
                batch = self.dataset.get_batch(batch_size, batch_idx)
                
                # Your training logic
                loss = self._train_batch(batch)
                
                if (batch_idx // batch_size) % 10 == 0:
                    print(f"  Batch {batch_idx // batch_size}: Loss = {loss:.4f}")
    
    def _train_batch(self, batch):
        """Train on single batch."""
        # Implement your training logic
        return 0.5  # Example loss


# Usage
if __name__ == "__main__":
    dataset = TrainingDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Statistics: {dataset.statistics()}")
    
    # Get first example
    example = dataset[0]
    print(f"\nFirst example: {example['input'][:100]}...")
```

### Example 5: Complete Module with All Features

```python
# /workspaces/Aisupea/agent/__init__.py
from data_generator.loader import DataPipeline
from data_generator.loader import DataProcessor

class IntelligentAgent:
    """Agent that integrates all data generator features."""
    
    def __init__(self, agent_name: str = 'default'):
        self.name = agent_name
        self.pipeline = DataPipeline()
        self.processor = DataProcessor()
        
        # Load all relevant data
        self.load_knowledge()
        self.load_behaviors()
        self.load_strategies()
    
    def load_knowledge(self):
        """Load knowledge base."""
        self.knowledge = self.pipeline.get_module_dataset('knowledge')
        self.knowledge_items = {}
        
        for source_name, source_data in self.knowledge['sources'].items():
            items = source_data.get('items', [])
            self.knowledge_items[source_name] = items
    
    def load_behaviors(self):
        """Load behavioral patterns from agent module data."""
        self.behaviors = self.pipeline.get_module_dataset('agent')
        self.behavior_patterns = {}
        
        for source_name, source_data in self.behaviors['sources'].items():
            items = source_data.get('items', [])
            for item in items:
                behavior_type = item.get('type', 'default')
                if behavior_type not in self.behavior_patterns:
                    self.behavior_patterns[behavior_type] = []
                self.behavior_patterns[behavior_type].append(item)
    
    def load_strategies(self):
        """Load strategic decision patterns."""
        # Use knowledge and reasoning data for strategies
        self.strategies = {}
        self.strategies['problem_solving'] = self._build_problem_solving_strategies()
        self.strategies['learning'] = self._build_learning_strategies()
    
    def _build_problem_solving_strategies(self):
        """Build problem-solving strategies from knowledge."""
        strategies = []
        
        # Extract strategies from GitHub code examples
        github_data = self.knowledge['sources'].get('github', {}).get('items', [])
        for item in github_data[:50]:  # First 50 items
            strategies.append({
                'name': item.get('title', 'unnamed'),
                'code': item.get('content', ''),
                'language': 'python',
                'source': 'github'
            })
        
        return strategies
    
    def _build_learning_strategies(self):
        """Build learning strategies from research papers."""
        strategies = []
        
        # Extract strategies from ArXiv papers
        arxiv_data = self.knowledge['sources'].get('arxiv', {}).get('items', [])
        for item in arxiv_data[:30]:  # First 30 papers
            strategies.append({
                'name': item.get('title', 'unnamed'),
                'abstract': item.get('abstract', ''),
                'authors': item.get('authors', []),
                'year': item.get('year', ''),
                'source': 'arxiv'
            })
        
        return strategies
    
    def act(self, situation: str) -> dict:
        """Agent takes action based on situation."""
        # Analyze situation using knowledge
        context = self._analyze_situation(situation)
        
        # Find relevant behaviors
        relevant_behaviors = self._find_relevant_behaviors(situation)
        
        # Choose strategy
        strategy = self._choose_strategy(context, relevant_behaviors)
        
        # Execute action
        action = self._execute_action(strategy, context)
        
        return {
            'situation': situation,
            'context': context,
            'chosen_strategy': strategy['name'],
            'action': action,
            'confidence': 0.85
        }
    
    def _analyze_situation(self, situation: str) -> dict:
        """Analyze the situation using knowledge."""
        # Find relevant knowledge items
        relevant_items = []
        for source_name, items in self.knowledge_items.items():
            for item in items[:10]:  # Sample items
                if any(word in str(item).lower() for word in situation.lower().split()):
                    relevant_items.append((source_name, item))
        
        return {
            'situation': situation,
            'relevant_knowledge_sources': list(set(s[0] for s in relevant_items)),
            'relevant_items': len(relevant_items)
        }
    
    def _find_relevant_behaviors(self, situation: str):
        """Find relevant behavioral patterns."""
        relevant = []
        for behavior_type, patterns in self.behavior_patterns.items():
            for pattern in patterns[:5]:
                relevant.append((behavior_type, pattern))
        return relevant
    
    def _choose_strategy(self, context, behaviors):
        """Choose best strategy for context."""
        # Select from problem-solving or learning strategies
        if len(context['relevant_knowledge_sources']) > 0:
            strategies = self.strategies['problem_solving']
        else:
            strategies = self.strategies['learning']
        
        return strategies[0] if strategies else {'name': 'default', 'action': 'think'}
    
    def _execute_action(self, strategy, context):
        """Execute the chosen strategy."""
        return f"Executing {strategy['name']} based on context"
    
    def report_status(self):
        """Report agent status."""
        return {
            'name': self.name,
            'knowledge_sources': len(self.knowledge_items),
            'knowledge_items': sum(len(items) for items in self.knowledge_items.values()),
            'behaviors_loaded': len(self.behavior_patterns),
            'strategies_available': sum(len(s) for s in self.strategies.values())
        }


# Usage
if __name__ == "__main__":
    agent = IntelligentAgent('main_agent')
    
    print(f"Agent Status: {agent.report_status()}")
    
    # Agent acts on a situation
    result = agent.act("I need to solve a programming problem")
    print(f"\nAgent Action: {result}")
```

## Integration Checklist

- [ ] Generate knowledge base: `python -m data_generator generate`
- [ ] Verify generation: `python -m data_generator stats`
- [ ] Import DataPipeline in your module
- [ ] Load dataset for your module: `pipeline.get_module_dataset('module_name')`
- [ ] Process data using DataProcessor if needed
- [ ] Use data in your module's logic
- [ ] Test integration with sample data
- [ ] Monitor performance and storage

## Performance Tips for Integration

1. **Lazy Loading**: Don't load all datasets at once
   ```python
   # Do this (lazy)
   self.pipeline = DataPipeline()
   self.dataset = self.pipeline.get_module_dataset(self.module_name)
   
   # Not this (eager loads everything)
   all_datasets = self.pipeline.get_all_datasets()
   ```

2. **Cache Processed Data**: Reuse processed datasets
   ```python
   if not hasattr(self, '_cached_dataset'):
       self._cached_dataset = self.pipeline.get_module_dataset('brain')
   return self._cached_dataset
   ```

3. **Sample Large Datasets**: Use statistical sampling for large data
   ```python
   import random
   sample = random.sample(items, min(1000, len(items)))
   ```

4. **Stream Processing**: Process items one at a time
   ```python
   for item in source_data['items']:
       process(item)  # Process singly, don't load all in memory
   ```

## Troubleshooting Integration

| Issue | Solution |
|-------|----------|
| "No module named/data_generator" | Make sure data_generator folder is in /workspaces/Aisupea |
| "Dataset is empty" | Run `python -m data_generator generate` to create data |
| "ModuleNotFoundError" | Check data_generator/__init__.py exists |
| "Out of memory" | Use lazy loading or reduce sample size |
| "Data not found for module X" | Generate data for that specific module |

## Next Steps

1. **Generate the knowledge base**: `python -m data_generator generate`
2. **Integrate with your modules**: Use examples above as templates
3. **Test integration**: Run your modules and verify data is loaded
4. **Monitor performance**: Use `python -m data_generator stats` for tracking
5. **Update regularly**: Regenerate monthly for fresh knowledge

---

**You're ready to power your AI modules with knowledge!** 🚀

For questions, see the [DATA_GENERATION_GUIDE.md](../DATA_GENERATION_GUIDE.md) or [data_generator/README.md](./README.md)
