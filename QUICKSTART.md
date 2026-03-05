# Aisupea Data Generator - Quick Start

## ⭐ NOW SUPPORTING LARGE-SCALE GENERATION (5.5GB+)

The data generator now fetches **5.5GB+ of knowledge** from **7+ sources** (up from 550MB from 6 sources).

## 🚀 Start Here

### 1️⃣ Initialize (First Time Only)

```bash
python -m data_generator setup
```

### 2️⃣ Generate Knowledge Base

#### Option A: LARGE-SCALE (Recommended) 🎯

```bash
# Generate 5.5GB+ with all sources
python -m data_generator large_scale

# OR smaller scale for testing
python -m data_generator large_scale --scale small  # 500MB (30-60 min)
python -m data_generator large_scale --scale large  # 3-4GB (2-4 hours)
```

#### Option B: Standard Mode

```bash
# Generate standard 550MB (original)
python -m data_generator generate
```

### 3️⃣ Check Progress

```bash
python -m data_generator stats
```

---

## 📊 What's New

| Feature | Before | Now |
|---------|--------|-----|
| Data Sources | 6 | 7+ with variants |
| Per Module | 50MB | 500MB |
| Total Capacity | 550MB | 5.5GB+ |
| New Sources | - | Stack Exchange, YouTube Captions, News, Academic Papers, Code Snippets, Docs |

---

## 📝 Common Commands

### View Help
```bash
python -m data_generator --help
python -m data_generator large_scale --help
```

### LARGE-SCALE Generation (5.5GB+) 🌍
```bash
# All modules with all sources (5.5GB, 4-8 hours)
python -m data_generator large_scale

# Specific modules
python -m data_generator large_scale --module brain reasoning thinking

# Different scales
python -m data_generator large_scale --scale small    # 500MB, 30-60 min
python -m data_generator large_scale --scale large    # 3-4GB, 2-4 hours (default)
python -m data_generator large_scale --scale xlarge   # 5.5GB+, 4-8 hours
```

### Standard Generation (550MB)
```bash
# Generate all modules
python -m data_generator generate

# Specific module
python -m data_generator generate --module brain

# Force re-download
python -m data_generator generate --force

# With integrity verification
python -m data_generator generate --verify
```

### Inspect Data
```bash
# Load all modules
python -m data_generator load

# Load specific module
python -m data_generator load --module thinking
```

### Process Data
```bash
# Process all modules
python -m data_generator process

# Process specific module
python -m data_generator process --module knowledge

# Export to files
python -m data_generator process --module memory --export ./processed
```

### Statistics
```bash
python -m data_generator stats
```

### Cleanup
```bash
# Remove data older than 30 days
python -m data_generator clean --days 30

# Remove very old data
python -m data_generator clean --days 7
```

---

## 💻 Use in Python Code

### Simple Usage

```python
from data_generator.loader import DataPipeline

# Create pipeline
pipeline = DataPipeline()

# Get data for your module
dataset = pipeline.get_module_dataset('brain')

# Access sources
wikipedia = dataset['sources']['wikipedia']
github = dataset['sources']['github']
arxiv = dataset['sources']['arxiv']
```

### Process Items

```python
# Get items from a source
items = dataset['sources']['wikipedia']['items']

# Process each item
for item in items:
    title = item['title']
    content = item['content']
    url = item['url']
    license = item['license']
    
    print(f"Title: {title}")
    print(f"Content length: {len(content)} chars")
    print(f"URL: {url}")
    print(f"License: {license}")
```

### Use with Your Module

```python
from data_generator.loader import DataPipeline

class MyModule:
    def __init__(self):
        # Load knowledge for this module
        pipeline = DataPipeline()
        self.dataset = pipeline.get_module_dataset('my_module')
    
    def process(self):
        # Use knowledge from all sources
        for source_name, source_data in self.dataset['sources'].items():
            items = source_data.get('items', [])
            print(f"Processing {len(items)} from {source_name}")
```

---

## 📊 Modules with Data

After generation, you can get data for any of these 11 modules:

```
1. brain         - Core cognition
2. reasoning     - Logic & inference
3. thinking      - Planning & reflection
4. knowledge     - Facts & entities
5. memory        - Experience & learning
6. core          - Math & utilities
7. models        - Model configs
8. training      - Training examples
9. inference     - Test cases
10. agent        - Behaviors & strategies
11. interface    - User interactions
```

---

## 🔧 Configuration

### Change Settings

```python
from data_generator.config import DataGeneratorSettings

settings = DataGeneratorSettings()

# Change where data is stored
settings.data_path = "./my_knowledge_base"

# Change generation mode (fast/balanced/thorough)
settings.generation_mode = "balanced"  # default, recommended

# Save settings
settings.save()
```

### Generation Modes

```
FAST (10s timeout)
  ├─ Quick testing
  ├─ ~50-100MB total
  └─ Good for development

BALANCED (30s timeout) ⭐ RECOMMENDED
  ├─ Production default
  ├─ ~500MB total
  └─ Good balance of speed/quality

THOROUGH (120s timeout)
  ├─ Maximum data
  ├─ ~550MB+ total
  └─ Best quality, takes several hours
```

---

## 🌍 Data Sources

| Source | Content | License |
|--------|---------|---------|
| **Wikipedia** | Encyclopedic articles | CC-BY-SA 3.0 |
| **Common Crawl** | Web content | CC0 |
| **Open Library** | Book metadata | CC0/CC-BY |
| **GitHub** | Code & docs | OSS |
| **ArXiv** | Scientific papers | CC |
| **Gutenberg** | Classic books | Public Domain |

---

## 📈 Monitor Progress

```bash
# Check generation status
watch python -m data_generator stats

# Or in Python
from data_generator import DataGenerator

gen = DataGenerator()
stats = gen.get_statistics()

print(f"Progress: {stats['completion_percent']:.1f}%")
print(f"Size: {stats['total_size_mb']:.0f}MB / 550MB")
print(f"Files: {stats['total_files']}")
```

---

## ⚡ Performance Tips

### Faster Generation
```bash
# Use fast mode
# In config: generation_mode = "fast"

# Or set timeout lower in code
python -m data_generator generate --force
```

### Reduce Memory Usage
```python
# Don't load everything at once
from data_generator.loader import DataPipeline

pipeline = DataPipeline()

# Only load one module
for module in ['brain', 'reasoning', 'thinking']:
    dataset = pipeline.get_module_dataset(module)
    # Process and clear before next module
```

### Sample Large Datasets
```python
import random

# Full dataset might be large
all_items = dataset['sources']['wikipedia']['items']

# Use random sample instead
sample = random.sample(all_items, min(1000, len(all_items)))
for item in sample:
    process(item)
```

---

## 🐛 Troubleshooting

### "No data found"
```bash
# Generate first
python -m data_generator generate

# Or for specific module
python -m data_generator generate --module brain
```

### "Connection timeout"
```bash
# Try faster mode
# Edit config: generation_mode = "fast"

# Or skip failed sources and retry
python -m data_generator generate --force
```

### "Out of memory"
```bash
# Use lazy loading in Python
# Load one module at a time
for name in ['brain', 'reasoning', 'thinking']:
    dataset = pipeline.get_module_dataset(name)
    # Process and release
```

### "Disk space low"
```bash
# Clean old data
python -m data_generator clean --days 7

# Or use fast mode which generates less
```

---

## 🎯 Next Steps

1. **Generate**: `python -m data_generator generate`
2. **Verify**: `python -m data_generator stats`
3. **Use**: See integration examples below
4. **Monitor**: Check `./aisupea_data/.logs/`

---

## 📚 Examples

### Brain Module with Knowledge

```python
from brain import BrainModule
from data_generator.loader import DataPipeline

brain = BrainModule()
pipeline = DataPipeline()
knowledge = pipeline.get_module_dataset('brain')

# Brain now has:
# - Wikipedia entries (general knowledge)
# - GitHub code (technical knowledge)
# - ArXiv papers (research knowledge)
# - And more...

thought = brain.think("What is AI?")
```

### Reasoning with Papers

```python
from data_generator.loader import DataPipeline

pipeline = DataPipeline()
reasoning_data = pipeline.get_module_dataset('reasoning')

# Get scientific papers
papers = reasoning_data['sources']['arxiv']['items']

# Use for logical reasoning
result = reasoning.infer(
    premises=["All A are B", "X is A"],
    hypothesis="X is B",
    reference_papers=papers  # Use papers for validation
)
```

### Training with Data

```python
from data_generator.loader import DataPipeline
from training import Trainer

pipeline = DataPipeline()
training_data = pipeline.get_module_dataset('training')

trainer = Trainer(model)
trainer.train(data=training_data)
```

### Knowledge Graph from Data

```python
from data_generator.loader import DataPipeline

pipeline = DataPipeline()
knowledge = pipeline.get_module_dataset('knowledge')

# Build knowledge graph from Wikipedia
for item in knowledge['sources']['wikipedia']['items']:
    entity = item['title']
    description = item['content']
    build_graph_node(entity, description)
```

---

## 🔗 Full Documentation

For complete details, see:

- **DATA_GENERATION_GUIDE.md** - Comprehensive user guide
- **data_generator/README.md** - Technical reference
- **data_generator/INTEGRATION_GUIDE.md** - Integration examples
- **SYSTEM_OVERVIEW.md** - Architecture overview

---

## ✅ Checklist

- [ ] Run `python -m data_generator setup`
- [ ] Run `python -m data_generator generate`
- [ ] Run `python -m data_generator stats` to verify
- [ ] Review integration examples
- [ ] Integrate with your AI modules
- [ ] Test with sample data
- [ ] Deploy to production

---

**Ready to generate knowledge?** 🚀

```bash
python -m data_generator setup
python -m data_generator generate
python -m data_generator stats
```

Questions? See the documentation files or check logs in `./aisupea_data/.logs/`
