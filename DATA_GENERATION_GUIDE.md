# Aisupea Data Generation Guide

## Overview

The Aisupea Data Generator creates a comprehensive 550MB knowledge base from non-copyrighted internet sources to power your AI modules. It supports 6 different data sources and organizes data specifically for 11 AI modules.

## Quick Start

### 1. Setup

```bash
python -m data_generator.runner setup
```

This initializes the data generator configuration and creates necessary directories.

### 2. Generate Data

```bash
# Generate all modules (550MB total)
python -m data_generator.runner generate

# Generate specific module (50MB)
python -m data_generator.runner generate --module brain

# Force re-download (skip cache)
python -m data_generator.runner generate --force

# Generate and verify integrity
python -m data_generator.runner generate --verify
```

### 3. Load and Inspect Data

```bash
# Load all modules
python -m data_generator.runner load

# Load specific module
python -m data_generator.runner load --module reasoning
```

### 4. Process Data

```bash
# Process all modules
python -m data_generator.runner process

# Process specific module
python -m data_generator.runner process --module knowledge

# Export processed data
python -m data_generator.runner process --module core --export ./processed_data
```

### 5. View Statistics

```bash
python -m data_generator.runner stats
```

Shows completion percentage, file counts, and per-module breakdown.

### 6. Clean Old Data

```bash
# Remove data older than 30 days
python -m data_generator.runner clean --days 30
```

## Data Sources

The system fetches from 6 non-copyrighted sources:

| Source | License | Content | Size Target |
|--------|---------|---------|-------------|
| **Wikipedia** | CC-BY-SA 3.0 | Encyclopedia articles, summaries | 100-150MB |
| **Common Crawl** | CC0 | Web page snapshots, text data | 100-150MB |
| **Open Library** | CC0 / CC-BY | Book metadata, descriptions | 50-100MB |
| **GitHub** | OSS Licenses | Public code repositories, documentation | 100-150MB |
| **ArXiv** | Open Access | Scientific papers, abstracts | 50-100MB |
| **Project Gutenberg** | Public Domain | Full texts of classic books | 50-100MB |

**Total: 550MB knowledge base**

## AI Modules

Data is organized for 11 AI modules:

1. **brain** - Core cognitive processing (50MB)
2. **reasoning** - Logical inference and deduction (50MB)
3. **thinking** - Planning and reflection (50MB)
4. **knowledge** - Knowledge graphs and facts (50MB)
5. **memory** - Experience storage and recall (50MB)
6. **core** - Foundation mathematics and utilities (50MB)
7. **models** - Pre-trained model weights and configs (50MB)
8. **training** - Training data and examples (50MB)
9. **inference** - Inference scenarios and test cases (50MB)
10. **agent** - Agent behaviors and strategies (50MB)
11. **interface** - User interaction patterns (50MB)

## Programmatic Usage

### Generate Data Programmatically

```python
from data_generator import DataGenerator

# Create generator
gen = DataGenerator(base_path="./aisupea_data")

# Generate all modules
gen.generate_all_modules()

# OR generate specific module
gen.generate_module_data("brain")

# Check statistics
stats = gen.get_statistics()
print(f"Completion: {stats['completion_percent']:.1f}%")
print(f"Total size: {stats['total_size_mb']:.2f}MB")
```

### Load Data

```python
from data_generator.loader import DataLoader

loader = DataLoader("./aisupea_data")

# Load all modules
all_data = loader.get_all_modules_data()

# OR load specific module
brain_data = loader.load_module_data("brain")
# Returns dict with keys: "wikipedia", "common_crawl", "open_library", etc.
```

### Process Data

```python
from data_generator.loader import DataPipeline

pipeline = DataPipeline("./aisupea_data")

# Get processed dataset for module
dataset = pipeline.get_module_dataset("reasoning")

# Dataset structure:
# {
#   'module': 'reasoning',
#   'sources': {
#     'wikipedia': {...processed data...},
#     'github': {...processed data...},
#     ...
#   },
#   'metadata': {...}
# }

# Export all processed data
pipeline.export_processed_data("./processed_data")
```

## Configuration

The system uses `DataGeneratorConfig` for source and module configuration:

```python
from data_generator.config import DataGeneratorConfig

config = DataGeneratorConfig.get_config()

# Access source configuration
sources = config['sources']
print(sources['wikipedia']['url'])

# Access module configuration
modules = config['modules']
print(modules['brain']['description'])

# Access generation modes
modes = config['generation_modes']
print(modes['balanced']['timeout'])  # seconds
```

## Data Format

Generated data follows this directory structure:

```
./aisupea_data/
├── brain/
│   ├── wikipedia.json
│   ├── common_crawl.json
│   ├── open_library.json
│   ├── github.json
│   ├── arxiv.json
│   └── gutenberg.json
├── reasoning/
│   ├── wikipedia.json
│   ├── ...
├── thinking/
├── knowledge/
├── memory/
├── core/
├── models/
├── training/
├── inference/
├── agent/
├── interface/
└── .metadata.json
```

Each JSON file contains:
```json
{
  "module": "brain",
  "source": "wikipedia",
  "timestamp": "2024-01-01T00:00:00Z",
  "items": [
    {
      "id": "unique_id",
      "title": "Title",
      "content": "Content...",
      "url": "source_url",
      "license": "CC-BY-SA 3.0",
      "metadata": {...}
    },
    ...
  ],
  "count": 1000,
  "size_bytes": 5242880,
  "sha256": "hash"
}
```

## Generation Modes

Three generation modes are available:

### Fast Mode (Timeout: 10s per source)
- Quick data fetching
- Limited content per source
- Ideal for testing
- Generates ~50-100MB total

### Balanced Mode (Timeout: 30s per source) ⭐ DEFAULT
- Moderate data fetching
- Good content coverage
- Recommended for production
- Generates ~500MB total (target: 550MB)

### Thorough Mode (Timeout: 120s per source)
- Comprehensive data fetching
- Maximum content per source
- Best data quality
- Generates ~550MB+ total
- Warning: May take several hours

Configure mode in `DataGeneratorSettings`:

```python
from data_generator.config import DataGeneratorSettings

settings = DataGeneratorSettings()
settings.generation_mode = "balanced"  # or "fast", "thorough"
settings.save()
```

## Troubleshooting

### Data Generation Fails

**Issue**: "Failed to connect to source"
- **Solution**: Check internet connection, some sources may be temporarily down
- **Workaround**: Use `--force` flag to retry, or skip failed sources

**Issue**: "Timeout error"
- **Solution**: Use faster generation mode or increase timeout in config
- **Command**: `python -m data_generator.runner generate --module brain`

### Data Corruption

**Issue**: "Integrity check failed"
- **Solution**: Regenerate the data with `--force` flag
- **Command**: `python -m data_generator.runner generate --force --verify`

### Storage Issues

**Issue**: "Not enough disk space"
- **Solution**: Clean old data or use fast mode
- **Commands**: 
  - `python -m data_generator.runner clean --days 7`
  - `python -m data_generator.runner generate --module brain` (partial)

### Missing Data

**Issue**: "Module data not found"
- **Solution**: Generate data first
- **Command**: `python -m data_generator.runner generate`

## Performance Tips

1. **Start with specific modules**: Generate one module at a time to monitor progress
   ```bash
   python -m data_generator.runner generate --module brain
   python -m data_generator.runner generate --module reasoning
   ```

2. **Use balanced mode**: Best trade-off between speed and quality
   ```bash
   # Set in DataGeneratorSettings
   settings.generation_mode = "balanced"
   ```

3. **Enable caching**: Reuse previously fetched data (default behavior)
   ```bash
   # Only re-download stale data
   python -m data_generator.runner generate
   ```

4. **Run in background**: Generate during idle time
   ```bash
   nohup python -m data_generator.runner generate &
   ```

5. **Monitor progress**: Check statistics periodically
   ```bash
   while true; do
     python -m data_generator.runner stats
     sleep 60
   done
   ```

## Integration with AI Modules

### Using Generated Data

```python
# In your AI module (e.g., brain)
from data_generator.loader import DataPipeline

class BrainModule:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.dataset = self.pipeline.get_module_dataset('brain')
        
    def train(self):
        # Use dataset['sources'] for training
        for source_name, source_data in self.dataset['sources'].items():
            self.process_source(source_name, source_data)
```

### Loading Module-Specific Data

```python
from data_generator.loader import DataProcessor

processor = DataProcessor()

# Get processing pipeline for a module
brain_processor = processor.get_module_processor('brain')

# Process raw data for the module
processed = brain_processor.process({
    'wikipedia': raw_wikipedia_data,
    'github': raw_github_data,
    ...
})
```

## License Information

All generated data respects original source licenses:
- **Wikipedia**: CC-BY-SA 3.0 (attribution required)
- **Common Crawl**: CC0 (public domain)
- **Open Library**: CC0 / CC-BY (varies by item)
- **GitHub**: Original repository licenses (OSS)
- **ArXiv**: Creative Commons licenses (open access)
- **Project Gutenberg**: Public domain

**IMPORTANT**: If using generated data commercially, ensure you comply with all source licenses.

## FAQ

**Q: How long does generation take?**
- A: Balanced mode: 2-4 hours. Fast mode: 30-60 minutes. Thorough mode: 6-12 hours.

**Q: Can I use generated data for commercial purposes?**
- A: Check each source's license. Wikipedia CC-BY-SA requires attribution. Common Crawl CC0 is free. Check others accordingly.

**Q: What if a source is down?**
- A: System handles gracefully - failed sources are skipped, others continue.

**Q: Can I modify the 550MB target?**
- A: Yes, in `DataGeneratorConfig`, adjust `target_size_mb` per module.

**Q: How often should I regenerate data?**
- A: Monthly recommended to keep knowledge current. Use `--force` to override cache.

**Q: Can I add custom data sources?**
- A: Yes, extend `DataSource` class in `data_generator/__init__.py`.

## Getting Help

- Check logs in `./aisupea_data/.logs/`
- Run `python -m data_generator.runner stats` for diagnostics
- Review configuration in `data_generator/config.py`
- Enable debug mode in `DataGeneratorSettings`

---

**Ready to begin?**
```bash
python -m data_generator.runner setup
python -m data_generator.runner generate
python -m data_generator.runner stats
```

Happy learning! 🚀
