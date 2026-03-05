# Data Generator Package

Comprehensive data generation system for Aisupea AI framework. Fetches and organizes non-copyrighted knowledge from the internet (550MB total) across 11 AI modules.

## Package Structure

```
data_generator/
├── __init__.py          # Core DataSource classes and DataGenerator orchestrator
├── __main__.py          # Module entry point (python -m data_generator)
├── loader.py            # DataLoader, DataProcessor, DataPipeline classes
├── config.py            # Configuration and settings management
├── runner.py            # CLI runner with subcommands
├── README.md            # This file
└── SOURCES_CONFIG       # Source configurations (optional)
```

## Module Components

### 1. `__init__.py` - Data Sources and Generator

**Classes:**
- `DataSource` (abstract base class)
  - `WikipediaDataSource` - Fetch Wikipedia articles
  - `CommonCrawlDataSource` - Fetch web page content
  - `OpenLibraryDataSource` - Fetch book metadata
  - `GithubDataSource` - Fetch public repositories
  - `ArxivDataSource` - Fetch scientific papers
  - `ProjectGutenbergDataSource` - Fetch classic books

- `DataGenerator` - Orchestrate multi-source data generation
  - `generate_all_modules()` - Generate for all 11 modules
  - `generate_module_data(module_name)` - Generate for one module
  - `get_module_data(module_name)` - Retrieve generated data
  - `get_statistics()` - Get generation statistics
  - `verify_data_integrity()` - Check data validity
  - `cleanup_old_data(days)` - Remove stale data

**Key Features:**
- Parallel source fetching
- Metadata caching with SQLite
- Integrity verification (SHA256)
- Graceful error handling
- Progress tracking
- Module-specific data organization

### 2. `loader.py` - Data Loading and Processing

**Classes:**
- `DataLoader` - Load generated data
  - `load_module_data(module_name)` - Load single module
  - `get_all_modules_data()` - Load all modules
  - Methods for different data formats

- `DataProcessor` - Process data for AI modules
  - `get_module_processor(module_name)` - Get processor for module
  - Module-specific processors:
    - `BrainProcessor` - Parse cognitive data
    - `ReasoningProcessor` - Parse inference data
    - `ThinkingProcessor` - Parse planning data
    - `KnowledgeProcessor` - Parse fact graphs
    - `MemoryProcessor` - Parse examples
    - `CoreProcessor` - Parse math data
    - `ModelsProcessor` - Parse model configs
    - `TrainingProcessor` - Parse training examples
    - `InferenceProcessor` - Parse test cases
    - `AgentProcessor` - Parse behaviors
    - `InterfaceProcessor` - Parse UI patterns

- `DataPipeline` - End-to-end data orchestration
  - `get_module_dataset(module_name)` - Get processed dataset
  - `get_all_datasets()` - Get all processed datasets
  - `export_processed_data(path)` - Export to files
  - Batch processing support

**Key Features:**
- Streaming data loading
- Module-specific processing
- Batch support
- Export functionality
- Caching of processed data

### 3. `config.py` - Configuration Management

**Classes:**
- `DataGeneratorConfig` - Static configuration
  - SOURCES_CONFIG (6 sources with URLs, metadata)
  - MODULES_CONFIG (11 modules with descriptions)
  - GENERATION_MODES (fast, balanced, thorough)

- `DataGeneratorSettings` - User preferences
  - `data_path` - Where to store generated data
  - `generation_mode` - fast/balanced/thorough
  - `enable_caching` - Use cached data
  - `verify_integrity` - Check SHA256
  - Persistence (save/load to JSON)

- Helper function: `setup_data_generator()` - Initialize system

**Key Features:**
- Centralized configuration
- Multiple generation modes
- Persistent settings
- Easy customization

### 4. `runner.py` - CLI Interface

**Class:**
- `DataGeneratorCLI` - Command-line interface with subcommands

**Commands:**
- `setup` - Initialize data generator
- `generate` - Create knowledge data
- `load` - Load and inspect data
- `process` - Process data for modules
- `stats` - Show statistics
- `clean` - Remove old data

**Usage:**
```bash
# Generate all data
python -m data_generator generate

# Generate specific module
python -m data_generator generate --module brain

# Load and inspect
python -m data_generator load --module reasoning

# Process and export
python -m data_generator process --module knowledge --export ./out

# View statistics
python -m data_generator stats
```

## Architecture Overview

```
                    ┌─────────────────┐
                    │  DataGenerator  │
                    │   (Orchestrator)│
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌─────────┐        ┌─────────┐        ┌──────────┐
    │Wikipedia │        │Common   │        │Open      │
    │DataSource│        │CrawlDS  │        │LibraryDS │
    └─────────┘        └─────────┘        └──────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                     ┌───────┴────────┐
                     │  DataLoader    │
                     │  (Load & Read) │
                     └───────┬────────┘
                             │
                     ┌───────┴────────────┐
                     │ DataProcessor     │
                     │ (Module-Specific) │
                     └───────┬────────────┘
                             │
                     ┌───────┴─────────┐
                     │ DataPipeline    │
                     │ (Orchestrate)   │
                     └─────────────────┘
```

## Data Flow

1. **Generation Phase**
   - `DataGenerator.generate_all_modules()`
   - For each module:
     - For each source (Wikipedia, GitHub, ArXiv, etc.):
       - Fetch data (caching enabled)
       - Validate integrity
       - Save as JSON
   - Total: 550MB across 11 modules

2. **Loading Phase**
   - `DataLoader.load_module_data(module_name)`
   - For each source:
     - Read JSON file
     - Parse content
     - Return as dict

3. **Processing Phase**
   - `DataProcessor.get_module_processor(module_name)`
   - Use module-specific processor
   - Transform raw data to structured format
   - Cache processed results

4. **Usage Phase**
   - AI modules consume processed data
   - Use `DataPipeline.get_module_dataset()`
   - Access via `dataset['sources'][source_name]`

## Example Workflows

### Workflow 1: Generate All Data

```python
from data_generator import DataGenerator

# Create generator
gen = DataGenerator(base_path="./aisupea_knowledge")

# Generate all 11 modules (takes 2-4 hours)
gen.generate_all_modules()

# Check progress
stats = gen.get_statistics()
print(f"Generated: {stats['total_size_mb']:.2f}MB / 550MB")
```

### Workflow 2: Load and Use Data in Brain Module

```python
from data_generator.loader import DataPipeline

# Create pipeline
pipeline = DataPipeline()

# Get processed dataset for brain module
dataset = pipeline.get_module_dataset('brain')

# Access data from different sources
wiki_data = dataset['sources']['wikipedia']
github_data = dataset['sources']['github']
arxiv_data = dataset['sources']['arxiv']

# Use in training or inference
for item in wiki_data['items']:
    # item = {'id': '...', 'title': '...', 'content': '...', ...}
    process_item(item)
```

### Workflow 3: Process Custom Data

```python
from data_generator.loader import DataProcessor

# Create processor
processor = DataProcessor()

# Get processor for reasoning module
reasoning_proc = processor.get_module_processor('reasoning')

# Process raw data
processed = reasoning_proc.process(raw_data)

# Use processed data
print(f"Processed {processed['count']} items")
for logic_item in processed['items']:
    print(f"  - {logic_item['rule']}")
```

### Workflow 4: CLI-Based Generation

```bash
# Setup
python -m data_generator setup

# Generate specific modules one by one
python -m data_generator generate --module brain
python -m data_generator generate --module reasoning
python -m data_generator generate --module thinking

# Check progress
python -m data_generator stats

# Export processed data
python -m data_generator process --export ./processed_knowledge
```

## Data Sources Details

### 1. Wikipedia (CC-BY-SA 3.0)
- **URL**: https://en.wikipedia.org/
- **Content**: Encyclopedic articles, summaries, overviews
- **Target**: 100-150MB
- **License**: Creative Commons Attribution-ShareAlike 3.0
- **Best For**: Knowledge base, general information

### 2. Common Crawl (CC0)
- **URL**: https://commoncrawl.org/
- **Content**: Web page snapshots, text content
- **Target**: 100-150MB
- **License**: Public Domain (CC0)
- **Best For**: Web content, diverse text, general knowledge

### 3. Open Library (CC0/CC-BY)
- **URL**: https://openlibrary.org/
- **Content**: Book metadata, descriptions, summaries
- **Target**: 50-100MB
- **License**: varies (mostly CC0/CC-BY)
- **Best For**: Literary knowledge, book information

### 4. GitHub (OSS)
- **URL**: https://github.com/
- **Content**: Public code, documentation, readmes
- **Target**: 100-150MB
- **License**: Original repository licenses
- **Best For**: Code knowledge, technical documentation

### 5. ArXiv (Open Access)
- **URL**: https://arxiv.org/
- **Content**: Scientific papers, abstracts, preprints
- **Target**: 50-100MB
- **License**: Creative Commons (varies)
- **Best For**: Scientific knowledge, research, papers

### 6. Project Gutenberg (Public Domain)
- **URL**: https://www.gutenberg.org/
- **Content**: Full text classic books, literature
- **Target**: 50-100MB
- **License**: Public Domain
- **Best For**: Literary knowledge, classic texts

## Module Configuration

Each of the 11 modules has specific data organization:

```python
MODULES_CONFIG = {
    'brain': {
        'description': 'Core cognitive processing and thought',
        'target_size_mb': 50,
        'sources': ['wikipedia', 'github', 'arxiv'],
        'processor': 'BrainProcessor',
        # ... more config
    },
    'reasoning': {
        'description': 'Logical inference and deduction',
        'target_size_mb': 50,
        'sources': ['arxiv', 'github', 'wikipedia'],
        'processor': 'ReasoningProcessor',
        # ... more config
    },
    # ... 9 more modules
}
```

## Performance Characteristics

| Operation | Time | Memory | Storage |
|-----------|------|--------|---------|
| Generate all modules | 2-4 hours | ~200MB RAM | 550MB disk |
| Generate single module | 10-30 min | ~50MB RAM | 50MB disk |
| Load all modules | 30-60 sec | ~300MB RAM | Cache |
| Process all modules | 1-2 min | ~200MB RAM | 550MB out |
| Query single item | <1ms | Minimal | N/A |

## Customization

### Add Custom Data Source

```python
from data_generator import DataSource

class CustomDataSource(DataSource):
    """Custom data source."""
    
    def fetch_data(self, module_name: str, config: dict) -> dict:
        """Fetch data for module."""
        items = []
        # Your custom fetch logic
        return {'items': items}
```

### Modify Module Configuration

```python
from data_generator.config import DataGeneratorConfig

config = DataGeneratorConfig.get_config()
config['modules']['brain']['target_size_mb'] = 100
```

### Custom Processing Pipeline

```python
from data_generator.loader import DataProcessor

class CustomProcessor(DataProcessor):
    """Custom data processor."""
    
    def process_custom(self, raw_data: dict) -> dict:
        """Custom processing logic."""
        return processed_data
```

## Troubleshooting

### Source Connection Issues

If a source fails to connect:
1. Check internet connection
2. Verify source URL is accessible
3. Try reducing timeout in fast mode
4. Use cached data from previous run
5. Skip problematic source and retry others

### Storage Issues

If disk space is low:
1. Clean old data: `python -m data_generator clean --days 7`
2. Use faster generation mode
3. Generate one module at a time
4. Increase system disk space

### Memory Issues

If running out of RAM:
1. Close other applications
2. Use DataLoader with streaming
3. Process one module at a time
4. Reduce batch size in DataProcessor

### Integrity Failures

If data integrity check fails:
1. Regenerate with `--force` flag
2. Check for disk corruption
3. Verify available disk space
4. Use `--verify` during generation

## Best Practices

1. **Generate incrementally**: Generate one module at a time to monitor progress
2. **Use balanced mode**: Best speed/quality trade-off
3. **Enable caching**: Reuse previously fetched data
4. **Verify integrity**: Check data after generation
5. **Monitor storage**: Keep track of disk space
6. **Update regularly**: Regenerate monthly for fresh knowledge
7. **Backup important data**: Keep local copies of critical data
8. **Process selectively**: Only process needed modules

## Dependencies

**Pure Python - No external imports required!**
- Uses only standard library modules:
  - `urllib` - HTTP requests
  - `json` - Data serialization
  - `gzip` - Compression
  - `hashlib` - Integrity checks
  - `pathlib` - File operations
  - `sqlite3` - Metadata caching
  - `threading` - Parallel fetching
  - `xml.etree` - XML parsing

## License

The data generator framework is part of Aisupea. Generated data respects all original source licenses:
- Wikipedia: CC-BY-SA 3.0 (attribution required)
- Common Crawl: CC0 (public domain)
- Open Library: CC0/CC-BY (varies)
- GitHub: Original repository licenses
- ArXiv: Creative Commons (open access)
- Project Gutenberg: Public domain

## Support

For issues, questions, or contributions:
1. Check [DATA_GENERATION_GUIDE.md](../DATA_GENERATION_GUIDE.md) for usage
2. Review logs in `./aisupea_data/.logs/`
3. Enable debug mode in `DataGeneratorSettings`
4. Run diagnostic: `python -m data_generator stats`

---

**Ready to generate knowledge?**
```bash
python -m data_generator setup
python -m data_generator generate
```
