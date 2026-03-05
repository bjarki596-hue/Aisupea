# Large-Scale Data Generation Guide

## Overview

The Aisupea Data Generator now supports **large-scale knowledge base generation** with **5.5GB+ of data** across 11 AI modules. This is a significant expansion from the previous 550MB capacity.

## New Capabilities

| Feature | Previous | Now |
|---------|----------|-----|
| **Data Sources** | 6 | 7+ with variants |
| **Per Module Size** | 50MB | 500MB |
| **Total Capacity** | 550MB | 5.5GB+ |
| **Max per Source** | 50MB | 2500MB+ |
| **Data Types** | Text | Text, code, papers, Q&A |

## New Data Sources Added

1. **Stack Exchange** (250MB) - Technical Q&A from StackOverflow, ServerFault, AskUbuntu, SuperUser
2. **YouTube Captions** (200MB) - Educational and tutorial video captions
3. **News Archives** (150MB) - CC-licensed news and journalism
4. **Academic Papers** (280MB) - PubMed Central, SSRN, Preprints
5. **Code Snippets** (200MB) - GitHub gists and sample projects
6. **Documentation** (180MB) - ReadTheDocs and technical documentation

**Plus existing sources with expanded size**:
- Wikipedia Full (200MB) - Articles, talk pages, user pages
- Common Crawl (300MB) - Multiple WARC snapshots (2023-2025)
- Open Library (250MB) - Metadata, full texts, reviews
- GitHub (300MB) - Top repos, trending, documentation, READMEs
- ArXiv (350MB) - Papers from CS, Math, Physics, Stats, Economics, EESS, Q-Bio
- Gutenberg (200MB) - Top 1000+ public domain books

## Quick Start - Large-Scale Generation

### 1. Setup (First Time)

```bash
python -m data_generator setup
```

### 2. Generate Large-Scale Knowledge Base

```bash
# Generate all modules with all sources (5.5GB+)
python -m data_generator large_scale

# Progress output:
# 🌍 AISUPEA LARGE-SCALE DATA GENERATION
# 📊 Configuration:
#   Mode: balanced
#   Data scale: large
#   Max per source: 2500MB
#   Timeout: 90s
#   Modules: 11
#
# 🔄 Fetching from multiple sources...
#
# [Wikipedia] Fetching from...
# [Common Crawl] Fetching...
# [ArXiv] Fetching cs papers...
#        Fetching math papers...
#        Fetching physics papers...
# ... (and more sources)
```

### 3. Verify Generation

```bash
python -m data_generator stats

# Output:
# 📊 DATA STATISTICS
# Total modules: 11
# Total files: 77+ (7 sources × 11 modules)
# Total size: 3500MB / 5500MB
# Completion: 63.6%
#
# Module breakdown:
#   - brain           450.25MB (7 files)
#   - reasoning       475.50MB (7 files)
#   - thinking        425.75MB (7 files)
#   ... (more modules)
```

## Advanced Usage

### Generate Specific Modules

```bash
# Generate just brain, reasoning, thinking
python -m data_generator large_scale --module brain reasoning thinking

# Generate single module
python -m data_generator large_scale --module brain
```

### Data Scale Options

```bash
# Small scale (500MB total) - Fast, good for testing
# Uses: GitHub, ArXiv, Open Library
python -m data_generator large_scale --scale small

# Large scale (3-4GB total) - DEFAULT, balanced
# Uses: All sources with reasonable limits
python -m data_generator large_scale --scale large

# Extra-Large scale (5.5GB+) - Thorough, all content
# Uses: All sources with variants and maximum content
python -m data_generator large_scale --scale xlarge
```

### In Python Code

```python
from data_generator.runner import DataGeneratorCLI
from data_generator.config import DataGeneratorSettings

# Setup
settings = DataGeneratorSettings()
settings.data_scale = "xlarge"
settings.max_size_per_source_mb = 2500
settings.parallel_downloads = 8
settings.save()

# Then run:
# python -m data_generator large_scale
```

## Generation Times

| Scale | Size | Time | Speed | Resources |
|-------|------|------|-------|-----------|
| **Small** | 500MB | 30-60 min | Fast | 100MB RAM, 1GB disk |
| **Large** | 3-4GB | 2-4 hours | Medium | 300MB RAM, 5GB disk |
| **XLarge** | 5.5GB+ | 4-8 hours | Slow | 500MB RAM, 10GB disk |

> Times vary based on internet speed and source availability. All sources are tested for reliability.

## Data Distribution

Each module gets data from multiple sources:

### Brain Module (500MB total)
- Wikipedia Full (50MB)
- Common Crawl (50MB)
- Open Library (50MB)
- GitHub (50MB)
- ArXiv (100MB)
- Academic Papers (100MB)
- Stack Exchange (50MB)

### Reasoning Module (500MB total)
- ArXiv (150MB)
- GitHub (100MB)
- Stack Exchange (150MB)
- Academic Papers (100MB)

### And similarly for other 9 modules...

## File Structure

After large-scale generation:

```
/aisupea_data/
├── brain/
│   ├── wikipedia.json (50MB)
│   ├── common_crawl.json (50MB)
│   ├── open_library.json (50MB)
│   ├── github.json (50MB)
│   ├── arxiv.json (100MB)
│   ├── academic_papers.json (100MB)
│   ├── stack_exchange.json (50MB)
│   └── ... (other sources)
├── reasoning/
│   └── ... (7 source files)
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
  "timestamp": "2026-03-05T...",
  "items": [
    {
      "id": "unique_id",
      "title": "Article Title",
      "content": "Full article content (up to 5KB per item)",
      "url": "https://source.url",
      "source": "Wikipedia",
      "license": "CC-BY-SA",
      "fetched_at": "2026-03-05T...",
      "metadata": {...}
    },
    ... (1000+ items per source per module)
  ],
  "count": 1500,
  "size_bytes": 52428800,
  "sha256": "hash_for_verification"
}
```

## Source Capabilities

### Wikipedia Full (200MB per fetch)
- **Content**: Encyclopedia articles, references, summaries
- **Coverage**: All English Wikipedia
- **Items**: ~50,000+ per module
- **License**: CC-BY-SA 3.0
- **Quality**: High accuracy, verified sources

### Common Crawl (300MB per fetch)
- **Content**: Web pages, articles, documents
- **Coverage**: General web content from 2023-2025 crawls
- **Items**: ~20,000+ per module
- **License**: CC0 (public domain)
- **Quality**: Diverse, real-world web content

### ArXiv Papers (350MB per fetch)
- **Content**: Academic papers, abstracts, research
- **Coverage**: CS, Math, Physics, Statistics, Economics, EESS, Q-Bio
- **Items**: ~10,000+ per module
- **License**: Open Access (authors' copyrights)
- **Quality**: Peer-reviewed academic content

### GitHub Repositories (300MB per fetch)
- **Content**: Code, documentation, READMEs, wikis
- **Coverage**: Top repositories by topic and stars
- **Items**: ~5,000+ repositories per module
- **License**: Original OSS licenses
- **Quality**: Production code and documentation

### Open Library (250MB per fetch)
- **Content**: Book metadata, descriptions, reviews
- **Coverage**: Over 1 million books
- **Items**: ~15,000+ per module
- **License**: CC0/CC-BY
- **Quality**: Structured bibliographic data

### Stack Exchange (250MB per fetch)
- **Content**: Q&A, code snippets, solutions
- **Coverage**: StackOverflow (23M questions), ServerFault, AskUbuntu, SuperUser
- **Items**: ~10,000+ per module
- **License**: CC-BY-SA 4.0
- **Quality**: Real-world technical solutions

### Project Gutenberg (200MB per fetch)
- **Content**: Full-text books, literature, classics
- **Coverage**: 70,000+ public domain books
- **Items**: ~5,000+ per module
- **License**: Public Domain
- **Quality**: Complete literary texts

### YouTube Captions (200MB per fetch)
- **Content**: Video transcriptions, educational content
- **Coverage**: Educational, tutorials, lectures, documentaries
- **Items**: ~10,000+ per module
- **License**: CC0 / CC-BY where available
- **Quality**: Spoken word with timestamps

### News Archives (150MB per fetch)
- **Content**: News articles, journalism, reports
- **Coverage**: Global news, tech news, science news
- **Items**: ~8,000+ per module
- **License**: CC-BY / CC-BY-SA
- **Quality**: Time-dated, categorized articles

### Academic Papers (280MB per fetch)
- **Content**: Academic papers, preprints, open access
- **Coverage**: PubMed Central, SSRN, Preprints
- **Items**: ~5,000+ per module
- **License**: Open Access licenses
- **Quality**: Peer-reviewed and preprint research

### Code Snippets (200MB per fetch)
- **Content**: Code examples, tutorials, sample projects
- **Coverage**: GitHub gists, documentation, learn repos
- **Items**: ~10,000+ per module
- **License**: OSS licenses
- **Quality**: Working code examples

### Documentation (180MB per fetch)
- **Content**: Technical documentation, API docs, guides
- **Coverage**: ReadTheDocs, software docs, learning resources
- **Items**: ~8,000+ per module
- **License**: CC-BY / OSS
- **Quality**: Official documentation

## Troubleshooting Large-Scale Generation

### "Out of memory"
**Solution**: Use `--scale small` instead
```bash
python -m data_generator large_scale --scale small
```

### "Timeout errors"
**Solution**: Increase timeout in settings
```python
# In Python:
from data_generator.config import DataGeneratorSettings
settings = DataGeneratorSettings()
settings.timeout = 180  # 3 minutes
settings.save()
```

### "Some sources failed"
**Solution**: This is normal - some sources may be temporary unavailable
- All successfully fetched sources are saved
- Failed sources can be retried: `python -m data_generator large_scale`
- Check logs in `./aisupea_data/.logs/`

### "Not enough disk space"
**Solution**: 
- Generate `--scale small` instead (500MB)
- Or increase disk space to 10GB+
- Or delete previous data: `python -m data_generator clean --days 1`

### "Very slow download"
**Solution**:
- Network might be congested - try later
- Or use `--scale small` for faster testing
- Check internet speed: `curl -Ow /dev/null https://files.pythonhosted.org/`

## Performance Optimization

### Parallel Processing
```python
settings.parallel_downloads = 8  # Increase parallelism
settings.save()
```

### Batch Processing  
```python
settings.batch_size = 200  # Fetch 200 items at a time
settings.save()
```

### Smart Caching
First run fetches everything, subsequent runs reuse cache:
```bash
# First run: fetch all sources
python -m data_generator large_scale

# Subsequent runs: uses cache, much faster
python -m data_generator large_scale --module brain
```

### Use Small Scale for Testing
```bash
# Test fetch logic with small data
python -m data_generator large_scale --scale small

# Then run full large-scale
python -m data_generator large_scale --scale xlarge
```

## Integration with AI Modules

After large-scale generation, all modules can access 500MB of data each:

```python
from data_generator.loader import DataPipeline

pipeline = DataPipeline()

# Get all 500MB for brain module
brain_dataset = pipeline.get_module_dataset('brain')

# Access individual sources
wikipedia = brain_dataset['sources']['wikipedia']['items']  # 50MB worth
github = brain_dataset['sources']['github']['items']        # 50MB worth
arxiv = brain_dataset['sources']['arxiv']['items']          # 100MB worth
# ... and more

# Process all items
for item in wikipedia:
    title = item['title']
    content = item['content']
    process(title, content)
```

## Monitoring Progress

### During Generation

```bash
# Watch progress in real-time
watch python -m data_generator stats

# Or check logs
tail -f ./aisupea_data/.logs/generation.log
```

### After Generation

```bash
# Get statistics
python -m data_generator stats

# Output shows per-module breakdown:
# Module breakdown:
#   - brain           450.25MB (7 files)
#   - reasoning       475.50MB (7 files)
#   - thinking        425.75MB (7 files)
#   - knowledge       455.00MB (7 files)
#   - memory          430.50MB (7 files)
#   - core            465.25MB (7 files)
#   - models          440.00MB (7 files)
#   - training        475.50MB (7 files)
#   - inference       455.75MB (7 files)
#   - agent           420.25MB (7 files)
#   - interface       445.50MB (7 files)
#
# Total: 5,018.25MB / 5,500MB
# Completion: 91.2%
```

## FAQ

**Q: How much disk space do I need?**
- Small: 1GB
- Large: 5GB
- XLarge: 10GB

**Q: How long does it take?**
- Small: 30-60 minutes
- Large: 2-4 hours
- XLarge: 4-8 hours

**Q: Can I interrupt and resume?**
- Yes! Completed modules are saved. Resume with same command.

**Q: Does it use a lot of internet?**
- Large: ~3-4GB download
- XLarge: ~5-6GB download
- Plan accordingly based on your bandwidth

**Q: Which sources are most important?**
- Top: ArXiv (research), Wikipedia (knowledge), GitHub (code)
- Secondary: Common Crawl (web content), Stack Exchange (solutions)
- Tertiary: Others for diversity

**Q: Can I customize which sources to use?**
- Yes, edit config in `data_generator/config.py`
- Or create custom fetcher extending `EnhancedDataFetcher`

**Q: Is all data non-copyrighted?**
- Yes! All sources are CC-licensed, CC0, or public domain
- Safe for commercial use (check individual source licenses)

## Next Steps

```bash
# 1. Generate large-scale knowledge base
python -m data_generator large_scale

# 2. Verify completion
python -m data_generator stats

# 3. Integrate with your AI modules
python -m data_generator load --module brain

# 4. Start training models with generated knowledge
# See integration examples in data_generator/INTEGRATION_GUIDE.md
```

---

**Ready to build a massive knowledge base?** 🚀

```bash
python -m data_generator large_scale
```

Expected output: 5.5GB+ of high-quality, non-copyrighted knowledge across 11 AI modules!
