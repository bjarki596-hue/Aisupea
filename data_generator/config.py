"""
Aisupea Data Generator Setup and Configuration

Configuration for different data generation modes and settings.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class DataGeneratorConfig:
    """Configuration for data generator."""

    # Data source configurations (EXPANDED - much larger data volumes)
    SOURCES_CONFIG = {
        "Wikipedia Full": {
            "enabled": True,
            "priority": 1,
            "max_size_mb": 200,
            "description": "Full Wikipedia article text and metadata",
            "license": "CC-BY-SA 3.0",
            "url": "https://dumps.wikimedia.org/enwiki/latest/",
            "variants": ["articles", "talk_pages", "user_pages"],
            "modules": ["brain", "thinking", "knowledge", "agent", "interface", "reasoning"]
        },
        "Common Crawl": {
            "enabled": True,
            "priority": 2,
            "max_size_mb": 300,
            "description": "Common Crawl CC0 licensed web content (multiple WARC snapshots)",
            "license": "CC0",
            "url": "https://commoncrawl.s3.amazonaws.com/",
            "variants": ["2025-crawl", "2024-crawl", "2023-crawl"],
            "modules": ["core", "models", "training", "memory", "thinking"]
        },
        "Open Library Full": {
            "enabled": True,
            "priority": 3,
            "max_size_mb": 250,
            "description": "Open Library complete book metadata and descriptions",
            "license": "CC0, CC-BY, Public Domain",
            "url": "https://openlibrary.org/",
            "variants": ["metadata", "full_texts", "reviews"],
            "modules": ["thinking", "knowledge", "interface", "memory", "brain"]
        },
        "GitHub Repositories": {
            "enabled": True,
            "priority": 4,
            "max_size_mb": 300,
            "description": "GitHub top repositories and documentation",
            "license": "Various OSS licenses",
            "url": "https://github.com/",
            "variants": ["top_repos", "trending", "documentation", "readmes"],
            "modules": ["core", "models", "training", "inference", "agent", "reasoning"]
        },
        "ArXiv Papers": {
            "enabled": True,
            "priority": 5,
            "max_size_mb": 350,
            "description": "ArXiv open access papers (full texts and metadata)",
            "license": "Authors' copyrights, open access",
            "url": "https://arxiv.org/",
            "variants": ["cs", "math", "physics", "stat", "eess"],
            "modules": ["brain", "reasoning", "models", "training", "inference", "memory"]
        },
        "Project Gutenberg": {
            "enabled": True,
            "priority": 6,
            "max_size_mb": 200,
            "description": "Project Gutenberg public domain books (full texts)",
            "license": "Public Domain",
            "url": "https://www.gutenberg.org/",
            "variants": ["top_1000", "by_language", "by_genre"],
            "modules": ["thinking", "knowledge", "memory", "interface", "brain"]
        },
        "Stack Exchange": {
            "enabled": True,
            "priority": 7,
            "max_size_mb": 250,
            "description": "Stack Exchange Q&A data (technical and general)",
            "license": "CC-BY-SA 4.0",
            "url": "https://archive.org/details/stackexchange",
            "variants": ["stackoverflow", "askubuntu", "serverfault", "superuser"],
            "modules": ["core", "reasoning", "training", "agent", "inference"]
        },
        "YouTube Captions": {
            "enabled": True,
            "priority": 8,
            "max_size_mb": 200,
            "description": "YouTube video captions (CC0 and licensed)",
            "license": "Varies (CC0, CC-BY)",
            "url": "https://www.youtube.com/",
            "variants": ["educational", "tutorials", "lectures", "documentaries"],
            "modules": ["thinking", "training", "interface", "memory"]
        },
        "News Archives": {
            "enabled": True,
            "priority": 9,
            "max_size_mb": 150,
            "description": "CC-licensed news archives and journalism",
            "license": "CC-BY, CC-BY-SA",
            "url": "https://archive.org/details/news",
            "variants": ["global_news", "tech_news", "science_news"],
            "modules": ["knowledge", "reasoning", "interface", "agent"]
        },
        "Academic Papers": {
            "enabled": True,
            "priority": 10,
            "max_size_mb": 280,
            "description": "Open access academic papers (PubMed Central, SSRN, etc.)",
            "license": "Open Access licenses",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/",
            "variants": ["pubmed_central", "ssrn", "preprints"],
            "modules": ["brain", "reasoning", "models", "training", "inference", "memory"]
        },
        "Code Snippets": {
            "enabled": True,
            "priority": 11,
            "max_size_mb": 200,
            "description": "Open source code snippets and examples",
            "license": "OSS",
            "url": "https://github.com/",
            "variants": ["gist_repos", "sample_projects", "tutorials"],
            "modules": ["core", "training", "inference", "agent"]
        },
        "Documentation": {
            "enabled": True,
            "priority": 12,
            "max_size_mb": 180,
            "description": "Technical and software documentation",
            "license": "CC-BY, OSS licenses",
            "url": "https://readthedocs.org/",
            "variants": ["software_docs", "tutorials", "api_docs"],
            "modules": ["core", "models", "training", "interface"]
        }
    }

    # Module configurations
    MODULES_CONFIG = {
        "brain": {
            "description": "Consciousness and meta-reasoning module",
            "target_size_mb": 500,
            "data_types": ["text", "metadata", "philosophy", "research"],
            "required_sources": []
        },
        "reasoning": {
            "description": "Advanced reasoning engines",
            "target_size_mb": 500,
            "data_types": ["text", "code", "metadata", "logic", "proofs"],
            "required_sources": []
        },
        "thinking": {
            "description": "Higher-level cognitive processes",
            "target_size_mb": 500,
            "data_types": ["text", "philosophy", "psychology", "books", "essays"],
            "required_sources": []
        },
        "knowledge": {
            "description": "Knowledge representation and storage",
            "target_size_mb": 500,
            "data_types": ["text", "structured data", "encyclopedic", "semantic graphs"],
            "required_sources": []
        },
        "memory": {
            "description": "Memory systems for agents",
            "target_size_mb": 500,
            "data_types": ["text", "code", "examples", "historical data"],
            "required_sources": []
        },
        "core": {
            "description": "Core tensor and mathematical operations",
            "target_size_mb": 500,
            "data_types": ["code", "research papers", "documentation", "examples"],
            "required_sources": []
        },
        "models": {
            "description": "ML models and architectures",
            "target_size_mb": 500,
            "data_types": ["code", "research papers", "documentation", "examples", "benchmarks"],
            "required_sources": []
        },
        "training": {
            "description": "Training utilities and algorithms",
            "target_size_mb": 500,
            "data_types": ["code", "research papers", "tutorials", "examples", "datasets"],
            "required_sources": []
        },
        "inference": {
            "description": "Inference engines and generation",
            "target_size_mb": 500,
            "data_types": ["code", "documentation", "papers", "examples", "use cases"],
            "required_sources": []
        },
        "agent": {
            "description": "Autonomous agent architecture",
            "target_size_mb": 500,
            "data_types": ["text", "code", "strategies", "behaviors", "examples"],
            "required_sources": []
        },
        "interface": {
            "description": "CLI and user interfaces",
            "target_size_mb": 500,
            "data_types": ["code", "documentation", "UI patterns", "examples"],
            "required_sources": []
        }
    }

    # Generation modes (EXPANDED - much larger data volumes)
    GENERATION_MODES = {
        "fast": {
            "description": "Fast generation (smaller sources only) - ~500MB total",
            "timeout": 30,
            "sources": ["GitHub Repositories", "ArXiv Papers"],
            "verify_integrity": False,
            "max_items_per_source": 5000
        },
        "balanced": {
            "description": "Balanced generation (all sources) - ~3-4GB total",
            "timeout": 90,
            "sources": list(SOURCES_CONFIG.keys()),
            "verify_integrity": True,
            "max_items_per_source": 20000,
            "parallel_sources": 4
        },
        "thorough": {
            "description": "Thorough generation with full content - ~5.5GB+ total",
            "timeout": 240,
            "sources": list(SOURCES_CONFIG.keys()),
            "verify_integrity": True,
            "retry_count": 3,
            "max_items_per_source": 50000,
            "parallel_sources": 8,
            "include_variants": True
        }
    }

    @classmethod
    def get_config(cls) -> Dict:
        """Get full configuration."""
        return {
            "sources": cls.SOURCES_CONFIG,
            "modules": cls.MODULES_CONFIG,
            "generation_modes": cls.GENERATION_MODES,
            "total_capacity_mb": 5500,
            "target_modules": list(cls.MODULES_CONFIG.keys()),
        }

    @classmethod
    def save_config(cls, filepath: str = "data_generator_config.json"):
        """Save configuration to file."""
        config = cls.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str = "data_generator_config.json") -> Optional[Dict]:
        """Load configuration from file."""
        if not Path(filepath).exists():
            return None

        with open(filepath, 'r') as f:
            return json.load(f)


class DataGeneratorSettings:
    """User-configurable settings for data generation."""

    def __init__(self):
        self.mode = "balanced"  # fast, balanced, thorough
        self.data_path = "./aisupea_data"
        self.force_update = False
        self.parallel_downloads = 4
        self.verify_integrity = True
        self.cleanup_after = False
        self.max_retries = 3
        self.timeout = 90
        self.verbose = True
        self.data_scale = "large"  # small (500MB), large (3-4GB), xlarge (5.5GB+)
        self.max_size_per_source_mb = 2500  # Allow up to 2.5GB per source
        self.include_all_variants = True
        self.batch_size = 100  # Items per batch to fetch
        self.compression = "gzip"  # gzip or none

    def to_dict(self) -> Dict:
        """Convert settings to dictionary."""
        return {
            "mode": self.mode,
            "data_path": self.data_path,
            "force_update": self.force_update,
            "parallel_downloads": self.parallel_downloads,
            "verify_integrity": self.verify_integrity,
            "cleanup_after": self.cleanup_after,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "verbose": self.verbose,
            "data_scale": self.data_scale,
            "max_size_per_source_mb": self.max_size_per_source_mb,
            "include_all_variants": self.include_all_variants,
            "batch_size": self.batch_size,
            "compression": self.compression
        }

    def save(self, filepath: str = "data_generator_settings.json"):
        """Save settings to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Settings saved to {filepath}")

    def load(self, filepath: str = "data_generator_settings.json"):
        """Load settings from file."""
        if not Path(filepath).exists():
            return

        with open(filepath, 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        print(f"Settings loaded from {filepath}")


def setup_data_generator():
    """Setup data generator with default configuration."""
    print("\n" + "="*70)
    print("📊 AISUPEA DATA GENERATOR SETUP")
    print("="*70)

    # Create default configuration
    config = DataGeneratorConfig.get_config()

    print("\n✓ Default configuration created:")
    print(f"  Total capacity: {config['total_capacity_mb']}MB")
    print(f"  Modules: {len(config['modules'])}")
    print(f"  Data sources: {len(config['sources'])}")

    # Create default settings
    settings = DataGeneratorSettings()

    print("\n✓ Default settings created:")
    print(f"  Mode: {settings.mode}")
    print(f"  Data path: {settings.data_path}")
    print(f"  Timeout: {settings.timeout}s")
    print(f"  Max retries: {settings.max_retries}")

    # Create data directory
    data_dir = Path(settings.data_path)
    data_dir.mkdir(exist_ok=True)

    print(f"\n✓ Data directory created: {data_dir}")

    # Save configuration and settings
    DataGeneratorConfig.save_config()
    settings.save()

    print("\n" + "="*70)
    print("✅ SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python -m data_generator.runner")
    print("2. Or use in code:")
    print("   from data_generator import DataGenerator")
    print("   gen = DataGenerator()")
    print("   gen.generate_all_modules()")
    print("\n")

    return config, settings


if __name__ == "__main__":
    setup_data_generator()