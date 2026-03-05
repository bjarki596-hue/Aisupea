"""
Aisupea Data Generator Runner

Main entry point for running the data generator with easy-to-use CLI.
Now supports large-scale data generation (5.5GB+).
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from data_generator import DataGenerator
from data_generator.loader import DataLoader, DataPipeline
from data_generator.config import DataGeneratorConfig, DataGeneratorSettings, setup_data_generator
from data_generator.enhanced_fetcher import EnhancedDataFetcher


class DataGeneratorCLI:
    """Command-line interface for data generator."""

    def __init__(self):
        self.config = DataGeneratorConfig.get_config()
        self.settings = DataGeneratorSettings()

        # Load saved settings if available
        if Path("data_generator_settings.json").exists():
            self.settings.load()

    def run_generate(self, args):
        """Run data generation."""
        print("\n" + "="*70)
        print("🚀 AISUPEA DATA GENERATION")
        print("="*70)

        # Create generator with settings
        generator = DataGenerator(base_path=self.settings.data_path)

        # Generate data
        if args.module:
            # Generate specific module
            print(f"\nGenerating data for module: {args.module}")
            success = generator.generate_module_data(args.module, force_update=args.force)
            if not success:
                print(f"❌ Failed to generate data for {args.module}")
                return 1
        else:
            # Generate all modules
            print(f"\nGenerating data for all modules ({len(self.config['modules'])} modules)...")
            success = generator.generate_all_modules(force_update=args.force)
            if not success:
                print(f"❌ Some modules failed to generate")
                return 1

        # Print statistics
        stats = generator.get_statistics()
        print("\n" + "="*70)
        print("📊 GENERATION STATISTICS")
        print("="*70)
        print(f"Modules generated: {stats['total_modules']}")
        print(f"Total files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.2f}MB / {stats['target_size_mb']}MB")
        print(f"Completion: {stats['completion_percent']:.1f}%")

        if args.verify:
            # Verify data integrity
            print("\n" + "="*70)
            print("🔍 VERIFYING DATA INTEGRITY")
            print("="*70)
            integrity = generator.verify_data_integrity()
            valid_count = sum(1 for v in integrity.values() if v)
            print(f"Valid modules: {valid_count}/{len(integrity)}")

        print("\n✅ Generation complete!\n")
        return 0

    def run_load(self, args):
        """Load and inspect data."""
        print("\n" + "="*70)
        print("📂 LOADING DATA")
        print("="*70)

        loader = DataLoader(self.settings.data_path)

        if args.module:
            # Load specific module
            print(f"\nLoading module: {args.module}")
            data = loader.load_module_data(args.module)
            print(f"Loaded {len(data)} data sources:")
            for source_name, source_data in data.items():
                data_type = type(source_data).__name__
                data_size = len(str(source_data))
                print(f"  - {source_name}: {data_type} ({data_size} bytes)")
        else:
            # Load all modules
            print("\nLoading all modules...")
            all_data = loader.get_all_modules_data()
            print(f"Loaded {len(all_data)} modules:")
            for module_name, sources in all_data.items():
                print(f"  - {module_name}: {len(sources)} sources")

        print("\n✅ Data loading complete!\n")
        return 0

    def run_process(self, args):
        """Process data for modules."""
        print("\n" + "="*70)
        print("⚙️ PROCESSING DATA")
        print("="*70)

        pipeline = DataPipeline(self.settings.data_path)

        if args.module:
            # Process specific module
            print(f"\nProcessing module: {args.module}")
            dataset = pipeline.get_module_dataset(args.module)
            print(f"Processed {len(dataset['sources'])} data sources")

            # Show processed data summary
            for source_name, source_data in dataset['sources'].items():
                print(f"  - {source_name}:")
                for key, value in source_data.items():
                    if isinstance(value, list):
                        print(f"    {key}: {len(value)} items")
                    else:
                        print(f"    {key}: {value}")
        else:
            # Process all modules
            print("\nProcessing all modules...")
            all_datasets = pipeline.get_all_datasets()
            print(f"Processed {len(all_datasets)} modules")

        if args.export:
            # Export processed data
            print(f"\nExporting processed data to {args.export}...")
            pipeline.export_processed_data(args.export)
            print(f"✓ Exported to {args.export}")

        print("\n✅ Data processing complete!\n")
        return 0

    def run_stats(self, args):
        """Show statistics about generated data."""
        print("\n" + "="*70)
        print("📊 DATA STATISTICS")
        print("="*70)

        generator = DataGenerator(base_path=self.settings.data_path)
        stats = generator.get_statistics()

        print(f"\nTotal modules: {stats['total_modules']}")
        print(f"Total files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.2f}MB / {stats['target_size_mb']}MB")
        print(f"Completion: {stats['completion_percent']:.1f}%")
        print(f"Last updated: {stats['last_updated'] or 'Never'}")

        # Show module breakdown
        print(f"\nModule breakdown:")
        for module_name in sorted(generator.list_available_modules()):
            module_data = generator.get_module_data(module_name)
            size_mb = module_data['size'] / 1024 / 1024
            num_files = len(module_data['files'])
            print(f"  - {module_name:12} {size_mb:6.2f}MB ({num_files} files)")

        print("\n✅ Statistics complete!\n")
        return 0

    def run_setup(self, args):
        """Run initial setup."""
        setup_data_generator()
        return 0

    def run_clean(self, args):
        """Clean old data."""
        print("\n" + "="*70)
        print("🧹 CLEANING DATA")
        print("="*70)

        generator = DataGenerator(base_path=self.settings.data_path)
        days = args.days if args.days else 30
        removed = generator.cleanup_old_data(days)

        print(f"\nRemoved {removed} old data files (older than {days} days)")
        print("\n✅ Cleanup complete!\n")
        return 0

    def run_large_scale(self, args):
        """Run large-scale data generation from multiple sources."""
        print("\n" + "="*70)
        print("🌍 AISUPEA LARGE-SCALE DATA GENERATION")
        print("="*70)
        
        fetcher = EnhancedDataFetcher(
            max_size_mb=self.settings.max_size_per_source_mb,
            timeout=self.settings.timeout
        )
        
        modules = args.module if args.module else list(self.config['modules'].keys())
        if isinstance(modules, str):
            modules = [modules]
        
        output_dir = self.settings.data_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Configuration:")
        print(f"  Mode: {self.settings.mode}")
        print(f"  Data scale: {self.settings.data_scale}")
        print(f"  Max per source: {self.settings.max_size_per_source_mb}MB")
        print(f"  Timeout: {self.settings.timeout}s")
        print(f"  Modules: {len(modules)}")
        
        print(f"\n🔄 Fetching from multiple sources...")
        
        # Fetch from all available sources
        sources_data = {
            'Wikipedia': fetcher.fetch_wikipedia_full(),
            'Common Crawl': fetcher.fetch_common_crawl(),
            'ArXiv': fetcher.fetch_arxiv_papers(),
            'GitHub': fetcher.fetch_github_repos(),
            'Open Library': fetcher.fetch_open_library(),
            'Gutenberg': fetcher.fetch_gutenberg_books(),
            'Stack Exchange': fetcher.fetch_stack_exchange(),
        }
        
        total_size_mb = 0
        
        # Distribute data across modules
        for module in modules:
            print(f"\n📁 Processing module: {module}")
            
            for source_name, items in sources_data.items():
                if not items:
                    print(f"  ⚠️ No data from {source_name}")
                    continue
                
                output_file, size_mb = fetcher.save_data(
                    module, source_name, items, output_dir
                )
                total_size_mb += size_mb
        
        print("\n" + "="*70)
        print(f"✅ LARGE-SCALE GENERATION COMPLETE")
        print("="*70)
        print(f"Total data generated: {total_size_mb:.2f}MB")
        print(f"Modules populated: {len(modules)}")
        print(f"Data location: {output_dir}")
        
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aisupea Data Generator - Fetch non-copyrighted knowledge from the internet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup and generate all data
  python -m data_generator.runner setup
  python -m data_generator.runner generate

  # Generate specific module
  python -m data_generator.runner generate --module brain

  # LARGE-SCALE generation (5.5GB+)
  python -m data_generator.runner large_scale
  python -m data_generator.runner large_scale --module brain reasoning thinking
  python -m data_generator.runner large_scale --scale xlarge

  # Load and inspect data
  python -m data_generator.runner load --module reasoning

  # Process data and export
  python -m data_generator.runner process --module knowledge --export ./processed

  # Show statistics
  python -m data_generator.runner stats

  # Clean old data
  python -m data_generator.runner clean --days 30
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup data generator')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate knowledge data')
    gen_parser.add_argument('--module', help='Specific module to generate')
    gen_parser.add_argument('--force', action='store_true', help='Force re-download')
    gen_parser.add_argument('--verify', action='store_true', help='Verify integrity')

    # Load command
    load_parser = subparsers.add_parser('load', help='Load and inspect data')
    load_parser.add_argument('--module', help='Specific module to load')

    # Process command
    proc_parser = subparsers.add_parser('process', help='Process data for modules')
    proc_parser.add_argument('--module', help='Specific module to process')
    proc_parser.add_argument('--export', help='Export processed data to path')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show data statistics')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old data')
    clean_parser.add_argument('--days', type=int, help='Remove data older than N days')

    # Large-scale generation command
    large_parser = subparsers.add_parser('large_scale', help='Generate large-scale knowledge base (5.5GB+)')
    large_parser.add_argument('--module', nargs='+', help='Specific modules to generate (space-separated)')
    large_parser.add_argument('--scale', choices=['small', 'large', 'xlarge'], default='large', help='Data scale')

    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        print("\n💡 Tip: Run 'python -m data_generator.runner setup' to get started\n")
        return 0

    # Create CLI and run command
    cli = DataGeneratorCLI()
    command_method = getattr(cli, f'run_{args.command}', None)

    if not command_method:
        print(f"Unknown command: {args.command}")
        return 1

    return command_method(args)


if __name__ == "__main__":
    sys.exit(main())