#!/usr/bin/env python
"""Quick test to verify data_generator syntax is fixed."""

import sys
sys.path.insert(0, '/workspaces/Aisupea')

try:
    from data_generator.runner import main, DataGeneratorCLI
    from data_generator.config import DataGeneratorConfig, DataGeneratorSettings
    from data_generator.enhanced_fetcher import EnhancedDataFetcher
    print("✅ All imports successful!")
    print(f"✅ main() function: {main}")
    print(f"✅ DataGeneratorCLI: {DataGeneratorCLI}")
    print(f"✅ DataGeneratorConfig: {DataGeneratorConfig}")
    print(f"✅ EnhancedDataFetcher: {EnhancedDataFetcher}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
