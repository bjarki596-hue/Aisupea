"""
Aisupea Data Generator - Main Entry Point

This module allows running the data generator with:
  python -m data_generator

Or with specific command:
  python -m data_generator.runner generate
"""

import sys
from .runner import main

if __name__ == "__main__":
    sys.exit(main())
