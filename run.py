#!/usr/bin/env python3
"""Candor Take-Home: One-command execution entrypoint."""
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from candor.cli import main

if __name__ == "__main__":
    main()
