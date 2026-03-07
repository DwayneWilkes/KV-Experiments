"""Pytest configuration: ensure the stats directory is importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
