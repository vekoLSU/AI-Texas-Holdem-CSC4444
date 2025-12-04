#!/usr/bin/env python3
"""Run the competition adapter.

Usage:
    python scripts/run_competition.py [apiKey] [table] [player] [serverUrl]

Example:
    python scripts/run_competition.py dev table-1 bot1 ws://localhost:8080
"""
import sys
import asyncio

# Add src to path so we can import poker_bot
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker_bot.core.competition_adapter import main

if __name__ == "__main__":
    asyncio.run(main())
