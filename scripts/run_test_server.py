#!/usr/bin/env python3
"""Run the test server for local bot testing.

The test server replays historical poker hands from JSONL log files.

Usage:
    python scripts/run_test_server.py

Environment Variables:
    TEST_SERVER_LOG - Path to JSONL log file (default: historical_logs/*.jsonl)
"""
import sys
import asyncio

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_server import main

if __name__ == "__main__":
    asyncio.run(main())
