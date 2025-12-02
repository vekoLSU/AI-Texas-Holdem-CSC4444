#!/usr/bin/env python3
"""
Run the competition bot with ensemble system.

Connects your existing Bot (with GTO/Exploiter/Defender ensemble)
to the competition infrastructure via WebSocket adapter.

Usage:
    python scripts/run_competition.py --player-id "my_bot"
    python scripts/run_competition.py --player-id "my_bot" --server "ws://192.168.1.100:8080"

Arguments:
    --player-id: Your unique player ID (required)
    --server: WebSocket server URL (default: ws://localhost:8080)
    --api-key: API key for authentication (default: dev)
    --table: Table ID to join (default: table-1)
    --verbose: Enable debug logging
"""
import sys
import asyncio
import argparse
import logging

# Add src to path so we can import poker_bot
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker_bot.core.competition_adapter import CompetitionAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Competition bot with ensemble system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--player-id",
        type=str,
        required=True,
        help="Your unique player ID"
    )

    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8080",
        help="WebSocket server URL"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="dev",
        help="API key for authentication"
    )

    parser.add_argument(
        "--table",
        type=str,
        default="table-1",
        help="Table ID to join"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║        🃏  COMPETITION BOT - ENSEMBLE SYSTEM  🃏        ║
║     GTO Agent | Exploiter Agent | Defender Agent        ║
╚══════════════════════════════════════════════════════════╝
    """)

    logger.info(f"Player ID: {args.player_id}")
    logger.info(f"Server: {args.server}")
    logger.info(f"Table: {args.table}")
    logger.info(f"API Key: {args.api_key}")
    logger.info(f"")
    logger.info(f"🤖 Initializing ensemble bot...")
    logger.info(f"   ✓ GTO Agent (balanced, unexploitable)")
    logger.info(f"   ✓ Exploiter Agent (aggressive vs weak opponents)")
    logger.info(f"   ✓ Defender Agent (defensive vs aggression)")
    logger.info(f"   ✓ Meta-Controller (agent selection)")
    logger.info(f"   ✓ Opponent Tracker (profiling)")
    logger.info(f"")

    # Create competition adapter
    adapter = CompetitionAdapter(
        api_key=args.api_key,
        table=args.table,
        player=args.player_id,
        server_url=args.server
    )

    try:
        logger.info(f"🚀 Starting bot...")
        await adapter.run()
        logger.info(f"✅ Bot finished successfully")

    except KeyboardInterrupt:
        logger.info(f"\n👋 Bot stopped by user")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
