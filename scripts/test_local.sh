#!/bin/bash
#
# Local Testing Script for Competition Bot
#
# This script helps you test your bot against the competition server locally.
# It assumes you have the competition server repository cloned.
#
# Usage:
#   ./scripts/test_local.sh
#
# Or manually run the steps below
#

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🃏  LOCAL BOT TESTING SCRIPT  🃏                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Configuration
COMPETITION_REPO_PATH="${COMPETITION_REPO_PATH:-../Texas-HoldEm-Infrastructure}"
API_KEY="${API_KEY:-dev}"
TABLE_ID="${TABLE_ID:-table-1}"
BOT_PLAYER_ID="${BOT_PLAYER_ID:-test_bot_1}"

echo "Configuration:"
echo "  Competition Repo: $COMPETITION_REPO_PATH"
echo "  API Key: $API_KEY"
echo "  Table: $TABLE_ID"
echo "  Bot Player ID: $BOT_PLAYER_ID"
echo ""

# Check if competition repo exists
if [ ! -d "$COMPETITION_REPO_PATH" ]; then
    echo "❌ Error: Competition repository not found at $COMPETITION_REPO_PATH"
    echo ""
    echo "Please clone the competition repository:"
    echo "  git clone https://github.com/dtaing11/Texas-HoldEm-Infrastructure"
    echo ""
    echo "Or set COMPETITION_REPO_PATH environment variable:"
    echo "  export COMPETITION_REPO_PATH=/path/to/Texas-HoldEm-Infrastructure"
    exit 1
fi

# Start competition server in background
echo "🚀 Starting competition server..."
cd "$COMPETITION_REPO_PATH"

# Build if needed
if [ ! -f "./poker-server" ]; then
    echo "   Building server..."
    go build -o poker-server ./
fi

# Start server
export API_KEY="$API_KEY"
export START_KEY="supersecret"
./poker-server &
SERVER_PID=$!

echo "   ✓ Server started (PID: $SERVER_PID)"
echo ""

# Wait for server to start
sleep 2

# Return to bot directory
cd - > /dev/null

# Start bot
echo "🤖 Starting test bot..."
python scripts/run_competition.py \
    --player-id "$BOT_PLAYER_ID" \
    --api-key "$API_KEY" \
    --table "$TABLE_ID" \
    --verbose &
BOT_PID=$!

echo "   ✓ Bot started (PID: $BOT_PID)"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping processes..."

    if [ ! -z "$BOT_PID" ] && kill -0 $BOT_PID 2>/dev/null; then
        kill $BOT_PID
        echo "   ✓ Bot stopped"
    fi

    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID
        echo "   ✓ Server stopped"
    fi

    echo ""
    echo "👋 Test session ended"
}

# Register cleanup on script exit
trap cleanup EXIT INT TERM

echo "✅ Test environment running!"
echo ""
echo "Your bot is now connected to the competition server."
echo "The bot will play hands as they come."
echo ""
echo "Press Ctrl+C to stop the test."
echo ""
echo "To connect additional bots, run in another terminal:"
echo "  python scripts/run_competition.py --player-id 'bot_2'"
echo ""

# Wait for bot process (will run until manually stopped)
wait $BOT_PID
