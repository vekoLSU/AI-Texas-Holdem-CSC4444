# Competition Bot Setup Guide

## Overview

Your competition bot uses a sophisticated **ensemble system** with three specialist agents:
- **GTO Agent**: Balanced, unexploitable play (vs strong opponents)
- **Exploiter Agent**: Aggressive exploitation (vs weak opponents)
- **Defender Agent**: Defensive, variance-minimizing (vs aggressive opponents)

The **Meta-Controller** automatically selects the best agent based on opponent profiling.

---

## Prerequisites

1. **Python 3.8+** installed
2. **Competition server** repository cloned:
   ```bash
   git clone https://github.com/dtaing11/Texas-HoldEm-Infrastructure
   ```
3. **Go 1.23+** installed (for competition server)
4. **Dependencies** installed:
   ```bash
   cd AI-Texas-Holdem-CSC4444-dev
   pip install -r requirements.txt
   ```

---

## Quick Start

### Step 1: Start Competition Server

```bash
cd /path/to/Texas-HoldEm-Infrastructure
go build -o poker-server ./
export API_KEY=dev
export START_KEY=supersecret
./poker-server
```

Server runs on `ws://localhost:8080`

### Step 2: Run Your Bot

```bash
cd /path/to/AI-Texas-Holdem-CSC4444-dev
python scripts/run_competition.py --player-id "my_bot"
```

**That's it!** Your bot is now connected and playing.

---

## Command Line Options

```bash
python scripts/run_competition.py [OPTIONS]

Required:
  --player-id ID         Your unique player ID (e.g., "my_bot_v1")

Optional:
  --server URL           WebSocket server URL
                         Default: ws://localhost:8080

  --api-key KEY          API key for authentication
                         Default: dev

  --table ID             Table ID to join
                         Default: table-1

  --verbose              Enable debug logging
```

### Examples

**Basic usage:**
```bash
python scripts/run_competition.py --player-id "aggressive_bot"
```

**Remote server:**
```bash
python scripts/run_competition.py \
    --player-id "my_bot" \
    --server "ws://192.168.1.100:8080"
```

**Debug mode:**
```bash
python scripts/run_competition.py \
    --player-id "debug_bot" \
    --verbose
```

---

## Testing Locally

### Automated Testing Script

```bash
./scripts/test_local.sh
```

This script:
1. Starts the competition server
2. Launches your bot
3. Manages both processes
4. Cleans up on exit

Press `Ctrl+C` to stop.

### Manual Multi-Bot Testing

**Terminal 1: Start server**
```bash
cd Texas-HoldEm-Infrastructure
./poker-server
```

**Terminal 2: Start bot 1**
```bash
cd AI-Texas-Holdem-CSC4444-dev
python scripts/run_competition.py --player-id "bot_1"
```

**Terminal 3: Start bot 2**
```bash
python scripts/run_competition.py --player-id "bot_2"
```

**Terminal 4: Start bot 3**
```bash
python scripts/run_competition.py --player-id "bot_3"
```

---

## Architecture Overview

### System Flow

```
Competition Server (JSON/WebSocket)
         ↓
competition_adapter.py (Protocol Translation)
         ↓
bot.py (Ensemble Orchestrator)
         ↓
meta_controller.py (Agent Selection)
         ↓
[GTO Agent | Exploiter Agent | Defender Agent]
         ↓
opponent_tracker.py (Profiling)
         ↓
hand_evaluator.py (Hand Strength)
```

### Key Components

**1. Competition Adapter** (`src/poker_bot/core/competition_adapter.py`)
- WebSocket client
- JSON state translation (server format → bot format)
- JSON action translation (bot format → server format)

**2. Bot** (`src/poker_bot/core/bot.py`)
- Ensemble orchestrator
- Opponent tracking
- Hand evaluation
- Training data logging (optional)

**3. Meta-Controller** (`src/poker_bot/core/meta_controller.py`)
- Opponent classification
- Agent selection logic
- Ensemble voting (when uncertain)

**4. Specialist Agents** (`src/poker_bot/agents/`)
- `agent_gto.py`: GTO baseline strategy
- `agent_exploiter.py`: Aggressive exploitation
- `agent_defender.py`: Defensive variance minimization

**5. Support Systems**
- `opponent_tracker.py`: Tracks VPIP, aggression, tendencies
- `hand_evaluator.py`: Calculates hand strength, draws

---

## State Format

### Competition Server Sends (JSON)

```json
{
  "type": "state",
  "state": {
    "table": {
      "id": "table-1",
      "players": [
        {
          "id": "player1",
          "chips": 950,
          "action": "CALL",
          "cards": [
            {"rank": "A", "suit": "HEART"},
            {"rank": "K", "suit": "SPADE"}
          ]
        }
      ],
      "phase": "FLOP"
    },
    "pot": 150,
    "phase": "FLOP",
    "board": [
      {"rank": "Q", "suit": "DIAMOND"},
      {"rank": "J", "suit": "CLUB"},
      {"rank": "T", "suit": "HEART"}
    ],
    "toActIdx": 0
  }
}
```

### Bot Receives (Translated)

```python
{
    "type": "state",
    "phase": "FLOP",
    "pot": 150,
    "currentBet": 20,
    "cards": ["As", "Kh"],          # Our hole cards
    "communityCards": ["Qd", "Jc", "Th"],
    "currentPlayer": "player1",
    "players": [
        {"id": "player1", "chips": 950, "bet": 20, "folded": False},
        {"id": "player2", "chips": 900, "bet": 50, "folded": False}
    ]
}
```

---

## Action Format

### Bot Sends (Internal)

```python
{"type": "action", "action": "fold"}
{"type": "action", "action": "call"}
{"type": "action", "action": "check"}
{"type": "action", "action": "raise", "amount": 100}
```

### Competition Server Receives (Translated)

```json
{"type": "act", "action": "FOLD", "amount": 0}
{"type": "act", "action": "CALL", "amount": 0}
{"type": "act", "action": "CHECK", "amount": 0}
{"type": "act", "action": "RAISE", "amount": 100}
```

---

## Agent Selection Logic

The Meta-Controller automatically selects the best agent:

### Decision Tree

```
1. Confidence < 75%?
   → Use ENSEMBLE (vote from all 3 agents)

2. Chip leader?
   → Use DEFENDER (protect lead)

3. Short stack?
   → Use EXPLOITER (aggressive urgency)

4. Opponent type:
   - Weak/Random/Fish → EXPLOITER
   - Aggressive/LAG/Maniac → DEFENDER
   - Strong/TAG/GTO → GTO

5. Table dynamics:
   - Very aggressive (aggression > 2.5) → DEFENDER
   - Very loose (VPIP > 45%) → EXPLOITER

6. Default → GTO (safe baseline)
```

---

## Logging

### Console Output

Bot logs to console:
- Connection status
- Hand summaries
- Actions taken
- Showdown results
- Agent selection decisions
- Opponent tracking updates

### Training Data Logging (Optional)

Set environment variable to enable:
```bash
export POKER_ENABLE_TRAINING_LOGS=1
python scripts/run_competition.py --player-id "my_bot"
```

Logs saved to: `logs/training_decisions.jsonl`

---

## Troubleshooting

### Bot Won't Connect

**Problem:** `Connection error: [Errno 111] Connection refused`

**Solution:** Make sure competition server is running:
```bash
cd Texas-HoldEm-Infrastructure
./poker-server
```

---

### Wrong Action Format Error

**Problem:** Server rejects actions

**Check:** Recent fix ensures actions use:
- Type: `"act"` (not `"action"`)
- Action names: `"FOLD"`, `"CALL"`, `"CHECK"`, `"RAISE"` (uppercase)

If errors persist, check `competition_adapter.py` line 126-153.

---

### Bot Makes Bad Decisions

**Causes:**
1. **Not enough opponent data** - Needs 5+ hands to profile opponents
2. **Wrong agent selected** - Meta-controller learning opponent types
3. **State estimation errors** - Server doesn't expose all betting info

**Solutions:**
- Let bot play 10-20 hands to warm up profiling
- Check `--verbose` logs to see agent selection reasoning
- Verify state translation in adapter

---

### ImportError or ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'poker_bot'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

Or run from project root:
```bash
cd AI-Texas-Holdem-CSC4444-dev
python scripts/run_competition.py --player-id "my_bot"
```

---

## Performance Tips

### 1. Warm Up Period

The bot needs 10-20 hands to:
- Profile opponents accurately
- Learn player tendencies
- Calibrate aggression levels

**First few hands may be suboptimal** as it learns.

### 2. Agent Selection

Monitor agent selection with `--verbose`:
```bash
python scripts/run_competition.py --player-id "my_bot" --verbose
```

Look for lines like:
```
[ENSEMBLE] Agent: exploiter | Opponent: fish | Confidence: 0.85
```

### 3. Opponent Tracking

Bot tracks for each opponent:
- VPIP (Voluntarily Put Money In Pot)
- PFR (Preflop Raise)
- Aggression Factor
- Fold to C-Bet
- Player type classification

More hands = better profiling = better decisions.

---

## Competition Day Checklist

**Before the competition:**
- [ ] Test bot locally with `./scripts/test_local.sh`
- [ ] Verify bot connects to server successfully
- [ ] Confirm bot plays at least 10 hands without crashes
- [ ] Check that all three agents are working
- [ ] Review logs for any errors

**During the competition:**
- [ ] Start bot with your unique player ID
- [ ] Monitor console output for errors
- [ ] Note opponent types being detected
- [ ] Watch for agent selection decisions
- [ ] Don't interrupt bot during hands

**After the competition:**
- [ ] Review training logs (if enabled)
- [ ] Analyze which agents performed best
- [ ] Check opponent profiles learned
- [ ] Identify any bugs or issues

---

## Advanced Configuration

### Disable Training Logs

```bash
export POKER_ENABLE_TRAINING_LOGS=0
python scripts/run_competition.py --player-id "my_bot"
```

### Custom Log Path

```bash
export POKER_TRAINING_LOG="logs/my_custom_log.jsonl"
python scripts/run_competition.py --player-id "my_bot"
```

### Different API Key

```bash
python scripts/run_competition.py \
    --player-id "my_bot" \
    --api-key "production_key_123"
```

---

## Files Modified for Competition

### Created/Updated:
- ✅ `scripts/run_competition.py` - Competition entry point
- ✅ `scripts/test_local.sh` - Local testing script
- ✅ `COMPETITION_SETUP.md` - This guide

### Modified:
- ✅ `src/poker_bot/core/competition_adapter.py` - Fixed protocol issues
  - Changed action type from `"action"` to `"act"`
  - Fixed action names to uppercase (`"CALL"` not `"call"`)
  - Improved currentBet estimation

### Unchanged (Your Ensemble System):
- ✅ `src/poker_bot/core/bot.py` - Ensemble orchestrator
- ✅ `src/poker_bot/core/meta_controller.py` - Agent selection
- ✅ `src/poker_bot/agents/agent_gto.py` - GTO agent
- ✅ `src/poker_bot/agents/agent_exploiter.py` - Exploiter agent
- ✅ `src/poker_bot/agents/agent_defender.py` - Defender agent
- ✅ `src/poker_bot/evaluation/opponent_tracker.py` - Profiling
- ✅ `src/poker_bot/evaluation/hand_evaluator.py` - Hand evaluation

---

## Support

If you encounter issues:

1. **Check logs** with `--verbose` flag
2. **Test locally** with `./scripts/test_local.sh`
3. **Verify server** is running and accessible
4. **Check state translation** in `competition_adapter.py`
5. **Review console output** for error messages

---

## Summary

**Your bot is ready for competition!**

✅ Ensemble system with 3 specialist agents
✅ Automatic opponent profiling
✅ Dynamic agent selection
✅ WebSocket connection to competition server
✅ Full state/action translation
✅ Comprehensive logging

**To run:**
```bash
python scripts/run_competition.py --player-id "YOUR_BOT_NAME"
```

**Good luck! 🎰🃏**
