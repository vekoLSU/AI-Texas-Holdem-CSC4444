# 🃏 Texas Hold'em Poker Bot

An AI poker bot for Texas Hold'em tournaments. It uses advanced decision-making, hand evaluation, and opponent modeling.

## Features
- Hand strength and equity calculation
- GTO + exploitative strategy
- Opponent profiling (TAG, LAG, Fish, Rock, etc.)
- Position and pot odds awareness
- Dynamic aggression and bluffing

## Quick Start

### Requirements
- Python 3.7+
- websockets library

### Install
```powershell
pip install -r requirements.txt
```

### Run Test Server & Bot
```powershell
# Terminal 1: Start test server
python test_server.py

# Terminal 2: Start bot
python bot.py dev table-1 TestBot ws://localhost:8080
```

## Usage
- The bot will play 1,000 hands against simulated opponents.
- Results and stats are shown in the console.

## Competition Mode
- For real tournaments, use the official engine and adapter:
```powershell
python competition_adapter.py dev table-1 MyBot ws://localhost:8080
```

## Troubleshooting
- If you see connection errors, make sure the server is running first.
- To free up port 8080:
```powershell
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## Files
- bot.py: Main bot logic
- decision_engine.py: Decision making
- hand_evaluator.py: Hand strength
- opponent_tracker.py: Opponent modeling
- test_server.py: Local test server
- requirements.txt: Dependencies

---

Good luck at the tables!
