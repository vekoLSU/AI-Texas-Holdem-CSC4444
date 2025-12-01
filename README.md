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

## Training Setup
1. **Collect decisions** – Logging is enabled by default. Run the bot and it will write JSON lines to `logs/training_decisions.jsonl`. Customize the path with `POKER_TRAINING_LOG=/tmp/decisions.jsonl`.
2. **Train models** – Once you have data, run:
   ```bash
   python -m training.train_agent --log-path logs/training_decisions.jsonl --agent-name gto --epochs 20
   ```
   Trained weights are stored under `models/` for easy loading later.
3. **Key modules** – `training/data_collector.py` handles logging, `training/state_encoder.py` normalizes features, and `training/train_agent.py` provides a ready-to-run training loop using PyTorch.

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
