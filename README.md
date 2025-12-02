# Texas Hold'em Poker Bot

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
- websockets, numpy, pandas, torch, scikit-learn, tqdm

### Install
```bash
# Install package in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Run Test Server & Bot
```bash
# Terminal 1: Start test server
python scripts/run_test_server.py

# Terminal 2: Start bot
python scripts/run_bot.py dev table-1 TestBot ws://localhost:8080
```

## Usage

### Running the Bot
```bash
# Standard entry point
python scripts/run_bot.py [apiKey] [table] [player] [serverUrl]

# Example
python scripts/run_bot.py dev table-1 bot1 ws://localhost:8080

# Alternative: Python module
python -m poker_bot.core.bot dev table-1 bot1 ws://localhost:8080
```

### Competition Mode
For real tournaments with the Texas-HoldEm-Infrastructure:
```bash
python scripts/run_competition.py dev table-1 MyBot ws://localhost:8080
```

### Testing Locally
```bash
# Start the test server (replays historical hands)
python scripts/run_test_server.py

# Or set custom log file
TEST_SERVER_LOG=historical_logs/sample_real_hands.jsonl python scripts/run_test_server.py
```

## Training Setup

1. **Collect decisions** – Logging is enabled by default. Run the bot and it will write JSON lines to `logs/training_decisions.jsonl`. Customize the path with `POKER_TRAINING_LOG=/tmp/decisions.jsonl`.

2. **Train models** – Once you have data, run:
   ```bash
   python -m poker_bot.training.train_agent --log-path logs/training_decisions.jsonl --agent-name gto --epochs 20
   ```
   Trained weights are stored under `models/` for easy loading later.

3. **Key modules**:
   - `src/poker_bot/training/data_collector.py` - Decision logging
   - `src/poker_bot/training/state_encoder.py` - Feature normalization
   - `src/poker_bot/training/train_agent.py` - PyTorch training loop

## Project Structure

```
AI-Texas-Holdem-CSC4444/
├── scripts/              # Entry point scripts
│   ├── run_bot.py
│   ├── run_competition.py
│   └── run_test_server.py
├── src/poker_bot/        # Main package
│   ├── core/             # Bot infrastructure
│   ├── agents/           # Strategy agents (GTO, Exploiter, Defender)
│   ├── evaluation/       # Hand & opponent evaluation
│   └── training/         # ML training infrastructure
├── tests/                # Test infrastructure
├── tools/                # Utility scripts
├── docs/                 # Additional documentation
└── logs/                 # Training logs (runtime)
```

## Documentation

- [ENSEMBLE_STRATEGY.md](ENSEMBLE_STRATEGY.md) - Multi-agent ensemble architecture
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions and options
- [COMPETITION_ANALYSIS.md](COMPETITION_ANALYSIS.md) - Competition strategy
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Universal opponent detection
- [TOURNAMENT_MATH.md](TOURNAMENT_MATH.md) - Poker math reference
- Additional docs in [docs/](docs/) directory

## Troubleshooting

- If you see connection errors, make sure the server is running first.
- If imports fail, ensure you've installed the package: `pip install -e .`
- To free up port 8080:
```bash
# macOS/Linux
lsof -ti:8080 | xargs kill -9

# Windows PowerShell
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```
