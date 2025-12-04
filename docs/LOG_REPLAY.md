# Historical Hand Replay System

This document describes the poker hand replay system that allows the test server to replay real historical poker hands instead of generating synthetic ones.

## Important: Denomination Requirements

**This game uses whole dollar amounts only with minimum $10 bets.** Most online poker sites (including PokerStars) use cent denominations in their hand histories. When converting hand histories, you **must** use the `--scale` parameter to convert amounts to whole dollars:

```bash
# Required for this game: Scale $0.50/$1.00 to $10/$20
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 20
```

Without scaling, you'll get fractional dollar amounts that don't match the game's requirements.

## Overview

The replay system consists of:
1. **Sample hand history files** in PokerStars text format
2. **Converter script** that transforms hand histories to JSONL (with scaling support)
3. **Modified test_server.py** that replays hands from JSONL files

## Data Sources

### Included Sample Data

The repository includes sample PokerStars hand history files in `historical_logs/sample_pokerstars.txt`. These are realistic 6-max no-limit hold'em cash game hands that demonstrate various scenarios:

- Premium starting hands (AA, KK, AK)
- Drawing hands (flush draws, straight draws)
- Multi-street action (preflop through river)
- Different outcomes (wins, losses, folds)
- Varied pot sizes and stack depths

### Public Hand History Datasets

For larger datasets, see these publicly available sources:

- **[uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)** - 21M+ poker hands in PHH format from multiple sites including PokerStars
- **[Poker Hand History Zenodo Dataset](https://zenodo.org/records/13997158)** - Comprehensive poker hand dataset
- **PokerStars Hand History Export** - You can export your own hands from PokerStars client

### Adding Your Own Data

To add your own PokerStars hand histories:

1. Export hands from PokerStars (via client settings or email request)
2. Place the `.txt` file in `historical_logs/`
3. Run the converter (see below)

The converter supports standard PokerStars hand history format with these features:
- Cash games (no-limit, pot-limit, fixed-limit)
- Multiple betting rounds (preflop, flop, turn, river)
- 2-10 players
- Standard action types (fold, check, call, bet, raise)

## Hand History Format

### Input: PokerStars Text Format

The converter expects standard PokerStars hand history format:

```
PokerStars Hand #243891234567: Hold'em No Limit ($0.50/$1.00 USD) - 2023/07/15 14:23:11 ET
Table 'Hydra II' 6-max Seat #3 is the button
Seat 1: Hero ($150.00 in chips)
Seat 2: Villain1 ($98.50 in chips)
Seat 3: Villain2 ($205.00 in chips)
Villain1: posts small blind $0.50
Villain2: posts big blind $1
*** HOLE CARDS ***
Dealt to Hero [Kd Kh]
Hero: raises $3 to $4
Villain1: calls $3.50
Villain2: calls $3
*** FLOP *** [3d 6h 9c]
Villain1: checks
Villain2: checks
Hero: bets $8
...
```

### Output: JSONL Format

Each line in the output JSONL contains one complete hand with all phases (example with 20x scaling applied):

```json
{
  "hand_id": "243891234567",
  "small_blind": 10,
  "big_blind": 20,
  "button_seat": 3,
  "hero_name": "Hero",
  "hero_cards": [["K", "d"], ["K", "h"]],
  "players_initial": {"Hero": 3000, "Villain1": 1970, "Villain2": 4100},
  "phases": {
    "PREFLOP": {
      "phase": "PREFLOP",
      "pot": 30,
      "currentBet": 80,
      "communityCards": [],
      "players": [...]
    },
    "FLOP": {...},
    "TURN": {...},
    "RIVER": {...}
  },
  "showdown": {
    "winner": "Hero",
    "pot": 2160,
    "final_board": [["3", "d"], ["6", "h"], ["9", "c"], ["2", "s"], ["Q", "h"]]
  }
}
```

## Using the Converter

### Basic Usage (With Scaling for Whole Dollar Games)

**IMPORTANT:** This game uses whole dollar denominations with minimum $10 bets. Most PokerStars hand histories use cent denominations ($0.50/$1.00 blinds). Use the `--scale` parameter to convert to whole dollars:

```bash
# Convert $0.50/$1.00 blinds to $10/$20 blinds (20x scaling)
python3 tools/convert_hh_to_jsonl.py historical_logs/sample_pokerstars.txt historical_logs/sample_real_hands.jsonl --scale 20
```

This will multiply ALL amounts by 20:
- $0.50 small blind → $10
- $1.00 big blind → $20
- $150 stack → $3,000
- All bets, raises, and pots scaled proportionally

### Scaling Examples

```bash
# Scale $0.50/$1 to $10/$20 (recommended for this game)
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 20

# Scale $0.50/$1 to $5/$10
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 10

# Scale $1/$2 to $10/$20
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 10

# No scaling (use original amounts)
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl
```

### Specifying Hero Name

If your hand histories use a different player name than "Hero":

```bash
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --hero YourPlayerName --scale 20
```

### What the Converter Does

The converter will:
- Parse each hand from the input file
- Extract hole cards, community cards, and betting actions
- Apply scaling to all monetary amounts (blinds, stacks, bets, pots)
- Track pot size and chip stacks through all phases
- Record the winner and final outcome
- Write one JSONL entry per hand

### Converter Options

```bash
python3 tools/convert_hh_to_jsonl.py <input.txt> <output.jsonl> [OPTIONS]

Arguments:
  input.txt        - PokerStars hand history text file
  output.jsonl     - Output file in JSONL format

Options:
  --hero NAME      - Player name to track as hero (default: "Hero")
  --scale FACTOR   - Multiply all amounts by this factor (default: 1.0)
                     Use 20 for $10/$20 blinds, 10 for $5/$10 blinds
  --help           - Show detailed help message
```

## Running the Test Server

### Default Mode

By default, the server uses the first JSONL file found in `historical_logs/`:

```bash
python3 test_server.py
```

Output:
```
📂 Loading hands from: historical_logs/sample_real_hands.jsonl
✅ Loaded 8 hands
============================================================
🎰  Poker Test Server - Historical Hand Replay  🎰
============================================================

Replaying 8 hands from real poker history
Server starting on ws://localhost:8080
Waiting for bot connections...
```

### Using Environment Variable

Set `TEST_SERVER_LOG` to specify which file to use:

```bash
export TEST_SERVER_LOG=historical_logs/my_custom_hands.jsonl
python3 test_server.py
```

Or inline:

```bash
TEST_SERVER_LOG=historical_logs/sample_real_hands.jsonl python3 test_server.py
```

### Using Command Line Argument

Pass the log file path directly:

```bash
python3 test_server.py historical_logs/sample_real_hands.jsonl
```

### Priority Order

The server checks for log files in this order:
1. Command line argument (highest priority)
2. `TEST_SERVER_LOG` environment variable
3. First `.jsonl` file in `historical_logs/` directory
4. Error if none found

## How Hand Replay Works

When a bot connects, the server:

1. **Loads all hands** from the JSONL file into memory
2. **Iterates through each hand** sequentially
3. **For each betting phase** (PREFLOP → FLOP → TURN → RIVER):
   - Sends a `state` message with current game state
   - Waits up to 10 seconds for bot's action
   - If bot folds or times out, ends hand early
4. **Sends showdown message** with final outcome
5. **Repeats** until all hands are replayed

### State Messages

Each state message contains:

```json
{
  "type": "state",
  "hand": 1,
  "phase": "FLOP",
  "pot": 60,
  "currentBet": 0,
  "cards": [["K", "d"], ["K", "h"]],
  "communityCards": [["3", "d"], ["6", "h"], ["9", "c"]],
  "currentPlayer": "TestBot",
  "players": [
    {
      "id": "TestBot",
      "chips": 150,
      "bet": 0,
      "folded": false,
      "position": 0,
      "cards": [["K", "d"], ["K", "h"]]
    },
    {
      "id": "Villain1",
      "chips": 98,
      "bet": 0,
      "folded": false,
      "position": 1,
      "cards": null
    }
  ]
}
```

### Showdown Messages

After the final betting round:

```json
{
  "type": "showdown",
  "winner": "TestBot",
  "pot": 150,
  "winning_hand": "Historical Result",
  "board": [["3", "d"], ["6", "h"], ["9", "c"], ["2", "s"], ["Q", "h"]],
  "player_cards": [["K", "d"], ["K", "h"]]
}
```

## Testing Your Bot

### Quick Test

Use the included sample data:

```bash
# Terminal 1: Start test server
python3 test_server.py

# Terminal 2: Run your bot
python3 bot.py
```

Your bot will face 8 real historical scenarios with varied:
- Starting hands (premium pairs, suited connectors, drawing hands)
- Board textures (dry, wet, coordinated)
- Pot sizes ($30-$5,730 pots) and stack depths ($1,420-$5,700)
- Multi-street betting sequences with whole dollar amounts

**Note:** The included sample data is pre-scaled to $10/$20 blinds (whole dollars).

### Extended Testing

For more thorough testing:

1. **Download larger datasets** from the sources listed above
2. **Convert them** using the converter script **with --scale 20**
3. **Configure the server** to use your dataset
4. **Run extensive tests** with thousands of hands

Example:

```bash
# Download and convert a large dataset (with scaling!)
curl -o historical_logs/large_dataset.txt https://example.com/pokerstars_hands.txt
python3 tools/convert_hh_to_jsonl.py historical_logs/large_dataset.txt historical_logs/large_dataset.jsonl --scale 20

# Run tests
TEST_SERVER_LOG=historical_logs/large_dataset.jsonl python3 test_server.py
```

## Troubleshooting

### No JSONL files found

**Error:**
```
❌ Error: No JSONL log files found. Please set TEST_SERVER_LOG or provide log file path.
```

**Solution:**
```bash
# Convert some hand histories first
python3 tools/convert_hh_to_jsonl.py historical_logs/sample_pokerstars.txt historical_logs/sample_real_hands.jsonl
```

### Converter returns 0 hands

**Issue:** Converter can't parse your hand history file.

**Common causes:**
- File format is not PokerStars standard format
- File uses different language/locale
- Hands are from tournaments (partial support)
- Hero name doesn't match

**Solution:**
```bash
# Specify correct hero name
python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl YourActualPlayerName

# Check file format matches PokerStars standard
head -20 input.txt
```

### Bot times out on every action

**Issue:** Bot takes longer than 10 seconds to respond.

**Solution:** The timeout is configurable in `test_server.py` line 127:
```python
action_msg = await asyncio.wait_for(
    websocket.recv(),
    timeout=10.0  # Increase this value
)
```

## File Structure

```
AI-Texas-Holdem-CSC4444/
├── historical_logs/
│   ├── sample_pokerstars.txt      # Sample PokerStars hand histories (raw)
│   └── sample_real_hands.jsonl    # Converted hands (ready for replay)
├── tools/
│   └── convert_hh_to_jsonl.py     # Converter script
├── test_server.py                  # Modified replay server
├── bot.py                          # Your poker bot
└── LOG_REPLAY.md                   # This documentation
```

## Advanced Usage

### Filtering Hands

You can modify the converter to filter specific types of hands:

```python
# In convert_hh_to_jsonl.py, add filtering logic
def should_include_hand(hand_data):
    # Only include hands where hero has premium pairs
    hero_cards = hand_data['hero_cards']
    ranks = [card[0] for card in hero_cards]
    return ranks[0] == ranks[1] and ranks[0] in ['A', 'K', 'Q']

# In convert_file function:
if hand_data and should_include_hand(hand_data):
    converted_hands.append(hand_data)
```

### Batch Conversion

Convert multiple files:

```bash
for file in historical_logs/*.txt; do
    python3 tools/convert_hh_to_jsonl.py "$file" "${file%.txt}.jsonl"
done
```

### Analyzing Replay Results

The server logs all bot actions. You can pipe output to analyze performance:

```bash
python3 test_server.py 2>&1 | tee replay_log.txt
grep "Bot action" replay_log.txt | wc -l  # Count total actions
```

## Additional Resources

- [PokerStars Hand History Format Specification](https://arxiv.org/html/2312.11753v2)
- [PHH Standard Format](https://github.com/uoftcprg/phh-std/)
- [Poker Hand History Parser Libraries](https://github.com/HHSmithy/PokerHandHistoryParser)

## Support

If you encounter issues:
1. Check that your input file matches PokerStars format
2. Verify the hero name in your hand histories
3. Test with the included `sample_pokerstars.txt` first
4. Check converter output for error messages

For bugs or feature requests, please open an issue in the repository.
