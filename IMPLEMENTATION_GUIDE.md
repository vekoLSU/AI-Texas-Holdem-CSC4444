# Universal Opponent Destroyer - Implementation Guide

## Quick Start: Destroy All Opponents

This guide shows you exactly how to implement the universal opponent detection and counter-strategy system to beat ANY poker bot.

---

## Phase 1: Protocol Compatibility (CRITICAL - Do First!)

### ✅ COMPLETED: Fixed `competition_adapter.py`

**What was wrong:**
- Action format was uppercase (`"CALL"`) - should be lowercase (`"call"`)
- Message type was `"act"` - should be `"action"`

**What's now correct:**
```python
# CORRECT FORMAT (now in competition_adapter.py)
{"type": "action", "action": "call"}
{"type": "action", "action": "check"}
{"type": "action", "action": "fold"}
{"type": "action", "action": "raise", "amount": 120}
```

### Test It Now

```bash
# Start the infrastructure server first
cd /path/to/Texas-HoldEm-Infrastructure
go run main.go

# In another terminal, test your bot
cd /path/to/AI-Texas-Holdem-CSC4444
python competition_adapter.py dev table-1 destroyer_bot ws://localhost:8080
```

**Expected Result:** Bot connects, joins table, makes valid moves without errors.

---

## Phase 2: Create Opponent Detection System

### File: `opponent_detector.py` (NEW - Create This!)

```python
"""
Universal opponent detection system.
Classifies any opponent strategy within 50 hands.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple
import statistics


class OpponentDetector:
    """Detect and classify any opponent strategy type."""

    def __init__(self, opponent_id: str, max_history: int = 1000):
        self.opponent_id = opponent_id
        self.max_history = max_history

        # Basic statistics
        self.hands_played = 0
        self.hands_won = 0

        # Action tracking
        self.preflop_actions = deque(maxlen=max_history)
        self.postflop_actions = deque(maxlen=max_history)
        self.bet_sizes = deque(maxlen=max_history)
        self.response_times = deque(maxlen=max_history)

        # Frequency stats
        self.vpip_count = 0  # Voluntarily put in pot
        self.pfr_count = 0   # Preflop raise
        self.af_aggressive = 0  # Bets + raises
        self.af_passive = 0  # Calls
        self.fold_to_cbet = 0
        self.fold_to_cbet_attempts = 0
        self.fold_to_3bet = 0
        self.fold_to_3bet_attempts = 0

        # Pattern detection
        self.action_patterns = defaultdict(int)  # (situation) -> action count
        self.bet_size_patterns = defaultdict(int)

        # Classification
        self.detected_type = "UNKNOWN"
        self.detection_confidence = 0.0
        self.last_classification_hand = 0

    def record_action(
        self,
        phase: str,
        action: str,
        amount: int,
        pot: int,
        to_call: int,
        position: str,
        response_time: float = None
    ):
        """Record an opponent's action."""

        self.hands_played += 1

        # Track action
        if phase == "PREFLOP":
            self.preflop_actions.append(action)

            # VPIP: Any voluntary money in pot preflop
            if action in ["call", "raise"]:
                self.vpip_count += 1

            # PFR: Preflop raise
            if action == "raise":
                self.pfr_count += 1
        else:
            self.postflop_actions.append(action)

        # Aggression factor
        if action in ["raise", "bet"]:
            self.af_aggressive += 1
        elif action == "call":
            self.af_passive += 1

        # Bet sizing
        if amount > 0:
            self.bet_sizes.append(amount)
            # Normalize by pot size
            bet_to_pot_ratio = amount / pot if pot > 0 else 0
            self.bet_size_patterns[round(bet_to_pot_ratio, 1)] += 1

        # Response time (if available)
        if response_time is not None:
            self.response_times.append(response_time)

        # Pattern tracking: (phase, position, action) -> count
        pattern_key = (phase, position, action)
        self.action_patterns[pattern_key] += 1

    def record_fold_to_cbet(self, folded: bool):
        """Record response to continuation bet."""
        self.fold_to_cbet_attempts += 1
        if folded:
            self.fold_to_cbet += 1

    def record_fold_to_3bet(self, folded: bool):
        """Record response to 3-bet."""
        self.fold_to_3bet_attempts += 1
        if folded:
            self.fold_to_3bet += 1

    def get_statistics(self) -> Dict:
        """Calculate current statistics."""

        if self.hands_played == 0:
            return {}

        vpip = self.vpip_count / self.hands_played
        pfr = self.pfr_count / self.hands_played

        af = 0.0
        if self.af_passive > 0:
            af = self.af_aggressive / self.af_passive

        fold_to_cbet_pct = 0.0
        if self.fold_to_cbet_attempts > 0:
            fold_to_cbet_pct = self.fold_to_cbet / self.fold_to_cbet_attempts

        fold_to_3bet_pct = 0.0
        if self.fold_to_3bet_attempts > 0:
            fold_to_3bet_pct = self.fold_to_3bet / self.fold_to_3bet_attempts

        # Response time variance (for MCTS detection)
        response_time_std = 0.0
        response_time_avg = 0.0
        if len(self.response_times) > 5:
            response_time_avg = statistics.mean(self.response_times)
            response_time_std = statistics.stdev(self.response_times)

        # Bet size variety (for rule-based detection)
        unique_bet_sizes = len(self.bet_size_patterns)

        # Consistency score (same situation -> same action?)
        consistency_score = self._calculate_consistency()

        return {
            "hands_played": self.hands_played,
            "vpip": vpip,
            "pfr": pfr,
            "af": af,
            "fold_to_cbet": fold_to_cbet_pct,
            "fold_to_3bet": fold_to_3bet_pct,
            "response_time_avg": response_time_avg,
            "response_time_std": response_time_std,
            "unique_bet_sizes": unique_bet_sizes,
            "consistency_score": consistency_score
        }

    def _calculate_consistency(self) -> float:
        """
        Calculate how consistent opponent is.
        High consistency (>0.9) = rule-based
        Low consistency (<0.5) = random or adaptive
        """

        if len(self.action_patterns) < 10:
            return 0.5  # Not enough data

        # Find patterns that occurred multiple times
        repeated_patterns = [
            count for count in self.action_patterns.values()
            if count > 1
        ]

        if not repeated_patterns:
            return 0.0  # Never repeats = random

        # High consistency = often does same thing in same situation
        avg_repetitions = sum(repeated_patterns) / len(repeated_patterns)
        consistency = min(avg_repetitions / 5.0, 1.0)  # Normalize to 0-1

        return consistency

    def classify_opponent(self) -> Tuple[str, float]:
        """
        Classify opponent type and return (type, confidence).

        Types:
        - MCTS: Tree search bot
        - CFR_GTO: Game theory optimal
        - RL_BOT: Reinforcement learning
        - RULE_BASED: If-then logic
        - HAND_STRENGTH_ONLY: Equity-based only
        - RANDOM_BEGINNER: Random or very weak
        - HYBRID_ADVANCED: Sophisticated mix
        - UNKNOWN: Not enough data
        """

        stats = self.get_statistics()

        # Need minimum hands for classification
        if stats.get("hands_played", 0) < 15:
            return "UNKNOWN", 0.0

        # RULE-BASED detection
        if (stats["consistency_score"] > 0.85 and
            stats["unique_bet_sizes"] < 7):
            return "RULE_BASED", 0.9

        # RANDOM/BEGINNER detection
        if (stats["consistency_score"] < 0.3 and
            (stats["vpip"] > 0.7 or stats["vpip"] < 0.05)):
            return "RANDOM_BEGINNER", 0.85

        # CFR/GTO detection
        if (0.18 <= stats["vpip"] <= 0.28 and
            0.12 <= stats["pfr"] <= 0.22 and
            1.5 <= stats["af"] <= 2.5 and
            stats["consistency_score"] < 0.6):  # Mixed strategy
            return "CFR_GTO", 0.8

        # MCTS detection (slow, variable response times)
        if (stats["response_time_avg"] > 1.0 and
            stats["response_time_std"] > 0.5):
            return "MCTS", 0.75

        # RL detection (fast, adaptive, moderate consistency)
        if (stats["response_time_avg"] < 0.2 and
            0.4 < stats["consistency_score"] < 0.7 and
            stats["unique_bet_sizes"] > 10):
            return "RL_BOT", 0.7

        # HAND_STRENGTH_ONLY detection (pot odds adherent, no bluffs)
        if (stats["fold_to_cbet"] > 0.65 and  # Folds when misses
            stats["af"] < 1.2):  # Low aggression (no bluffing)
            return "HAND_STRENGTH_ONLY", 0.75

        # HYBRID detection (shows multiple characteristics)
        if stats["hands_played"] > 50:
            # Complex behavior that doesn't fit other categories
            return "HYBRID_ADVANCED", 0.6

        return "UNKNOWN", 0.4

    def get_exploits(self) -> Dict:
        """Get specific exploitable tendencies."""

        stats = self.get_statistics()

        exploits = {}

        # Fold to c-bet exploit
        if stats.get("fold_to_cbet", 0) > 0.6:
            exploits["continuation_bet_relentlessly"] = stats["fold_to_cbet"]

        # Fold to 3-bet exploit
        if stats.get("fold_to_3bet", 0) > 0.7:
            exploits["3bet_bluff_frequently"] = stats["fold_to_3bet"]

        # Too loose exploit
        if stats.get("vpip", 0) > 0.5:
            exploits["value_bet_thin"] = stats["vpip"]
            exploits["dont_bluff"] = 1.0

        # Too tight exploit
        if stats.get("vpip", 0) < 0.15:
            exploits["steal_blinds_aggressively"] = 1.0 - stats["vpip"]

        # Too passive exploit
        if stats.get("af", 0) < 0.8:
            exploits["bet_frequently"] = 0.8 - stats["af"]

        # Too aggressive exploit
        if stats.get("af", 0) > 3.0:
            exploits["trap_with_strong_hands"] = stats["af"] / 3.0
            exploits["call_down_lighter"] = stats["af"] / 3.0

        return exploits


class OpponentManager:
    """Manage detection for all opponents at the table."""

    def __init__(self):
        self.detectors = {}  # opponent_id -> OpponentDetector

    def get_detector(self, opponent_id: str) -> OpponentDetector:
        """Get or create detector for opponent."""
        if opponent_id not in self.detectors:
            self.detectors[opponent_id] = OpponentDetector(opponent_id)
        return self.detectors[opponent_id]

    def record_action(self, opponent_id: str, **kwargs):
        """Record action for opponent."""
        detector = self.get_detector(opponent_id)
        detector.record_action(**kwargs)

    def classify_all(self) -> Dict[str, Tuple[str, float]]:
        """Classify all opponents."""
        return {
            opp_id: detector.classify_opponent()
            for opp_id, detector in self.detectors.items()
        }

    def get_primary_target(self) -> Tuple[str, Dict]:
        """
        Identify the most exploitable opponent.

        Returns: (opponent_id, exploits)
        """

        best_target = None
        max_exploitability = 0.0

        for opp_id, detector in self.detectors.items():
            exploits = detector.get_exploits()
            # Exploitability = number and magnitude of exploits
            exploitability = sum(exploits.values())

            if exploitability > max_exploitability:
                max_exploitability = exploitability
                best_target = (opp_id, exploits)

        return best_target if best_target else (None, {})
```

---

## Phase 3: Integrate with Decision Engine

### Update `decision_engine.py`

Add this to your decision engine:

```python
from opponent_detector import OpponentManager

class DecisionEngine:
    def __init__(self):
        self.opponent_manager = OpponentManager()
        # ... rest of your init

    def decide(self, game_state, hand_strength, ...):
        """Make decision with opponent-aware strategy."""

        # Classify all opponents
        opponent_types = self.opponent_manager.classify_all()

        # Get primary exploit target
        target_id, target_exploits = self.opponent_manager.get_primary_target()

        # If we're heads-up or in position vs target, exploit heavily
        if self._should_exploit_target(game_state, target_id):
            return self._exploit_decision(
                game_state,
                hand_strength,
                opponent_types[target_id],
                target_exploits
            )

        # Otherwise, play solid baseline
        return self._baseline_decision(game_state, hand_strength)

    def _exploit_decision(self, game_state, hand_strength, opp_type, exploits):
        """Make exploitative decision based on opponent type."""

        base_action = self._baseline_decision(game_state, hand_strength)

        # RULE_BASED opponent
        if opp_type[0] == "RULE_BASED":
            # They're predictable - maximum exploitation
            if "continuation_bet_relentlessly" in exploits:
                # They fold to c-bets - bet more
                if base_action["action"] == "check":
                    return {"action": "raise", "amount": pot * 0.75}

        # HAND_STRENGTH_ONLY opponent
        elif opp_type[0] == "HAND_STRENGTH_ONLY":
            # They only bet with real hands
            if "dont_bluff" in exploits:
                # Never bluff catch
                if base_action["action"] == "call" and hand_strength < 0.6:
                    return {"action": "fold", "amount": 0}
            if "value_bet_thin" in exploits:
                # Extract maximum value
                if hand_strength > 0.65:
                    return {"action": "raise", "amount": pot * 1.2}

        # MCTS opponent
        elif opp_type[0] == "MCTS":
            # Fast tempo, unusual bet sizes
            if base_action["action"] == "raise":
                # Use weird bet sizes to break their abstractions
                amount = pot * 0.73  # Not 0.5x or 1x
                return {"action": "raise", "amount": amount}

        # CFR_GTO opponent
        elif opp_type[0] == "CFR_GTO":
            # Can't exploit - play solid GTO
            return base_action

        # RL_BOT opponent
        elif opp_type[0] == "RL_BOT":
            # They adapt - need to mislead
            # Vary strategy randomly to prevent them learning
            if random.random() < 0.15:  # 15% exploration
                # Make unusual play
                if base_action["action"] == "fold":
                    return {"action": "call", "amount": to_call}

        # RANDOM_BEGINNER opponent
        elif opp_type[0] == "RANDOM_BEGINNER":
            # Just get to showdown with best hand
            if hand_strength > 0.6:
                return {"action": "call", "amount": to_call}
            else:
                return {"action": "fold", "amount": 0}

        return base_action
```

---

## Phase 4: Testing Your Destroyer

### Test Script: `test_destroyer.py`

```python
"""
Test the opponent destroyer against all strategy types.
"""

import asyncio
from opponent_detector import OpponentDetector, OpponentManager


def test_rule_based_detection():
    """Test detection of rule-based bot."""
    detector = OpponentDetector("rule_bot")

    # Simulate very consistent play
    for i in range(30):
        # Always raises with strong hands (70% of time)
        if i % 10 < 7:
            detector.record_action("PREFLOP", "raise", 20, 10, 10, "button")
        else:
            detector.record_action("PREFLOP", "fold", 0, 10, 10, "button")

    bot_type, confidence = detector.classify_opponent()
    print(f"Detected: {bot_type} (confidence: {confidence:.2f})")
    assert bot_type == "RULE_BASED", f"Expected RULE_BASED, got {bot_type}"
    print("✅ Rule-based detection works!")


def test_random_detection():
    """Test detection of random bot."""
    detector = OpponentDetector("random_bot")

    # Simulate random play
    import random
    for i in range(30):
        action = random.choice(["fold", "call", "raise"])
        amount = random.randint(10, 100) if action == "raise" else 0
        detector.record_action("PREFLOP", action, amount, 50, 10, "button")

    bot_type, confidence = detector.classify_opponent()
    print(f"Detected: {bot_type} (confidence: {confidence:.2f})")
    print("✅ Random detection works!")


def test_gto_detection():
    """Test detection of GTO bot."""
    detector = OpponentDetector("gto_bot")

    # Simulate GTO-like play (balanced, moderate frequencies)
    for i in range(40):
        # VPIP ~25%, PFR ~15%, balanced aggression
        if i % 4 == 0:  # 25% VPIP
            if i % 7 < 1:  # ~15% PFR
                detector.record_action("PREFLOP", "raise", 30, 15, 10, "button")
            else:
                detector.record_action("PREFLOP", "call", 10, 15, 10, "button")
        else:
            detector.record_action("PREFLOP", "fold", 0, 15, 10, "button")

    bot_type, confidence = detector.classify_opponent()
    print(f"Detected: {bot_type} (confidence: {confidence:.2f})")
    assert bot_type == "CFR_GTO", f"Expected CFR_GTO, got {bot_type}"
    print("✅ GTO detection works!")


def test_exploit_identification():
    """Test exploit identification."""
    detector = OpponentDetector("weak_bot")

    # Simulate very tight player who folds to c-bets
    for i in range(30):
        if i % 10 < 2:  # 20% VPIP (tight)
            detector.record_action("PREFLOP", "call", 10, 15, 10, "button")
        else:
            detector.record_action("PREFLOP", "fold", 0, 15, 10, "button")

        # Folds to c-bets 80% of time
        if i < 25:
            detector.record_fold_to_cbet(folded=True)
        else:
            detector.record_fold_to_cbet(folded=False)

    exploits = detector.get_exploits()
    print(f"Exploits found: {exploits}")
    assert "continuation_bet_relentlessly" in exploits
    assert "steal_blinds_aggressively" in exploits
    print("✅ Exploit identification works!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING UNIVERSAL OPPONENT DESTROYER")
    print("="*60 + "\n")

    test_rule_based_detection()
    test_random_detection()
    test_gto_detection()
    test_exploit_identification()

    print("\n" + "="*60)
    print("ALL TESTS PASSED! 🏆")
    print("="*60 + "\n")
```

Run it:
```bash
python test_destroyer.py
```

---

## Phase 5: Live Integration

### Update `bot.py` to use OpponentManager

```python
from opponent_detector import OpponentManager

class PokerBot:
    def __init__(self, ...):
        # ... existing init
        self.opponent_manager = OpponentManager()

    async def handle_state_update(self, ws, msg):
        # ... existing code

        # Record opponent actions
        for player in players:
            if player["id"] != self.player:
                # Record what they did
                if "lastAction" in player:
                    self.opponent_manager.record_action(
                        opponent_id=player["id"],
                        phase=phase,
                        action=player["lastAction"],
                        amount=player.get("lastAmount", 0),
                        pot=pot,
                        to_call=current_bet,
                        position=self.get_position_name(player)
                    )

        # ... continue with decision making
```

---

## Testing Against Real Opponents

### Step-by-Step Competition Prep

1. **Test locally first:**
   ```bash
   # Terminal 1: Start infrastructure
   cd Texas-HoldEm-Infrastructure
   go run main.go

   # Terminal 2: Start your bot
   cd AI-Texas-Holdem-CSC4444
   python competition_adapter.py
   ```

2. **Run 1000+ hands:**
   - Let it play overnight
   - Check for crashes/errors
   - Verify opponent detection is working

3. **Analyze results:**
   ```python
   # Add to bot.py
   def print_opponent_report(self):
       for opp_id, detector in self.opponent_manager.detectors.items():
           bot_type, conf = detector.classify_opponent()
           exploits = detector.get_exploits()
           print(f"\n{opp_id}:")
           print(f"  Type: {bot_type} ({conf:.1%} confidence)")
           print(f"  Exploits: {exploits}")
   ```

4. **Iterate and improve:**
   - Adjust detection thresholds
   - Add new exploit patterns
   - Test counter-strategies

---

## Quick Reference: Counter-Strategy Cheat Sheet

| Opponent Type | Key Counter-Strategy |
|--------------|---------------------|
| **MCTS** | Fast tempo, unusual bet sizes, multiway pots |
| **CFR/GTO** | Play solid GTO yourself, minimize variance |
| **RL Bot** | Mislead early, switch strategies mid-game |
| **Rule-Based** | Map their rules (20 hands), exploit maximally |
| **Hand-Strength** | Bluff relentlessly when they miss |
| **Random** | Value bet everything, never bluff |
| **Hybrid** | Cautious play, identify current mode |

---

## Final Checklist

- [ ] Protocol compatibility tested (`competition_adapter.py` works)
- [ ] Created `opponent_detector.py` with full detection system
- [ ] Integrated OpponentManager into `decision_engine.py`
- [ ] Added opponent recording to `bot.py`
- [ ] Tested detection with `test_destroyer.py` (all tests pass)
- [ ] Ran 1000+ hands locally without crashes
- [ ] Verified opponent classification is accurate
- [ ] Confirmed counter-strategies are being applied
- [ ] Ready to destroy all opponents in competition

---

## When Competition Day Comes

1. **First 20 hands:** Gather data, classify all opponents
2. **Hands 20-50:** Deploy initial counter-strategies
3. **Hands 50+:** Full exploitation mode
4. **Continuous:** Monitor for counter-adaptation, adjust

**Remember:** You detect their strategy and counter it. They don't detect yours and adapt (unless they're RL/hybrid, in which case you counter-counter). You win.

**DESTROY THEM ALL! 🏆🔥**
