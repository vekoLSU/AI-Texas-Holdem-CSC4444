from typing import Dict, List
from poker_bot.agents.strategy_interface import StrategyInterface
import random


class DefenderAgent(StrategyInterface):
    """
    Agent C: Defender - "The Fortress"

    Purpose: Minimize variance vs aggressive opponents.

    Strategy:
    - Trap with strong hands (slow play)
    - Call down lighter vs aggression
    - Minimize bluffing (reduce risk)
    - Pot control with medium hands
    - Check-raise vs c-bets
    - Wait for premium hands to strike
    - Let aggressive opponents bluff off chips

    When to use:
    - vs Aggressive opponents (LAG, Maniac)
    - vs Opponents who bluff too much
    - When protecting chip lead
    - In crucial spots (bubble, final table)

    Expected performance:
    - vs Aggressive: +25% win rate
    - vs Maniac: +35% win rate
    - vs Weak opponents: +10% win rate (too passive)
    - Low variance, consistent
    """

    def __init__(self):
        super().__init__()
        self.name = "Defender_Agent"
        self.description = "Defensive agent - minimize variance vs aggression"

        # Defensive parameters
        self.bluff_frequency = 0.08  # 8% bluff rate (very low)
        self.trap_frequency = 0.40  # 40% slow play frequency
        self.call_down_threshold = 0.45  # Call with weaker hands vs aggression

    def decide(self,
               hand_cards: List[str],
               community_cards: List[str],
               hand_strength: Dict,
               phase: str,
               pot: int,
               to_call: int,
               our_chips: int,
               position: str,
               num_players: int,
               opponent_profiles: List[Dict],
               current_bet: int) -> Dict:
        """Defensive decision logic."""

        strength = hand_strength.get("strength", 0.0)
        hand_type = hand_strength.get("hand_type", "unknown")
        draw_potential = hand_strength.get("draw_potential", 0.0)

        # Analyze opponent aggression
        avg_aggression, avg_bluff_rate = self._analyze_opponent_aggression(opponent_profiles)

        # Adjust call-down threshold based on opponent aggression
        adjusted_call_threshold = self._calculate_call_threshold(avg_aggression, avg_bluff_rate)

        # Phase-specific strategy
        if phase == "PREFLOP":
            return self._decide_preflop(
                strength, pot, to_call, our_chips, position, num_players,
                avg_aggression
            )
        else:
            return self._decide_postflop(
                strength, hand_type, draw_potential, pot, to_call,
                our_chips, position, num_players, phase,
                avg_aggression, avg_bluff_rate, adjusted_call_threshold
            )

    def _decide_preflop(self, strength: float, pot: int, to_call: int,
                        our_chips: int, position: str, num_players: int,
                        avg_aggression: float) -> Dict:
        """Defensive preflop strategy."""

        # Premium hands (AA-QQ, AK)
        if strength >= 0.90:
            if to_call == 0:
                # Trap occasionally with premium hands
                if self._should_trap(avg_aggression):
                    # Limp/check to induce bluffs
                    return {"action": "check"}
                else:
                    # Standard raise
                    raise_amount = pot * 0.7
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            else:
                # Always call with premium hands (trap aggressive players)
                if avg_aggression > 2.0 and to_call < our_chips * 0.5:
                    # Just call to keep them betting
                    return {"action": "call"}
                else:
                    # Re-raise if they're not super aggressive
                    raise_amount = to_call * 2.5 + pot * 0.3
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}

        # Strong hands (JJ-TT, AQ, AJs)
        elif strength >= 0.75:
            if to_call == 0:
                # Standard raise
                raise_amount = pot * 0.6
                return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            elif to_call < our_chips * 0.20:
                # Call to see flop
                return {"action": "call"}
            else:
                # Fold to large bets without premium hands
                return {"action": "fold"}

        # Medium hands (99-77, suited connectors, broadway)
        elif strength >= 0.55:
            if to_call == 0:
                # Conservative raising (position-dependent)
                if position in ["late", "middle"]:
                    raise_amount = pot * 0.5
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
                else:
                    return {"action": "check"}
            elif to_call < pot * 0.3:
                # Call small bets
                return {"action": "call"}
            else:
                # Fold to aggression
                return {"action": "fold"}

        # Speculative hands (small pairs, suited cards)
        elif strength >= 0.40:
            if to_call == 0:
                # Check and see flop
                return {"action": "check"}
            elif to_call < pot * 0.2 and num_players >= 3:
                # Call with good implied odds
                return {"action": "call"}
            else:
                return {"action": "fold"}

        # Weak hands
        else:
            if to_call == 0:
                # Rarely bluff
                if position == "late" and num_players == 2 and random.random() < 0.1:
                    raise_amount = pot * 0.5
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
                else:
                    return {"action": "check"}
            else:
                return {"action": "fold"}

    def _decide_postflop(self, strength: float, hand_type: str,
                         draw_potential: float, pot: int, to_call: int,
                         our_chips: int, position: str, num_players: int,
                         phase: str, avg_aggression: float, avg_bluff_rate: float,
                         adjusted_call_threshold: float) -> Dict:
        """Defensive postflop strategy."""

        effective_strength = strength + draw_potential * 0.5

        # Very strong hands (two pair or better)
        if strength >= 0.75:
            if to_call == 0:
                # Trap with very strong hands vs aggressive opponents
                if self._should_trap(avg_aggression) and avg_aggression > 2.0:
                    # Check to induce bluffs
                    return {"action": "check"}
                else:
                    # Bet for value
                    bet_size = pot * 0.6
                    return {"action": "raise", "amount": min(int(bet_size), our_chips)}
            else:
                # Always call or raise with strong hands
                if avg_aggression > 2.5:
                    # Just call to keep them bluffing
                    return {"action": "call"}
                else:
                    # Raise for value vs moderate aggression
                    raise_amount = to_call * 2 + pot * 0.4
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}

        # Strong hands (top pair good kicker, overpair)
        elif strength >= 0.55:
            if to_call == 0:
                # Defensive value betting
                bet_size = pot * 0.5
                return {"action": "raise", "amount": min(int(bet_size), our_chips)}
            else:
                # Call down vs aggression
                if strength >= adjusted_call_threshold:
                    return {"action": "call"}
                elif to_call > pot * 1.2:
                    # Huge bet - fold unless we're strong enough
                    if strength >= 0.65:
                        return {"action": "call"}
                    else:
                        return {"action": "fold"}
                else:
                    return {"action": "call"}

        # Medium hands (middle pair, weak top pair)
        elif strength >= 0.40:
            if to_call == 0:
                # Pot control - check or small bet
                if num_players <= 2:
                    bet_size = pot * 0.35
                    return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                else:
                    return {"action": "check"}
            else:
                # Call down lighter vs aggression
                if avg_aggression > 2.0 and to_call < pot * 0.7:
                    # They might be bluffing
                    return {"action": "call"}
                elif to_call < pot * 0.4:
                    # Small bet - call
                    return {"action": "call"}
                else:
                    return {"action": "fold"}

        # Draws and weak made hands
        elif effective_strength >= 0.35:
            if draw_potential >= 0.30:  # Strong draw
                if to_call == 0:
                    # Check-raise with draws vs aggressive opponents
                    if avg_aggression > 2.5 and random.random() < 0.3:
                        # They'll likely bet, then we raise
                        return {"action": "check"}
                    else:
                        # Small bet with draws
                        bet_size = pot * 0.4
                        return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                else:
                    # Call with draws if price is right
                    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
                    if pot_odds < draw_potential * 0.85:
                        return {"action": "call"}
                    else:
                        return {"action": "fold"}
            else:
                # Weak hand, no draw
                if to_call == 0:
                    return {"action": "check"}
                else:
                    # Occasionally call down with bluff-catchers vs maniacs
                    if avg_bluff_rate > 0.4 and to_call < pot * 0.5 and phase == "RIVER":
                        return {"action": "call"}
                    else:
                        return {"action": "fold"}

        # Air (very weak hands)
        else:
            if to_call == 0:
                # Rarely bluff (only in good spots)
                if self._should_bluff(avg_aggression, position, num_players):
                    bet_size = pot * 0.5
                    return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                else:
                    return {"action": "check"}
            else:
                # Occasionally bluff-catch on river vs super aggressive
                if (phase == "RIVER" and avg_bluff_rate > 0.5 and
                    to_call < pot * 0.4 and random.random() < 0.15):
                    return {"action": "call"}
                else:
                    return {"action": "fold"}

    def _analyze_opponent_aggression(self, opponent_profiles: List[Dict]) -> tuple:
        """Analyze opponent aggression levels."""
        if not opponent_profiles:
            return 1.0, 0.15  # Default assumptions

        total_aggression = 0
        count = len(opponent_profiles)

        for profile in opponent_profiles:
            total_aggression += profile.get("aggression_factor", 1.0)

        avg_aggression = total_aggression / count

        # Estimate bluff rate based on aggression
        # Highly aggressive players likely bluff more
        estimated_bluff_rate = min((avg_aggression - 0.5) * 0.15, 0.5)

        return avg_aggression, max(estimated_bluff_rate, 0.10)

    def _calculate_call_threshold(self, avg_aggression: float,
                                   avg_bluff_rate: float) -> float:
        """
        Calculate how weak a hand we can call with.

        More aggressive opponents = call with weaker hands.
        """
        # Base threshold
        threshold = self.call_down_threshold

        # Decrease threshold (call lighter) vs aggression
        aggression_adjustment = (avg_aggression - 1.5) * 0.08
        bluff_adjustment = avg_bluff_rate * 0.15

        adjusted = threshold - aggression_adjustment - bluff_adjustment

        # Clamp between 0.30 and 0.60
        return max(0.30, min(adjusted, 0.60))

    def _should_trap(self, avg_aggression: float) -> bool:
        """Should we slow play/trap?"""
        # Trap more often vs aggressive opponents
        trap_chance = self.trap_frequency * (avg_aggression / 1.5)
        return random.random() < min(trap_chance, 0.6)

    def _should_bluff(self, avg_aggression: float, position: str,
                      num_players: int) -> bool:
        """Should we bluff? (rarely)"""
        bluff_chance = self.bluff_frequency

        # Only bluff in good spots
        if position != "late":
            bluff_chance *= 0.5

        if num_players > 2:
            bluff_chance *= 0.3

        # Less bluffing vs aggressive (they'll call/raise)
        if avg_aggression > 2.0:
            bluff_chance *= 0.5

        return random.random() < bluff_chance
