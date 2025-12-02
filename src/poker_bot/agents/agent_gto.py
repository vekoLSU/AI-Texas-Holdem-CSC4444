from typing import Dict, List
from poker_bot.agents.strategy_interface import StrategyInterface
import math


class GTOAgent(StrategyInterface):
    """
    Agent A: GTO Baseline - "The Professor"

    Purpose: Solid, unexploitable play based on game theory optimal (GTO) principles.

    Strategy:
    - Balanced ranges (mix of strong hands and bluffs)
    - Mixed strategies (randomized actions to prevent exploitation)
    - Game-theoretic bet sizing (pot odds based)
    - Position-aware play
    - Equity-driven decisions

    When to use:
    - vs Strong GTO opponents (minimize loss)
    - vs Unknown opponents (safe baseline)
    - When short-stacked (reduce variance)
    - When protecting chip lead

    Expected performance:
    - vs Random: +10% win rate
    - vs MCTS: +15% win rate
    - vs GTO: ~50% win rate (breakeven)
    - Safe but not maximum profit
    """

    def __init__(self):
        super().__init__()
        self.name = "GTO_Agent"
        self.description = "Game Theory Optimal baseline - unexploitable play"

        # GTO parameters (can be tuned)
        self.bluff_frequency = 0.20  # 20% bluff rate (balanced)
        self.value_bet_frequency = 0.70  # 70% value bet with strong hands
        self.cbet_frequency = 0.65  # 65% continuation bet frequency

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
        """GTO decision logic."""

        strength = hand_strength.get("strength", 0.0)
        hand_type = hand_strength.get("hand_type", "unknown")
        draw_potential = hand_strength.get("draw_potential", 0.0)

        # Calculate pot odds
        pot_odds = self._calculate_pot_odds(to_call, pot)

        # Calculate minimum equity needed to call
        min_equity = self._calculate_min_equity(to_call, pot)

        # Position multiplier
        position_multiplier = self._get_position_multiplier(position)

        # Adjust strength based on position
        adjusted_strength = strength * position_multiplier

        # Phase-specific strategy
        if phase == "PREFLOP":
            return self._decide_preflop(
                adjusted_strength, pot, to_call, our_chips,
                position, num_players, pot_odds
            )
        else:
            return self._decide_postflop(
                adjusted_strength, hand_type, draw_potential, pot,
                to_call, our_chips, position, num_players, pot_odds,
                phase, min_equity
            )

    def _decide_preflop(self, strength: float, pot: int, to_call: int,
                        our_chips: int, position: str, num_players: int,
                        pot_odds: float) -> Dict:
        """GTO preflop strategy."""

        # Premium hands (AA, KK, QQ, AK)
        if strength >= 0.90:
            if to_call == 0:
                # Standard raise: 3x BB (assuming BB ~= pot/4)
                raise_amount = max(pot * 0.75, 30)
                return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            elif to_call < our_chips * 0.25:
                # 3-bet with premium hands
                raise_amount = to_call * 3 + pot * 0.3
                return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            else:
                # Large bet - call with best hands
                return {"action": "call"}

        # Strong hands (JJ-TT, AQ, AJs)
        elif strength >= 0.75:
            if to_call == 0:
                # Open raise from any position
                raise_amount = pot * 0.6 + 20
                return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            elif to_call < our_chips * 0.15:
                # Call moderate bets
                return {"action": "call"}
            else:
                # Fold to large bets with marginal strong hands
                return {"action": "fold"}

        # Medium hands (99-77, suited connectors, broadway)
        elif strength >= 0.55:
            if to_call == 0:
                # Position-dependent open raising
                if position in ["late", "middle"]:
                    raise_amount = pot * 0.5 + 15
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
                else:
                    # Check from early position
                    return {"action": "check"}
            elif pot_odds < 0.25:  # Good odds
                return {"action": "call"}
            else:
                return {"action": "fold"}

        # Speculative hands (small pairs, suited cards)
        elif strength >= 0.40:
            if to_call == 0:
                # Steal attempt from late position
                if position == "late" and num_players <= 3:
                    raise_amount = pot * 0.5
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
                else:
                    return {"action": "check"}
            elif pot_odds < 0.15 and num_players >= 3:
                # Call with implied odds in multiway pots
                return {"action": "call"}
            else:
                return {"action": "fold"}

        # Weak hands
        else:
            if to_call == 0:
                # Occasional steal from late position
                if position == "late" and num_players == 2 and self._should_bluff():
                    raise_amount = pot * 0.6
                    return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
                else:
                    return {"action": "check"}
            else:
                # Fold weak hands
                return {"action": "fold"}

    def _decide_postflop(self, strength: float, hand_type: str,
                         draw_potential: float, pot: int, to_call: int,
                         our_chips: int, position: str, num_players: int,
                         pot_odds: float, phase: str, min_equity: float) -> Dict:
        """GTO postflop strategy."""

        # Effective strength (made hand + draw potential)
        effective_strength = strength + draw_potential * 0.6

        # Very strong hands (two pair or better)
        if strength >= 0.75:
            if to_call == 0:
                # Value bet for protection and value
                bet_size = self._calculate_gto_bet_size(pot, strength, num_players)
                return {"action": "raise", "amount": min(int(bet_size), our_chips)}
            elif pot_odds < 0.4:  # Getting good price
                # Raise for value
                raise_amount = to_call * 2 + pot * 0.4
                return {"action": "raise", "amount": min(int(raise_amount), our_chips)}
            else:
                # Call large bets with strong hands
                return {"action": "call"}

        # Strong hands (top pair good kicker, overpair)
        elif strength >= 0.55:
            if to_call == 0:
                # Standard value bet
                bet_size = pot * 0.65
                return {"action": "raise", "amount": min(int(bet_size), our_chips)}
            elif strength >= min_equity + 0.1:  # We have equity edge
                return {"action": "call"}
            elif to_call > pot:  # Large bet
                return {"action": "fold"}
            else:
                return {"action": "call"}

        # Medium hands (middle pair, weak top pair)
        elif strength >= 0.40:
            if to_call == 0:
                # Small bet or check
                if num_players <= 2:
                    bet_size = pot * 0.4
                    return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                else:
                    return {"action": "check"}
            elif pot_odds < 0.25:  # Good price
                return {"action": "call"}
            else:
                return {"action": "fold"}

        # Draws and weak made hands
        elif effective_strength >= 0.35:
            if draw_potential >= 0.25:  # Strong draw
                if to_call == 0:
                    # Semi-bluff with draws
                    if num_players <= 3 and self._should_bluff():
                        bet_size = pot * 0.5
                        return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                    else:
                        return {"action": "check"}
                elif pot_odds < draw_potential * 0.9:  # Good draw odds
                    return {"action": "call"}
                else:
                    return {"action": "fold"}
            else:
                # Weak hand, no draw
                if to_call == 0:
                    return {"action": "check"}
                else:
                    return {"action": "fold"}

        # Air (bluff candidates)
        else:
            if to_call == 0:
                # Balanced bluffing
                if self._should_bluff() and position in ["late", "middle"]:
                    bet_size = pot * 0.6
                    return {"action": "raise", "amount": min(int(bet_size), our_chips)}
                else:
                    return {"action": "check"}
            else:
                # Fold with nothing
                return {"action": "fold"}

    def _calculate_pot_odds(self, to_call: int, pot: int) -> float:
        """Calculate pot odds."""
        if to_call == 0:
            return 0.0
        return to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0

    def _calculate_min_equity(self, to_call: int, pot: int) -> float:
        """Calculate minimum equity needed to call."""
        if to_call == 0 or pot + to_call == 0:
            return 0.0
        return to_call / (pot + to_call)

    def _get_position_multiplier(self, position: str) -> float:
        """Get position-based strength multiplier."""
        multipliers = {
            "late": 1.15,
            "middle": 1.0,
            "early": 0.90,
            "heads-up": 1.10,
            "unknown": 1.0
        }
        return multipliers.get(position, 1.0)

    def _calculate_gto_bet_size(self, pot: int, strength: float,
                                 num_players: int) -> float:
        """
        Calculate GTO bet size.

        GTO typically uses geometric bet sizing:
        - 50-75% pot for value bets
        - Same sizing for bluffs (to remain balanced)
        """
        # Base bet size: 60% pot
        base_bet = pot * 0.6

        # Adjust for strength (slightly larger with nuts)
        strength_multiplier = 1.0 + (strength - 0.75) * 0.3

        # Adjust for player count (smaller bets multiway)
        player_multiplier = 1.0 if num_players <= 2 else 0.85

        return base_bet * strength_multiplier * player_multiplier

    def _should_bluff(self) -> bool:
        """
        Determine if we should bluff based on GTO frequency.

        GTO bluffing should be randomized to prevent exploitation.
        """
        import random
        return random.random() < self.bluff_frequency
