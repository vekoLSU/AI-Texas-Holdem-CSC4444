from abc import ABC, abstractmethod
from typing import Dict, List

class StrategyInterface(ABC):
    """
    Base interface for all poker strategy agents.

    All specialist agents (GTO, Exploiter, Defender) must implement this interface.
    This allows the MetaController to seamlessly switch between agents.
    """

    def __init__(self):
        self.name = "BaseStrategy"
        self.description = "Base strategy interface"

    @abstractmethod
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
        """
        Make a decision given the current game state.

        Args:
            hand_cards: Our hole cards (e.g., ["As", "Kh"])
            community_cards: Board cards (e.g., ["Qd", "Jc", "Th"])
            hand_strength: Dict with 'strength' (0.0-1.0) and 'hand_type'
            phase: Current phase (PREFLOP, FLOP, TURN, RIVER)
            pot: Current pot size
            to_call: Amount needed to call
            our_chips: Our chip stack
            position: Position at table (early, middle, late, heads-up)
            num_players: Number of active players
            opponent_profiles: List of opponent profile dicts
            current_bet: Current bet amount

        Returns:
            Dict with 'action' (fold/call/check/raise) and optional 'amount'
        """
        pass

    def get_name(self) -> str:
        """Return the strategy name."""
        return self.name

    def get_description(self) -> str:
        """Return the strategy description."""
        return self.description
