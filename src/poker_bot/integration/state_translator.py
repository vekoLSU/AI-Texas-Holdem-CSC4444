"""
State Translator: Converts competition JSON state to 50-dim numpy array for trained RL agent.

The agent was trained with a specific 50-dimensional state representation:
- 2 dims: Hole cards (rank only, no suits)
- 5 dims: Community cards (rank only, zero-padded)
- Game state features (pot, bets, chips, etc.)
- Padding to reach exactly 50 dimensions

Competition provides JSON with full poker state including suits.
We strip suits since the agent was trained without them.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class StateTranslator:
    """Translates competition JSON state to 50-dim numpy array for RL agent."""

    # Card rank mapping (suits stripped)
    RANK_MAP = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
        'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }

    # Phase encoding
    PHASE_MAP = {
        'WAITING': 0.0,
        'PREFLOP': 0.2,
        'FLOP': 0.4,
        'TURN': 0.6,
        'RIVER': 0.8,
        'SHOWDOWN': 1.0
    }

    def __init__(self, starting_chips: int = 1000):
        """
        Initialize state translator.

        Args:
            starting_chips: Default stack size for normalization (default: 1000)
        """
        self.starting_chips = starting_chips

    def translate(self, json_state: Dict, our_player_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert JSON state to 50-dim actor state and 70-dim critic state.

        Args:
            json_state: Competition JSON state message
            our_player_id: Our player ID to identify our cards

        Returns:
            (actor_state, critic_state): Tuple of numpy arrays (50,) and (70,)
        """
        if json_state.get("type") != "state" or "state" not in json_state:
            # Return zero states for non-state messages
            return np.zeros(50, dtype=np.float32), np.zeros(70, dtype=np.float32)

        state = json_state["state"]
        table = state.get("table", {})
        players = table.get("players", [])

        # Extract game info
        pot = state.get("pot", 0)
        phase = state.get("phase", "WAITING")
        board = state.get("board", [])
        to_act_idx = state.get("toActIdx", -1)

        # Find our player
        our_player, our_index = self._find_our_player(players, our_player_id)
        if not our_player:
            # Can't build state without our player info
            return np.zeros(50, dtype=np.float32), np.zeros(70, dtype=np.float32)

        # Extract our cards
        our_hole_cards = our_player.get("cards", [])

        # Build actor state (50 dims - partial information)
        actor_state = self._build_actor_state(
            our_hole_cards=our_hole_cards,
            community_cards=board,
            pot=pot,
            phase=phase,
            our_player=our_player,
            players=players,
            our_index=our_index
        )

        # Build critic state (70 dims - full information, includes opponent cards)
        # For competition, we don't have opponent cards, so we'll pad with zeros
        critic_state = self._build_critic_state(
            our_hole_cards=our_hole_cards,
            community_cards=board,
            pot=pot,
            phase=phase,
            players=players
        )

        return actor_state, critic_state

    def _find_our_player(self, players: List[Dict], our_player_id: str) -> Tuple[Optional[Dict], int]:
        """Find our player in the players list."""
        for i, player in enumerate(players):
            if player and player.get("id") == our_player_id:
                return player, i
        return None, -1

    def _card_to_rank(self, card: Dict) -> int:
        """
        Convert card object to rank integer (0-12).

        Args:
            card: {"rank": "A", "suit": "HEART"}

        Returns:
            rank: Integer 0-12 (2=0, 3=1, ..., A=12)
        """
        if not card or "rank" not in card:
            return 0

        rank_str = card["rank"]
        return self.RANK_MAP.get(rank_str, 0)

    def _build_actor_state(
        self,
        our_hole_cards: List[Dict],
        community_cards: List[Dict],
        pot: int,
        phase: str,
        our_player: Dict,
        players: List[Dict],
        our_index: int
    ) -> np.ndarray:
        """
        Build 50-dimensional actor state (partial information).

        State layout:
        - [0:2]: Hole cards (rank normalized by 13)
        - [2:7]: Community cards (5 cards, zero-padded)
        - [7]: Pot (normalized by starting chips)
        - [8]: Current bet to us (normalized)
        - [9]: Our chips (normalized)
        - [10]: Our current bet (normalized)
        - [11]: Number of active players (normalized by 7)
        - [12]: Our position (0-6, normalized by 7)
        - [13]: Phase encoding (0.0-1.0)
        - [14]: Stack-to-pot ratio
        - [15:50]: Zero padding
        """
        state = []

        # Hole cards (2 dims)
        hole_ranks = [self._card_to_rank(c) / 13.0 for c in our_hole_cards[:2]]
        while len(hole_ranks) < 2:
            hole_ranks.append(0.0)
        state.extend(hole_ranks[:2])

        # Community cards (5 dims, zero-padded)
        community_ranks = [self._card_to_rank(c) / 13.0 for c in community_cards]
        while len(community_ranks) < 5:
            community_ranks.append(0.0)
        state.extend(community_ranks[:5])

        # Pot (normalized)
        state.append(pot / self.starting_chips)

        # Current bet to us (estimate from game state)
        # Competition server doesn't directly expose "current bet to us"
        # We estimate it from the highestThisStreet in the game state
        current_bet = self._estimate_current_bet(players)
        our_bet = self._estimate_player_bet(our_player)
        to_call = max(0, current_bet - our_bet)
        state.append(to_call / self.starting_chips)

        # Our chips (normalized)
        our_chips = our_player.get("chips", 0)
        state.append(our_chips / self.starting_chips)

        # Our current bet (normalized)
        state.append(our_bet / self.starting_chips)

        # Active players count (normalized by max 7 players)
        active_players = sum(1 for p in players if p and not self._is_player_out(p))
        state.append(active_players / 7.0)

        # Our position (normalized)
        state.append(our_index / 7.0)

        # Phase encoding
        phase_encoding = self.PHASE_MAP.get(phase, 0.0)
        state.append(phase_encoding)

        # Stack-to-pot ratio (SPR)
        spr = our_chips / max(pot, 1)
        spr_normalized = min(spr / 10.0, 1.0)  # Cap at 10x pot
        state.append(spr_normalized)

        # Pad to 50 dimensions
        while len(state) < 50:
            state.append(0.0)

        return np.array(state[:50], dtype=np.float32)

    def _build_critic_state(
        self,
        our_hole_cards: List[Dict],
        community_cards: List[Dict],
        pot: int,
        phase: str,
        players: List[Dict]
    ) -> np.ndarray:
        """
        Build 70-dimensional critic state (full information).

        NOTE: In competition, we don't have opponent cards, so we pad with zeros.
        The critic is only used for value estimation, not action selection.

        State layout:
        - [0:2]: Our hole cards
        - [2:4]: Opponent hole cards (zeros in competition)
        - [4:9]: Community cards (5 cards)
        - [9]: Pot
        - [10]: Current bet
        - [11:14]: Our player state (chips, bet, not_folded)
        - [14:17]: Opponent player state (chips, bet, not_folded)
        - [17:70]: Zero padding
        """
        state = []

        # Our hole cards (2 dims)
        our_ranks = [self._card_to_rank(c) / 13.0 for c in our_hole_cards[:2]]
        while len(our_ranks) < 2:
            our_ranks.append(0.0)
        state.extend(our_ranks[:2])

        # Opponent hole cards (2 dims) - unknown in competition
        state.extend([0.0, 0.0])

        # Community cards (5 dims)
        community_ranks = [self._card_to_rank(c) / 13.0 for c in community_cards]
        while len(community_ranks) < 5:
            community_ranks.append(0.0)
        state.extend(community_ranks[:5])

        # Pot
        state.append(pot / self.starting_chips)

        # Current bet
        current_bet = self._estimate_current_bet(players)
        state.append(current_bet / self.starting_chips)

        # Our player state (3 dims)
        if players:
            our_player = players[0]  # Assume first is us (simplified)
            state.append(our_player.get("chips", 0) / self.starting_chips)
            state.append(self._estimate_player_bet(our_player) / self.starting_chips)
            state.append(1.0 if not self._is_player_out(our_player) else 0.0)
        else:
            state.extend([0.0, 0.0, 1.0])

        # Opponent player state (3 dims) - aggregate for multi-player
        if len(players) > 1:
            # Average opponent stats
            opp_chips = np.mean([p.get("chips", 0) for p in players[1:] if p])
            opp_bet = np.mean([self._estimate_player_bet(p) for p in players[1:] if p])
            opp_active = np.mean([1.0 if not self._is_player_out(p) else 0.0 for p in players[1:] if p])
            state.extend([
                opp_chips / self.starting_chips,
                opp_bet / self.starting_chips,
                opp_active
            ])
        else:
            state.extend([0.0, 0.0, 0.0])

        # Pad to 70 dimensions
        while len(state) < 70:
            state.append(0.0)

        return np.array(state[:70], dtype=np.float32)

    def _estimate_current_bet(self, players: List[Dict]) -> int:
        """
        Estimate the current bet amount from player states.

        This is a simplified heuristic since the competition server doesn't
        directly expose the current betting round's bet amount.
        """
        if not players:
            return 0

        # Look for max chips committed (rough estimate)
        # In reality, we'd need to track betting history
        max_bet = 0
        for p in players:
            if p:
                # The competition server uses "action" field but not current bet per round
                # We'll use a heuristic based on chip changes (if we tracked previous state)
                # For now, return 0 as a conservative estimate
                pass

        return max_bet

    def _estimate_player_bet(self, player: Dict) -> int:
        """Estimate player's current bet (simplified)."""
        # Competition server doesn't expose per-round bets directly
        # Return 0 as conservative estimate
        return 0

    def _is_player_out(self, player: Dict) -> bool:
        """Check if player is folded or all-in."""
        if not player:
            return True

        # Check if player has chips
        chips = player.get("chips", 0)
        if chips <= 0:
            return True

        # Check action status (if available)
        action = player.get("action", "")
        if action == "FOLD":
            return True

        return False

    def get_state_info(self, json_state: Dict, our_player_id: str) -> Dict:
        """
        Extract key game state information for decision-making.

        Returns:
            Dictionary with: pot, to_call, our_chips, phase, community_cards, etc.
        """
        if json_state.get("type") != "state" or "state" not in json_state:
            return {}

        state = json_state["state"]
        table = state.get("table", {})
        players = table.get("players", [])

        our_player, our_index = self._find_our_player(players, our_player_id)
        if not our_player:
            return {}

        pot = state.get("pot", 0)
        phase = state.get("phase", "WAITING")
        board = state.get("board", [])
        current_bet = self._estimate_current_bet(players)
        our_bet = self._estimate_player_bet(our_player)
        to_call = max(0, current_bet - our_bet)
        our_chips = our_player.get("chips", 0)

        return {
            "pot": pot,
            "to_call": to_call,
            "our_chips": our_chips,
            "phase": phase,
            "community_cards": board,
            "num_players": len([p for p in players if p and not self._is_player_out(p)]),
            "our_position": our_index,
            "current_bet": current_bet
        }


if __name__ == "__main__":
    # Test state translator
    print("Testing StateTranslator...")

    translator = StateTranslator(starting_chips=1000)

    # Example competition state
    test_state = {
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
                    },
                    {
                        "id": "player2",
                        "chips": 900,
                        "action": "RAISE",
                        "cards": []  # We don't see opponent cards
                    }
                ],
                "phase": "FLOP",
                "cardOpen": []
            },
            "pot": 150,
            "phase": "FLOP",
            "board": [
                {"rank": "Q", "suit": "DIAMOND"},
                {"rank": "J", "suit": "CLUB"},
                {"rank": "T", "suit": "HEART"}
            ],
            "toActIdx": 0,
            "hand": 5
        }
    }

    actor_state, critic_state = translator.translate(test_state, "player1")

    print(f"\nActor state shape: {actor_state.shape}")
    print(f"Actor state (first 15 dims): {actor_state[:15]}")
    print(f"\nCritic state shape: {critic_state.shape}")
    print(f"Critic state (first 15 dims): {critic_state[:15]}")

    state_info = translator.get_state_info(test_state, "player1")
    print(f"\nState info: {state_info}")

    print("\n✅ StateTranslator test complete!")
