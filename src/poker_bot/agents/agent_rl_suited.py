from typing import Dict, List
import torch
import numpy as np
from poker_bot.agents.strategy_interface import StrategyInterface
from poker_bot.training.networks import ActorCriticAgent


class RLSuitedAgent(StrategyInterface):
    """
    Agent D: RL Suited Agent - "The Neural Network"

    Purpose: Deep RL trained on 500K hands with suit information.

    Strategy:
    - Uses ActorCriticAgent neural network (trained with PPO)
    - State dimensions: actor=70, critic=90
    - Includes suit encoding (rank/13, suit/4)
    - Trained on real poker with flushes, straight flushes

    When to use:
    - vs Weak/Passive opponents (exploitative play)
    - When we have enough observations
    - Multi-player scenarios (trained on 6-player)

    Model:
    - Path: trained_models_suited/agent_suited_final.pt
    - Training: 500K hands with suit information
    - Actor state: 70 dims (includes suits)
    - Critic state: 90 dims (full information)
    """

    def __init__(self, model_path: str = "trained_models_suited/agent_suited_final.pt",
                 device: str = "cpu"):
        super().__init__()
        self.name = "RL_Suited_Agent"
        self.description = "Neural network trained with suit information"

        self.device = torch.device(device)

        # Load trained model
        self.agent = ActorCriticAgent(actor_state_dim=70, critic_state_dim=90)
        try:
            self.agent.load(model_path)
            self.agent.to(self.device)
            self.agent.eval()
            self.loaded = True
        except Exception as e:
            print(f"Warning: Could not load RL model from {model_path}: {e}")
            print("RL agent will fall back to random actions")
            self.loaded = False

        # Card encoding
        self.rank_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }
        self.suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}

        # Action mapping
        self.action_map = {0: "fold", 1: "call", 2: "check", 3: "raise"}

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
        """RL agent decision using trained neural network."""

        if not self.loaded:
            # Fallback to safe action if model didn't load
            if to_call == 0:
                return {"action": "check"}
            elif to_call < pot * 0.3:
                return {"action": "call"}
            else:
                return {"action": "fold"}

        # Build state representation
        actor_state = self._build_actor_state(
            hand_cards, community_cards, phase, pot, to_call,
            our_chips, position, num_players, opponent_profiles, current_bet
        )

        critic_state = self._build_critic_state(
            hand_cards, community_cards, phase, pot, to_call,
            our_chips, num_players, opponent_profiles, current_bet
        )

        # Convert to tensors
        actor_tensor = torch.FloatTensor(actor_state).unsqueeze(0).to(self.device)
        critic_tensor = torch.FloatTensor(critic_state).unsqueeze(0).to(self.device)

        # Get action from network
        with torch.no_grad():
            action_idx, amount_ratio, value = self.agent.get_action_and_value(
                actor_tensor, critic_tensor, deterministic=True
            )

        # Map action
        action_name = self.action_map.get(action_idx, "fold")

        # Calculate raise amount
        if action_name == "raise":
            raise_amount = int(amount_ratio * pot * 0.5)
            raise_amount = max(raise_amount, current_bet * 2, 20)
            raise_amount = min(raise_amount, our_chips)
            return {"action": action_name, "amount": raise_amount}
        else:
            return {"action": action_name}

    def _build_actor_state(self, hand_cards: List[str], community_cards: List[str],
                           phase: str, pot: int, to_call: int, our_chips: int,
                           position: str, num_players: int, opponent_profiles: List[Dict],
                           current_bet: int) -> np.ndarray:
        """
        Build 70-dimensional actor state WITH suit encoding.

        Layout:
        - [0:4]: Hole card 1 (rank/13, suit/4, rank^2/169, suit==X one-hot)
        - [4:8]: Hole card 2 (same encoding)
        - [8:28]: Community cards (5 cards * 4 dims each)
        - [28:40]: Game state features
        - [40:70]: Position, opponent, and table features
        """
        state = []

        # Encode hole cards (2 cards * 4 dims = 8 dims)
        for i in range(2):
            if i < len(hand_cards):
                card = hand_cards[i]
                rank, suit = self._parse_card(card)
                state.extend([
                    rank / 13.0,           # Rank normalized
                    suit / 4.0,            # Suit normalized
                    (rank * rank) / 169.0, # Rank squared (for pairs)
                    1.0 if suit == 0 else 0.0  # Suit indicator
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        # Encode community cards (5 cards * 4 dims = 20 dims)
        for i in range(5):
            if i < len(community_cards):
                card = community_cards[i]
                rank, suit = self._parse_card(card)
                state.extend([
                    rank / 13.0,
                    suit / 4.0,
                    (rank * rank) / 169.0,
                    1.0 if suit == 0 else 0.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        # Game state features (12 dims)
        state.extend([
            pot / 1000.0,                    # Pot size
            to_call / 1000.0,                # Amount to call
            our_chips / 1000.0,              # Our stack
            current_bet / 1000.0,            # Current bet
            self._encode_phase(phase),       # Phase encoding
            num_players / 7.0,               # Active players
            self._encode_position(position), # Position encoding
            (our_chips / max(pot, 1)) / 10.0, # SPR (stack-to-pot ratio)
            min((to_call / max(pot, 1)), 1.0), # Pot odds
            1.0 if to_call == 0 else 0.0,   # Can check
            1.0 if our_chips > current_bet * 3 else 0.0, # Can raise comfortably
            1.0 if our_chips < 10 * 10 else 0.0  # Short stack
        ])

        # Opponent features (aggregate if multiple opponents) (30 dims)
        if opponent_profiles:
            avg_chips = np.mean([p.get("chips", 1000) for p in opponent_profiles])
            avg_vpip = np.mean([p.get("vpip", 0.25) for p in opponent_profiles])
            avg_aggression = np.mean([p.get("aggression_factor", 1.0) for p in opponent_profiles])

            state.extend([
                avg_chips / 1000.0,
                avg_vpip,
                avg_aggression / 3.0,
                len([p for p in opponent_profiles if p.get("is_big_stack", False)]) / 7.0,
                len([p for p in opponent_profiles if p.get("is_short_stack", False)]) / 7.0,
            ])
        else:
            state.extend([1.0, 0.25, 0.33, 0.0, 0.0])

        # Additional features to reach 70 dims (25 dims)
        # Hand type indicators, draw potential, etc.
        state.extend([0.0] * 25)

        # Ensure exactly 70 dimensions
        state = state[:70]
        while len(state) < 70:
            state.append(0.0)

        return np.array(state, dtype=np.float32)

    def _build_critic_state(self, hand_cards: List[str], community_cards: List[str],
                            phase: str, pot: int, to_call: int, our_chips: int,
                            num_players: int, opponent_profiles: List[Dict],
                            current_bet: int) -> np.ndarray:
        """
        Build 90-dimensional critic state (full information).

        In competition we don't have opponent cards, so we pad with zeros.
        """
        state = []

        # Our hole cards (8 dims)
        for i in range(2):
            if i < len(hand_cards):
                card = hand_cards[i]
                rank, suit = self._parse_card(card)
                state.extend([rank / 13.0, suit / 4.0, (rank * rank) / 169.0, 1.0 if suit == 0 else 0.0])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        # Opponent hole cards (unknown in competition) (8 dims)
        state.extend([0.0] * 8)

        # Community cards (20 dims)
        for i in range(5):
            if i < len(community_cards):
                card = community_cards[i]
                rank, suit = self._parse_card(card)
                state.extend([rank / 13.0, suit / 4.0, (rank * rank) / 169.0, 1.0 if suit == 0 else 0.0])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        # Game state (10 dims)
        state.extend([
            pot / 1000.0,
            current_bet / 1000.0,
            our_chips / 1000.0,
            num_players / 7.0,
            self._encode_phase(phase),
            (pot / max(our_chips, 1)),
            1.0 if to_call == 0 else 0.0,
            to_call / 1000.0,
            (our_chips / max(pot, 1)) / 10.0,
            1.0 if our_chips < 100 else 0.0
        ])

        # Additional features to reach 90 dims (44 dims)
        state.extend([0.0] * 44)

        # Ensure exactly 90 dimensions
        state = state[:90]
        while len(state) < 90:
            state.append(0.0)

        return np.array(state, dtype=np.float32)

    def _parse_card(self, card: str) -> tuple:
        """
        Parse card string to (rank, suit) integers.

        Args:
            card: e.g., "As", "Kh", "2d"

        Returns:
            (rank, suit): Both as integers
        """
        if len(card) < 2:
            return 0, 0

        rank_char = card[0].upper()
        suit_char = card[1].lower()

        rank = self.rank_map.get(rank_char, 0)
        suit = self.suit_map.get(suit_char, 0)

        return rank, suit

    def _encode_phase(self, phase: str) -> float:
        """Encode phase as float."""
        phase_map = {
            "WAITING": 0.0,
            "PREFLOP": 0.2,
            "FLOP": 0.4,
            "TURN": 0.6,
            "RIVER": 0.8,
            "SHOWDOWN": 1.0
        }
        return phase_map.get(phase, 0.0)

    def _encode_position(self, position: str) -> float:
        """Encode position as float."""
        position_map = {
            "early": 0.2,
            "middle": 0.5,
            "late": 0.8,
            "heads-up": 1.0,
            "unknown": 0.5
        }
        return position_map.get(position, 0.5)
