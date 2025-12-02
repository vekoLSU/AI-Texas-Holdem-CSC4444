"""
ICM (Independent Chip Model) Adjuster for Elimination Tournaments

In elimination tournaments, chip value is not linear. Survival is more important
than accumulating chips. This module adjusts bot strategy for tournament play:

1. Bubble play: Ultra-conservative when close to elimination threshold
2. Big stack protection: Don't risk big stack on marginal spots
3. Short stack push/fold: Aggressive when desperate
4. Avoid dominated situations: Don't call off stack vs stronger ranges
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ICMAdjuster:
    """Adjusts bot actions for elimination tournament ICM considerations."""

    def __init__(self, total_players: int = 7, starting_chips: int = 1000):
        """
        Initialize ICM adjuster.

        Args:
            total_players: Total players in tournament (default: 7)
            starting_chips: Starting chip stack (default: 1000)
        """
        self.total_players = total_players
        self.starting_chips = starting_chips

        # ICM thresholds
        self.bubble_threshold = total_players  # When we're on the bubble
        self.short_stack_bb = 10  # Below 10BB is short stack
        self.critical_stack_bb = 5  # Below 5BB is critical

    def adjust_action(
        self,
        action: str,
        amount: int,
        game_state: Dict,
        hand_strength: float = 0.5
    ) -> tuple[str, int]:
        """
        Adjust action based on ICM considerations.

        Args:
            action: Original action ("FOLD", "CALL", "CHECK", "RAISE")
            amount: Original bet amount
            game_state: Current game state
            hand_strength: Hand strength estimate [0, 1]

        Returns:
            (adjusted_action, adjusted_amount): Modified action and amount
        """
        players_left = game_state.get("num_players", self.total_players)
        our_chips = game_state.get("our_chips", self.starting_chips)
        pot = game_state.get("pot", 0)
        to_call = game_state.get("to_call", 0)
        big_blind = game_state.get("big_blind", 10)

        # Calculate stack in big blinds
        stack_bb = our_chips / big_blind if big_blind > 0 else 999

        # Apply ICM adjustments in priority order

        # 1. BUBBLE PLAY - Ultra conservative
        if self._is_bubble(players_left):
            action, amount = self._adjust_for_bubble(
                action, amount, hand_strength, stack_bb, to_call, our_chips
            )

        # 2. SHORT STACK - Push/fold strategy
        elif stack_bb < self.short_stack_bb:
            action, amount = self._adjust_for_short_stack(
                action, amount, hand_strength, stack_bb, to_call, our_chips, pot
            )

        # 3. BIG STACK - Protect our stack
        elif self._is_big_stack(our_chips, game_state):
            action, amount = self._adjust_for_big_stack(
                action, amount, hand_strength, to_call, our_chips
            )

        # 4. GENERAL TOURNAMENT ADJUSTMENT - More conservative than cash game
        else:
            action, amount = self._general_tournament_adjustment(
                action, amount, hand_strength, to_call, our_chips, pot
            )

        return action, amount

    def _is_bubble(self, players_left: int) -> bool:
        """Check if we're on the bubble (one player away from elimination threshold)."""
        # Typically bubble is when we're close to final table or money
        # For 7-player tournament, bubble might be at 8->7 or specific threshold
        # Adjust based on tournament structure

        # Conservative: Consider bubble when 8 players left (1 needs to bust)
        return players_left == self.total_players + 1

    def _is_big_stack(self, our_chips: int, game_state: Dict) -> bool:
        """Check if we have a big stack relative to average."""
        # Estimate average stack
        total_chips = self.total_players * self.starting_chips
        players_left = game_state.get("num_players", self.total_players)
        avg_stack = total_chips / max(players_left, 1)

        # Big stack = 1.5x average or more
        return our_chips >= avg_stack * 1.5

    def _adjust_for_bubble(
        self,
        action: str,
        amount: int,
        hand_strength: float,
        stack_bb: float,
        to_call: int,
        our_chips: int
    ) -> tuple[str, int]:
        """
        Bubble play: Only play premium hands, avoid marginal spots.

        Goal: Let other players bust out first.
        """
        logger.info(f"[ICM] BUBBLE PLAY: players_left={self.total_players+1}")

        # On bubble, only play very strong hands
        if action == "RAISE":
            if hand_strength < 0.85:
                # Not strong enough to raise on bubble - fold or call depending on pot odds
                if to_call < our_chips * 0.1:  # Less than 10% of stack
                    action = "CALL"
                    amount = 0
                    logger.info(f"[ICM] Bubble: RAISE->CALL (hand_strength={hand_strength:.2f})")
                else:
                    action = "FOLD"
                    amount = 0
                    logger.info(f"[ICM] Bubble: RAISE->FOLD (hand_strength={hand_strength:.2f})")

        elif action == "CALL":
            # Only call with strong hands on bubble
            if hand_strength < 0.70:
                action = "FOLD"
                amount = 0
                logger.info(f"[ICM] Bubble: CALL->FOLD (hand_strength={hand_strength:.2f})")

        return action, amount

    def _adjust_for_short_stack(
        self,
        action: str,
        amount: int,
        hand_strength: float,
        stack_bb: float,
        to_call: int,
        our_chips: int,
        pot: int
    ) -> tuple[str, int]:
        """
        Short stack play: Push/fold strategy.

        With <10BB, can't afford to play small ball. Either shove or fold.
        """
        logger.info(f"[ICM] SHORT STACK: {stack_bb:.1f}BB")

        # Critical short stack (<5BB): Very wide push range
        if stack_bb < self.critical_stack_bb:
            if action in ["RAISE", "CALL"]:
                if hand_strength > 0.40:  # Wide range when desperate
                    # Go all-in
                    action = "RAISE"
                    amount = our_chips
                    logger.info(f"[ICM] Critical stack: ALL-IN (hand_strength={hand_strength:.2f})")
                else:
                    action = "FOLD"
                    amount = 0
                    logger.info(f"[ICM] Critical stack: FOLD (hand_strength={hand_strength:.2f})")

        # Short stack (5-10BB): Tight push range
        else:
            if action == "RAISE":
                # Convert raises to all-in
                if hand_strength > 0.55:
                    action = "RAISE"
                    amount = our_chips
                    logger.info(f"[ICM] Short stack: RAISE->ALL-IN")
                else:
                    action = "FOLD"
                    amount = 0
                    logger.info(f"[ICM] Short stack: RAISE->FOLD (hand_strength={hand_strength:.2f})")

            elif action == "CALL":
                # Only call with good hands when short
                if to_call > our_chips * 0.5:  # Calling >50% of stack
                    if hand_strength > 0.60:
                        # Pot committed - might as well shove
                        action = "RAISE"
                        amount = our_chips
                        logger.info(f"[ICM] Short stack: CALL->ALL-IN (pot committed)")
                    else:
                        action = "FOLD"
                        amount = 0
                        logger.info(f"[ICM] Short stack: CALL->FOLD (hand_strength={hand_strength:.2f})")

        return action, amount

    def _adjust_for_big_stack(
        self,
        action: str,
        amount: int,
        hand_strength: float,
        to_call: int,
        our_chips: int
    ) -> tuple[str, int]:
        """
        Big stack play: Protect the lead, avoid unnecessary risks.

        With a big stack, we can wait for premium spots. Don't gamble.
        """
        logger.info(f"[ICM] BIG STACK: {our_chips} chips")

        # With big stack, don't risk it on marginal hands
        if action in ["RAISE", "CALL"]:
            # If risking significant portion of stack
            risk_pct = (to_call + amount) / our_chips

            if risk_pct > 0.3:  # Risking >30% of stack
                if hand_strength < 0.75:
                    # Not strong enough to risk big stack
                    if action == "RAISE":
                        # Downgrade to call if small investment
                        if to_call < our_chips * 0.1:
                            action = "CALL"
                            amount = 0
                            logger.info(f"[ICM] Big stack: RAISE->CALL (protecting lead)")
                        else:
                            action = "FOLD"
                            amount = 0
                            logger.info(f"[ICM] Big stack: RAISE->FOLD (protecting lead)")
                    elif action == "CALL":
                        action = "FOLD"
                        amount = 0
                        logger.info(f"[ICM] Big stack: CALL->FOLD (protecting lead)")

        return action, amount

    def _general_tournament_adjustment(
        self,
        action: str,
        amount: int,
        hand_strength: float,
        to_call: int,
        our_chips: int,
        pot: int
    ) -> tuple[str, int]:
        """
        General tournament adjustments: More conservative than cash game.

        In tournaments, survival > chip accumulation (to a point).
        """
        # Reduce raise sizes slightly (more cautious)
        if action == "RAISE":
            # Cap raises at 80% of original amount (more conservative)
            original_amount = amount
            amount = int(amount * 0.8)

            # But ensure minimum raise
            min_raise = to_call + 10  # Minimum big blind
            amount = max(amount, min_raise)

            if amount != original_amount:
                logger.debug(f"[ICM] Tournament: Reduced raise from {original_amount} to {amount}")

        # Tighten calling range slightly
        if action == "CALL":
            call_pct = to_call / max(our_chips, 1)

            # If calling >25% of stack with marginal hand, fold instead
            if call_pct > 0.25 and hand_strength < 0.65:
                action = "FOLD"
                amount = 0
                logger.info(f"[ICM] Tournament: CALL->FOLD (risk={call_pct:.1%}, strength={hand_strength:.2f})")

        return action, amount

    def should_play_hand_preflop(
        self,
        hand_strength: float,
        position: int,
        num_players: int,
        stack_bb: float
    ) -> bool:
        """
        Determine if we should play a hand preflop based on ICM.

        Args:
            hand_strength: Preflop hand strength [0, 1]
            position: Our position (0=early, higher=later)
            num_players: Players left in tournament
            stack_bb: Our stack in big blinds

        Returns:
            True if should play, False if should fold
        """
        # Bubble: Very tight
        if self._is_bubble(num_players):
            return hand_strength > 0.75

        # Short stack: Position matters less, hand strength matters more
        if stack_bb < self.short_stack_bb:
            return hand_strength > 0.50

        # Normal play: Standard ranges
        # Adjust for position (play tighter in early position)
        position_factor = position / max(num_players, 1)
        threshold = 0.55 - (position_factor * 0.15)  # Later position = lower threshold

        return hand_strength > threshold


if __name__ == "__main__":
    # Test ICM adjuster
    print("Testing ICMAdjuster...")

    adjuster = ICMAdjuster(total_players=7, starting_chips=1000)

    # Test bubble play
    game_state_bubble = {
        "num_players": 8,  # Bubble!
        "our_chips": 800,
        "pot": 100,
        "to_call": 50,
        "big_blind": 10
    }

    print("\n1. Bubble play - RAISE with marginal hand:")
    action, amount = adjuster.adjust_action("RAISE", 150, game_state_bubble, hand_strength=0.70)
    print(f"   Original: RAISE 150 | Adjusted: {action} {amount}")

    # Test short stack
    game_state_short = {
        "num_players": 5,
        "our_chips": 80,  # 8BB
        "pot": 50,
        "to_call": 20,
        "big_blind": 10
    }

    print("\n2. Short stack (8BB) - RAISE:")
    action, amount = adjuster.adjust_action("RAISE", 40, game_state_short, hand_strength=0.65)
    print(f"   Original: RAISE 40 | Adjusted: {action} {amount}")

    # Test big stack
    game_state_big = {
        "num_players": 4,
        "our_chips": 3000,  # Big stack
        "pot": 200,
        "to_call": 800,  # Large call
        "big_blind": 20
    }

    print("\n3. Big stack - CALL large bet with decent hand:")
    action, amount = adjuster.adjust_action("CALL", 0, game_state_big, hand_strength=0.70)
    print(f"   Original: CALL | Adjusted: {action} {amount}")

    # Test preflop decisions
    print("\n4. Preflop hand selection:")
    should_play_bubble = adjuster.should_play_hand_preflop(0.70, 2, 8, 15)
    print(f"   Bubble (strength=0.70): Play={should_play_bubble}")

    should_play_short = adjuster.should_play_hand_preflop(0.60, 2, 5, 8)
    print(f"   Short stack (strength=0.60): Play={should_play_short}")

    print("\n✅ ICMAdjuster test complete!")
