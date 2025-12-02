"""
Action Translator: Converts RL agent output to competition JSON format.

Agent outputs:
- action: Integer 0-3 (0=fold, 1=call, 2=check, 3=raise)
- amount: Float 0-1 (continuous bet sizing ratio)

Competition expects:
- JSON message: {"type": "act", "action": "FOLD/CALL/CHECK/RAISE", "amount": chips}
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ActionTranslator:
    """Translates RL agent output to competition JSON actions."""

    # Action index mapping
    ACTION_MAP = {
        0: "FOLD",
        1: "CALL",
        2: "CHECK",
        3: "RAISE"
    }

    def __init__(self, big_blind: int = 10, min_raise: int = 10):
        """
        Initialize action translator.

        Args:
            big_blind: Big blind amount (default: 10)
            min_raise: Minimum raise amount (default: 10)
        """
        self.big_blind = big_blind
        self.min_raise = min_raise

    def translate(
        self,
        action_idx: int,
        amount_ratio: float,
        game_state: Dict
    ) -> Dict:
        """
        Convert agent output to competition JSON action.

        Args:
            action_idx: Action index (0=fold, 1=call, 2=check, 3=raise)
            amount_ratio: Bet sizing ratio [0, 1] (only used for raises)
            game_state: Current game state info (pot, to_call, our_chips, etc.)

        Returns:
            JSON action message: {"type": "act", "action": "RAISE", "amount": 100}
        """
        # Map action index to name
        action_name = self.ACTION_MAP.get(action_idx, "FOLD")

        pot = game_state.get("pot", 0)
        to_call = game_state.get("to_call", 0)
        our_chips = game_state.get("our_chips", 0)
        current_bet = game_state.get("current_bet", 0)

        # Validate and translate action
        if action_name == "FOLD":
            return self._make_fold()

        elif action_name == "CALL":
            return self._make_call(to_call, our_chips)

        elif action_name == "CHECK":
            return self._make_check(to_call)

        elif action_name == "RAISE":
            return self._make_raise(amount_ratio, pot, to_call, our_chips, current_bet)

        else:
            # Default fallback: fold
            logger.warning(f"Unknown action index {action_idx}, defaulting to FOLD")
            return self._make_fold()

    def _make_fold(self) -> Dict:
        """Create FOLD action."""
        return {
            "type": "act",
            "action": "FOLD",
            "amount": 0
        }

    def _make_call(self, to_call: int, our_chips: int) -> Dict:
        """
        Create CALL action.

        If to_call exceeds our chips, this becomes an all-in call
        (the server handles this automatically).
        """
        return {
            "type": "act",
            "action": "CALL",
            "amount": 0  # CALL doesn't need amount, server knows how much
        }

    def _make_check(self, to_call: int) -> Dict:
        """
        Create CHECK action.

        If there's a bet to call, CHECK is invalid - fall back to CALL.
        """
        if to_call > 0:
            # Can't check when facing a bet - call instead
            logger.info(f"Can't CHECK with {to_call} to call, switching to CALL")
            return self._make_call(to_call, 9999)  # Assume we can call

        return {
            "type": "act",
            "action": "CHECK",
            "amount": 0
        }

    def _make_raise(
        self,
        amount_ratio: float,
        pot: int,
        to_call: int,
        our_chips: int,
        current_bet: int
    ) -> Dict:
        """
        Create RAISE action with proper sizing.

        Args:
            amount_ratio: Agent's bet sizing ratio [0, 1]
            pot: Current pot size
            to_call: Amount we need to call
            our_chips: Our remaining chips
            current_bet: Current bet amount in the round

        Returns:
            RAISE action with calculated amount
        """
        # Scale amount_ratio to raise size
        # Use pot-based sizing: amount_ratio * pot * scaling_factor
        # Typical scaling: 0.5 (half pot) to 2.0 (double pot)
        scaling_factor = 0.5 + (amount_ratio * 1.5)  # Range: 0.5x to 2.0x pot
        raw_raise = int(pot * scaling_factor)

        # Raise amount is TOTAL amount to put in, not just the raise size
        # Competition server expects: total chips to commit
        raise_amount = to_call + raw_raise

        # Apply minimum raise constraint
        # Minimum raise = current_bet * 2 (at least double the bet)
        min_total_raise = max(current_bet * 2, to_call + self.min_raise)
        raise_amount = max(raise_amount, min_total_raise)

        # Cap at our chips (all-in protection)
        raise_amount = min(raise_amount, our_chips)

        # Ensure raise is at least min_raise above to_call
        if raise_amount < to_call + self.min_raise:
            if our_chips >= to_call + self.min_raise:
                raise_amount = to_call + self.min_raise
            else:
                # Can't make minimum raise - go all-in or call
                if our_chips > to_call:
                    raise_amount = our_chips  # All-in
                else:
                    # Can't raise, fall back to call
                    logger.info(f"Can't raise {raise_amount}, falling back to CALL")
                    return self._make_call(to_call, our_chips)

        logger.debug(f"RAISE: ratio={amount_ratio:.2f}, pot={pot}, raw={raw_raise}, final={raise_amount}")

        return {
            "type": "act",
            "action": "RAISE",
            "amount": int(raise_amount)
        }

    def validate_action(self, action: Dict, game_state: Dict) -> bool:
        """
        Validate that an action is legal in the current game state.

        Args:
            action: JSON action message
            game_state: Current game state

        Returns:
            True if action is valid, False otherwise
        """
        action_type = action.get("action", "")
        to_call = game_state.get("to_call", 0)
        our_chips = game_state.get("our_chips", 0)

        # CHECK is only valid with no bet to call
        if action_type == "CHECK" and to_call > 0:
            return False

        # RAISE must be at least minimum raise
        if action_type == "RAISE":
            amount = action.get("amount", 0)
            if amount < to_call + self.min_raise and amount < our_chips:
                return False

        # CALL/RAISE require chips
        if action_type in ["CALL", "RAISE"] and our_chips <= 0:
            return False

        return True

    def fallback_action(self, game_state: Dict) -> Dict:
        """
        Get a safe fallback action when agent fails.

        Returns FOLD if facing a bet, CHECK otherwise.
        """
        to_call = game_state.get("to_call", 0)

        if to_call > 0:
            # Facing a bet - fold
            logger.warning("Fallback action: FOLD (facing bet)")
            return self._make_fold()
        else:
            # No bet - check
            logger.warning("Fallback action: CHECK (no bet)")
            return self._make_check(0)


if __name__ == "__main__":
    # Test action translator
    print("Testing ActionTranslator...")

    translator = ActionTranslator(big_blind=10, min_raise=10)

    # Test game state
    game_state = {
        "pot": 100,
        "to_call": 20,
        "our_chips": 500,
        "current_bet": 20
    }

    # Test different actions
    print("\n1. FOLD (action_idx=0):")
    action = translator.translate(0, 0.0, game_state)
    print(f"   {action}")

    print("\n2. CALL (action_idx=1):")
    action = translator.translate(1, 0.0, game_state)
    print(f"   {action}")

    print("\n3. CHECK (action_idx=2) - should convert to CALL since facing bet:")
    action = translator.translate(2, 0.0, game_state)
    print(f"   {action}")

    print("\n4. RAISE (action_idx=3) with amount_ratio=0.5:")
    action = translator.translate(3, 0.5, game_state)
    print(f"   {action}")
    print(f"   Valid: {translator.validate_action(action, game_state)}")

    print("\n5. RAISE (action_idx=3) with amount_ratio=0.9 (large raise):")
    action = translator.translate(3, 0.9, game_state)
    print(f"   {action}")

    # Test with no bet (can check)
    game_state_no_bet = {
        "pot": 50,
        "to_call": 0,
        "our_chips": 500,
        "current_bet": 0
    }

    print("\n6. CHECK (action_idx=2) with no bet:")
    action = translator.translate(2, 0.0, game_state_no_bet)
    print(f"   {action}")

    print("\n7. Fallback action (facing bet):")
    action = translator.fallback_action(game_state)
    print(f"   {action}")

    print("\n8. Fallback action (no bet):")
    action = translator.fallback_action(game_state_no_bet)
    print(f"   {action}")

    print("\n✅ ActionTranslator test complete!")
