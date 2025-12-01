from typing import Dict, List, Optional
import math

class DecisionEngine:
    # Poker bot decision engine
    
    def __init__(self):
        self.aggression_factor = 1.2  # How aggressive the bot is
        self.bluff_frequency = 0.15   # 15% bluff rate
        
    def decide(self, hand_cards: List[str], community_cards: List[str], 
               hand_strength: Dict, phase: str, pot: int, to_call: int,
               our_chips: int, position: str, num_players: int,
               opponent_profiles: List[Dict], current_bet: int) -> Dict:
        # Decide action
        
        strength = hand_strength.get("strength", 0.0)
        hand_type = hand_strength.get("hand_type", "unknown")
        
        # Calculate pot odds
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
        else:
            pot_odds = 0.0
        
        # Position multiplier (late position = more aggressive)
        position_multiplier = self.get_position_multiplier(position)
        
        # Adjust strength based on number of players
        adjusted_strength = self.adjust_for_players(strength, num_players)
        
        # Adjust based on position
        adjusted_strength *= position_multiplier
        
        # Integrate opponent profiles for dynamic aggression and bluffing
        avg_aggression = 1.0
        avg_passivity = 1.0
        avg_vpip = 0.2
        if opponent_profiles:
            total_agg = sum([op.get('aggression', 1.0) for op in opponent_profiles])
            total_pass = sum([op.get('passivity', 1.0) for op in opponent_profiles])
            total_vpip = sum([op.get('vpip', 0.2) for op in opponent_profiles])
            avg_aggression = total_agg / len(opponent_profiles)
            avg_passivity = total_pass / len(opponent_profiles)
            avg_vpip = total_vpip / len(opponent_profiles)
        # Adjust aggression factor and bluff frequency
        self.aggression_factor = 1.2 + (avg_passivity - avg_aggression) * 0.2
        self.bluff_frequency = 0.15 + (0.25 - avg_vpip) * 0.1
        # Clamp values
        self.aggression_factor = max(0.8, min(self.aggression_factor, 2.0))
        self.bluff_frequency = max(0.05, min(self.bluff_frequency, 0.25))

        # Logging for analysis
        import logging
        logging.basicConfig(filename='bot_decisions.log', level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info(f"Phase: {phase}, Strength: {strength:.2f}, Pot: {pot}, ToCall: {to_call}, Position: {position}, Players: {num_players}, AggFactor: {self.aggression_factor:.2f}, BluffFreq: {self.bluff_frequency:.2f}")

        # Phase-specific strategy
        if phase == "PREFLOP":
            return self.decide_preflop(
                adjusted_strength, hand_type, pot, to_call, our_chips,
                position, num_players, pot_odds
            )
        else:
            return self.decide_postflop(
                adjusted_strength, hand_type, hand_strength, pot, to_call,
                our_chips, position, num_players, pot_odds, phase
            )
    
    def decide_preflop(self, strength: float, hand_type: str, pot: int,
                       to_call: int, our_chips: int, position: str,
                       num_players: int, pot_odds: float) -> Dict:
        # Preflop logic
        
        # Premium hands (AA, KK, QQ, AK)
        if strength >= 0.90:
            # Always raise with premium hands
            if to_call == 0:
                # No one has bet - make a strong raise
                raise_amount = max(pot * 0.75, pot + 30)
                return {"action": "raise", "amount": min(raise_amount, our_chips)}
            elif to_call < our_chips * 0.25:
                # Re-raise with premium hands
                raise_amount = to_call * 3 + pot * 0.5
                return {"action": "raise", "amount": min(raise_amount, our_chips)}
            else:
                # Big bet in front - call or push
                if strength >= 0.95:
                    return {"action": "raise", "amount": our_chips}  # All-in with AA/KK
                else:
                    return {"action": "call"}
        
        # Strong hands (JJ, TT, AQ, AJ suited)
        elif strength >= 0.75:
            if to_call == 0:
                raise_amount = pot * 0.6 + 20
                return {"action": "raise", "amount": min(raise_amount, our_chips)}
            elif to_call < our_chips * 0.15:
                return {"action": "call"}
            elif to_call < our_chips * 0.30 and position in ["late", "middle"]:
                return {"action": "call"}
            else:
                return {"action": "fold"}
        
        # Medium hands (pairs 77-99, suited connectors, broadway)
        elif strength >= 0.55:
            if to_call == 0:
                # Open raise from good position
                if position in ["late", "middle"]:
                    raise_amount = pot * 0.5 + 15
                    return {"action": "raise", "amount": min(raise_amount, our_chips)}
                else:
                    return {"action": "check"}
            elif to_call < pot * 0.3:
                # Good odds - call
                return {"action": "call"}
            elif to_call < our_chips * 0.1 and num_players <= 4:
                # Small bet, few players - call
                return {"action": "call"}
            else:
                return {"action": "fold"}
        
        # Speculative hands (small pairs, suited cards)
        elif strength >= 0.40:
            if to_call == 0:
                if position == "late" and num_players <= 3:
                    # Steal attempt
                    raise_amount = pot * 0.6
                    return {"action": "raise", "amount": min(raise_amount, our_chips)}
                else:
                    return {"action": "check"}
            elif to_call < pot * 0.2 and num_players >= 3:
                # Multiway pot with good odds
                return {"action": "call"}
            else:
                return {"action": "fold"}
        
        # Weak hands
        else:
            if to_call == 0 and position == "late" and num_players == 2:
                # Blind steal attempt heads-up
                raise_amount = pot * 0.7
                return {"action": "raise", "amount": min(raise_amount, our_chips)}
            elif to_call == 0:
                return {"action": "check"}
            else:
                return {"action": "fold"}
    
    def decide_postflop(self, strength: float, hand_type: str,
                        hand_strength: Dict, pot: int, to_call: int,
                        our_chips: int, position: str, num_players: int,
                        pot_odds: float, phase: str) -> Dict:
        # Postflop logic
        
        draw_potential = hand_strength.get("draw_potential", 0.0)
        
        # Calculate effective strength (made hand + draws)
        effective_strength = strength + draw_potential * (0.7 if num_players <= 3 else 0.5)
        
        # Very strong hands (two pair or better)
        if strength >= 0.75:
            if to_call == 0:
                # Bet for value
                bet_size = self.calculate_value_bet(pot, strength, num_players)
                return {"action": "raise", "amount": min(bet_size, our_chips)}
            elif to_call < pot * 0.8:
                # Raise for value
                raise_amount = to_call * 2.5 + pot * 0.5
                return {"action": "raise", "amount": min(raise_amount, our_chips)}
            else:
                # Large bet in front - call with very strong hands
                if strength >= 0.85:
                    return {"action": "call"}
                else:
                    # Consider fold to huge bet with marginal made hand
                    if to_call > pot * 1.5:
                        return {"action": "fold"}
                    return {"action": "call"}
        
        # Strong hands (top pair good kicker, overpair)
        elif strength >= 0.55:
            if to_call == 0:
                # Bet for value/protection
                bet_size = pot * (0.6 + self.aggression_factor * 0.1)
                return {"action": "raise", "amount": min(bet_size, our_chips)}
            elif pot_odds < 0.3 and to_call < pot:
                # Good price - call
                return {"action": "call"}
            elif to_call > pot * 0.7:
                # Big bet - need to be careful
                if strength >= 0.65 and phase != "RIVER":
                    return {"action": "call"}
                else:
                    return {"action": "fold"}
            else:
                return {"action": "call"}
        
        # Medium hands (middle pair, weak top pair)
        elif strength >= 0.40:
            if to_call == 0:
                # Check or small bet
                if num_players <= 2:
                    bet_size = pot * (0.4 + self.aggression_factor * 0.1)
                    return {"action": "raise", "amount": min(bet_size, our_chips)}
                else:
                    return {"action": "check"}
            elif pot_odds < 0.25:
                return {"action": "call"}
            else:
                return {"action": "fold"}
        
        # Draws and weak hands
        elif effective_strength >= 0.35:
            # We have a draw
            if draw_potential >= 0.25:
                # Strong draw
                if to_call == 0:
                    # Semi-bluff
                    if num_players <= 3:
                        bet_size = pot * (0.5 + self.bluff_frequency * 0.5)
                        return {"action": "raise", "amount": min(bet_size, our_chips)}
                    else:
                        return {"action": "check"}
                elif pot_odds < draw_potential * 0.8:
                    # Good pot odds for draw
                    return {"action": "call"}
                else:
                    return {"action": "fold"}
            else:
                # Weak hand, no draw
                if to_call == 0:
                    return {"action": "check"}
                else:
                    return {"action": "fold"}
        
        # Bluffs and very weak hands
        else:
            if to_call == 0:
                # Bluff opportunity
                if self.should_bluff(position, num_players, phase, pot, our_chips):
                    bet_size = pot * (0.65 + self.bluff_frequency * 0.5)
                    return {"action": "raise", "amount": min(bet_size, our_chips)}
                else:
                    return {"action": "check"}
            else:
                # Fold to any bet with nothing
                return {"action": "fold"}
    
    def calculate_value_bet(self, pot: int, strength: float, num_players: int) -> float:
        # Value bet size
        # Stronger hands bet more
        base_multiplier = 0.5 + (strength - 0.75) * 0.8
        
        # Fewer players = can bet more
        player_adjustment = 1.0 if num_players <= 2 else 0.8
        
        bet_size = pot * base_multiplier * player_adjustment * self.aggression_factor
        
        return max(bet_size, pot * 0.4)  # At least 40% pot
    
    def should_bluff(self, position: str, num_players: int, phase: str,
                     pot: int, our_chips: int) -> bool:
        # Should bluff
        
        # Don't bluff if short-stacked
        if our_chips < pot * 2:
            return False
        
        # More likely to bluff in late position
        if position == "late":
            bluff_chance = self.bluff_frequency * 1.5
        elif position == "middle":
            bluff_chance = self.bluff_frequency
        else:
            bluff_chance = self.bluff_frequency * 0.5
        
        # Less likely to bluff multiway
        if num_players > 2:
            bluff_chance *= 0.5
        
        # More likely to bluff on river (fewer cards to come)
        if phase == "RIVER":
            bluff_chance *= 1.3
        
        # Random decision based on bluff frequency
        import random
        return random.random() < bluff_chance
    
    def get_position_multiplier(self, position: str) -> float:
        # Position multiplier
        multipliers = {
            "late": 1.2,
            "middle": 1.0,
            "early": 0.85,
            "heads-up": 1.15,
            "unknown": 1.0
        }
        return multipliers.get(position, 1.0)
    
    def adjust_for_players(self, strength: float, num_players: int) -> float:
        # Adjust for player count
        # More players = need stronger hand
        if num_players >= 6:
            return strength * 0.85
        elif num_players >= 4:
            return strength * 0.92
        elif num_players == 2:
            return strength * 1.08  # Heads-up - can play more hands
        else:
            return strength
    
    def calculate_ev(self, win_prob: float, pot: int, to_call: int) -> float:
        # Expected Value
        # EV = (Prob of winning * Pot) - (Prob of losing * Cost to call)
        ev = (win_prob * (pot + to_call)) - ((1 - win_prob) * to_call)
        return ev
    
    def get_min_equity_to_call(self, pot: int, to_call: int) -> float:
        # Min equity to call
        if pot + to_call == 0:
            return 1.0
        return to_call / (pot + to_call)
