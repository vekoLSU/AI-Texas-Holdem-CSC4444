from typing import Any, Dict, List, Tuple, Union
from collections import Counter
import itertools


CardInput = Union[str, Dict[str, Any], Tuple[Any, Any], List[Any]]


class HandEvaluator:
    # Poker hand evaluator
    
    # Hand rankings (higher is better)
    HAND_RANKS = {
        "royal_flush": 10,
        "straight_flush": 9,
        "four_of_kind": 8,
        "full_house": 7,
        "flush": 6,
        "straight": 5,
        "three_of_kind": 4,
        "two_pair": 3,
        "pair": 2,
        "high_card": 1
    }
    
    # Card values
    CARD_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    # Preflop hand strength chart (Chen formula adapted)
    PREFLOP_STRENGTH = {
        'AA': 1.00, 'KK': 0.98, 'QQ': 0.96, 'JJ': 0.94, 'TT': 0.92,
        'AKs': 0.91, 'AQs': 0.89, 'AJs': 0.87, 'AKo': 0.86, 'ATs': 0.85,
        '99': 0.84, 'AQo': 0.83, 'KQs': 0.82, 'AJo': 0.81, '88': 0.80,
    }
    
    def __init__(self):
        pass
    
    def evaluate_hand_strength(self, hole_cards: List[CardInput], community_cards: List[CardInput], phase: str) -> Dict:
        # Get hand strength
        
        if phase == "PREFLOP":
            return self.evaluate_preflop(hole_cards)
        else:
            return self.evaluate_postflop(hole_cards, community_cards, phase)
    
    def evaluate_preflop(self, hole_cards: List[CardInput]) -> Dict:
        # Preflop strength
        if len(hole_cards) != 2:
            return {"strength": 0.0, "hand_type": "unknown", "confidence": 0.0}
        
        card1, card2 = hole_cards
        rank1, suit1 = self.parse_card(card1)
        rank2, suit2 = self.parse_card(card2)
        
        # Normalize to standard notation
        val1 = self.CARD_VALUES[rank1]
        val2 = self.CARD_VALUES[rank2]
        
        # Higher card first
        if val1 < val2:
            rank1, rank2 = rank2, rank1
        
        # Determine suited/offsuit
        suited = 's' if suit1 == suit2 else 'o'
        
        # Check for pair
        if rank1 == rank2:
            hand_notation = f"{rank1}{rank2}"
        else:
            hand_notation = f"{rank1}{rank2}{suited}"
        
        # Look up preflop strength
        base_strength = self.PREFLOP_STRENGTH.get(hand_notation, self.calculate_preflop_strength(val1, val2, suited == 's'))
        
        # Classify hand type
        if val1 == val2:
            if val1 >= 13:
                hand_type = "premium_pair"
            elif val1 >= 10:
                hand_type = "strong_pair"
            else:
                hand_type = "pocket_pair"
        elif suited == 's' and val1 == 14:
            hand_type = "suited_ace"
        elif val1 >= 12 and val2 >= 10:
            hand_type = "broadway"
        elif suited == 's' and abs(val1 - val2) <= 4:
            hand_type = "suited_connector"
        else:
            hand_type = "weak"
        
        return {
            "strength": base_strength,
            "hand_type": hand_type,
            "confidence": 0.85,
            "notation": hand_notation
        }
    
    def calculate_preflop_strength(self, val1: int, val2: int, suited: bool) -> float:
        """Calculate preflop strength using modified Chen formula"""
        high_card = max(val1, val2)
        low_card = min(val1, val2)
        
        # Base score from high card
        if high_card == 14:  # Ace
            score = 10
        elif high_card == 13:  # King
            score = 8
        elif high_card == 12:  # Queen
            score = 7
        elif high_card == 11:  # Jack
            score = 6
        else:
            score = max(high_card - 9, 0)
        
        # Pair bonus
        if val1 == val2:
            score = max(score * 2, 10)
        
        # Suited bonus
        if suited:
            score += 2
        
        # Gap penalty
        gap = high_card - low_card - 1
        if gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        elif gap >= 4:
            score -= 5
        
        # Straight bonus
        if gap == 0:  # Connector
            score += 1
        
        # Normalize to 0-1 range
        return min(max(score / 20.0, 0.0), 1.0)
    
    def evaluate_postflop(self, hole_cards: List[CardInput], community_cards: List[CardInput], phase: str) -> Dict:
        """Evaluate hand strength after the flop"""
        all_cards = hole_cards + community_cards

        if len(all_cards) < 5:
            # Can't make a full hand yet
            return {"strength": 0.3, "hand_type": "incomplete", "confidence": 0.5}

        # Find best 5-card combination
        best_hand = self.find_best_hand(all_cards)

        # Calculate hand strength based on made hand with context
        base_strength = self.get_hand_base_strength_contextual(best_hand, hole_cards, community_cards)

        # Calculate draw potential
        draw_strength = self.calculate_draw_potential(hole_cards, community_cards, phase)

        # Combine made hand and draw strength (reduce draw weight for strong hands)
        if base_strength >= 0.70:
            # Strong hands: focus on made hand
            total_strength = base_strength * 0.9 + draw_strength * 0.1
        else:
            # Weaker hands: draws matter more
            total_strength = base_strength * 0.7 + draw_strength * 0.3

        return {
            "strength": min(total_strength, 1.0),
            "hand_type": best_hand["type"],
            "hand_rank": best_hand["rank"],
            "draw_potential": draw_strength,
            "confidence": 0.9
        }
    
    def find_best_hand(self, cards: List[CardInput]) -> Dict:
        """Find the best 5-card poker hand from available cards"""
        if len(cards) < 5:
            return {"type": "incomplete", "rank": 0, "high_cards": []}
        
        # Check all 5-card combinations
        best_hand = {"type": "high_card", "rank": 1, "high_cards": []}
        best_value = 0
        
        for combo in itertools.combinations(cards, 5):
            hand = self.classify_hand(list(combo))
            hand_value = self.HAND_RANKS.get(hand["type"], 0)
            
            if hand_value > best_value:
                best_value = hand_value
                best_hand = hand
        
        return best_hand
    
    def classify_hand(self, cards: List[CardInput]) -> Dict:
        """Classify a 5-card hand"""
        if len(cards) != 5:
            return {"type": "incomplete", "rank": 0}
        
        parsed = [self.parse_card(c) for c in cards]
        ranks = [self.CARD_VALUES[r] for r, s in parsed]
        suits = [s for r, s in parsed]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = max(suit_counts.values()) == 5
        is_straight = self.is_straight(ranks)
        
        # Royal Flush
        if is_flush and is_straight and max(ranks) == 14 and min(ranks) == 10:
            return {"type": "royal_flush", "rank": 10, "high_cards": sorted(ranks, reverse=True)}
        
        # Straight Flush
        if is_flush and is_straight:
            return {"type": "straight_flush", "rank": 9, "high_cards": sorted(ranks, reverse=True)}
        
        # Four of a Kind
        if 4 in rank_counts.values():
            return {"type": "four_of_kind", "rank": 8, "high_cards": sorted(ranks, reverse=True)}
        
        # Full House
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return {"type": "full_house", "rank": 7, "high_cards": sorted(ranks, reverse=True)}
        
        # Flush
        if is_flush:
            return {"type": "flush", "rank": 6, "high_cards": sorted(ranks, reverse=True)}
        
        # Straight
        if is_straight:
            return {"type": "straight", "rank": 5, "high_cards": sorted(ranks, reverse=True)}
        
        # Three of a Kind
        if 3 in rank_counts.values():
            return {"type": "three_of_kind", "rank": 4, "high_cards": sorted(ranks, reverse=True)}
        
        # Two Pair
        if list(rank_counts.values()).count(2) == 2:
            return {"type": "two_pair", "rank": 3, "high_cards": sorted(ranks, reverse=True)}
        
        # Pair
        if 2 in rank_counts.values():
            return {"type": "pair", "rank": 2, "high_cards": sorted(ranks, reverse=True)}
        
        # High Card
        return {"type": "high_card", "rank": 1, "high_cards": sorted(ranks, reverse=True)}
    
    def is_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks))
        
        if len(sorted_ranks) != 5:
            return False
        
        # Normal straight
        if sorted_ranks[-1] - sorted_ranks[0] == 4:
            return True
        
        # Ace-low straight (A-2-3-4-5)
        if sorted_ranks == [2, 3, 4, 5, 14]:
            return True
        
        return False
    
    def calculate_draw_potential(self, hole_cards: List[CardInput], community_cards: List[CardInput], phase: str) -> float:
        """Calculate potential for improving hand on future streets"""
        if phase == "RIVER":
            return 0.0  # No more cards coming
        
        all_cards = hole_cards + community_cards
        parsed = [self.parse_card(c) for c in all_cards]
        ranks = [self.CARD_VALUES[r] for r, s in parsed]
        suits = [s for r, s in parsed]
        
        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)
        
        draw_score = 0.0
        
        # Flush draw
        if max(suit_counts.values()) == 4:
            # 9 outs for flush
            outs = 9
            if phase == "FLOP":
                draw_score += 0.35  # ~35% to hit by river
            else:  # TURN
                draw_score += 0.20  # ~20% to hit on river
        
        # Straight draw
        if self.has_straight_draw(ranks):
            # Open-ended = 8 outs, gutshot = 4 outs
            is_open_ended = self.is_open_ended_straight_draw(ranks)
            if is_open_ended:
                if phase == "FLOP":
                    draw_score += 0.32
                else:
                    draw_score += 0.17
            else:
                if phase == "FLOP":
                    draw_score += 0.17
                else:
                    draw_score += 0.09
        
        # Overcards
        hole_ranks = [self.CARD_VALUES[self.parse_card(c)[0]] for c in hole_cards]
        community_ranks = [self.CARD_VALUES[self.parse_card(c)[0]] for c in community_cards] if community_cards else []
        max_community = max(community_ranks) if community_ranks else 0
        overcards = sum(1 for hr in hole_ranks if hr > max_community)
        
        if overcards > 0:
            draw_score += overcards * 0.1
        
        return min(draw_score, 1.0)
    
    def has_straight_draw(self, ranks: List[int]) -> bool:
        """Check if there's a straight draw"""
        unique_ranks = sorted(set(ranks))
        
        # Check for 4 in a row
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i+3] - unique_ranks[i] == 3:
                return True
        
        # Check for gaps that could be filled
        if len(unique_ranks) >= 4:
            for combo in itertools.combinations(unique_ranks, 4):
                sorted_combo = sorted(combo)
                if sorted_combo[-1] - sorted_combo[0] <= 4:
                    return True
        
        return False
    
    def is_open_ended_straight_draw(self, ranks: List[int]) -> bool:
        """Check if it's an open-ended straight draw"""
        unique_ranks = sorted(set(ranks))
        
        # Look for 4 consecutive cards
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i+3] - unique_ranks[i] == 3:
                return True
        
        return False
    
    def get_hand_base_strength(self, hand: Dict) -> float:
        """Convert hand rank to strength value (legacy method - use contextual version)"""
        hand_rank = hand.get("rank", 1)

        # Map rank to strength (0.0 to 1.0)
        strength_map = {
            10: 1.00,  # Royal Flush
            9: 0.95,   # Straight Flush
            8: 0.90,   # Four of a Kind
            7: 0.85,   # Full House
            6: 0.75,   # Flush
            5: 0.65,   # Straight
            4: 0.75,   # Three of a Kind (FIXED: was 0.55)
            3: 0.55,   # Two Pair (FIXED: was 0.45)
            2: 0.40,   # Pair (base - will be adjusted by context)
            1: 0.20,   # High Card
        }

        return strength_map.get(hand_rank, 0.2)

    def get_hand_base_strength_contextual(self, hand: Dict, hole_cards: List[CardInput], community_cards: List[CardInput]) -> float:
        """Context-aware hand strength evaluation"""
        hand_rank = hand.get("rank", 1)
        hand_type = hand.get("type", "high_card")
        high_cards = hand.get("high_cards", [])

        # Parse hole cards
        hole_ranks = [self.CARD_VALUES[self.parse_card(c)[0]] for c in hole_cards]
        community_ranks = [self.CARD_VALUES[self.parse_card(c)[0]] for c in community_cards]

        # Base strength from hand type
        base_strength = self.get_hand_base_strength(hand)

        # Context adjustments for pairs
        if hand_type == "pair" and len(high_cards) > 0:
            pair_rank = max(r for r in high_cards if high_cards.count(r) >= 2)
            max_board_rank = max(community_ranks) if community_ranks else 0

            # Overpair: both hole cards higher than all board cards
            if len(hole_ranks) == 2 and hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > max_board_rank:
                # Pocket pair that's an overpair
                if pair_rank >= 13:  # AA/KK
                    base_strength = 0.85
                elif pair_rank >= 11:  # QQ/JJ
                    base_strength = 0.80
                elif pair_rank >= 9:  # TT/99
                    base_strength = 0.75
                else:  # 88 and below
                    base_strength = 0.70
            # Top pair: paired with highest board card
            elif pair_rank == max_board_rank:
                # Check kicker strength
                kicker = max([r for r in hole_ranks if r != pair_rank], default=0)
                if kicker >= 12:  # A or K kicker
                    base_strength = 0.65
                elif kicker >= 10:  # Q/J kicker
                    base_strength = 0.60
                else:
                    base_strength = 0.55
            # Middle pair
            elif pair_rank > min(community_ranks) and pair_rank < max_board_rank:
                base_strength = 0.45
            # Bottom pair or weak pair
            else:
                base_strength = 0.35

        # Context adjustments for three of a kind
        elif hand_type == "three_of_kind":
            # Trips are very strong - check if it's set (pocket pair) or trips (board pair)
            trip_rank = max(r for r in high_cards if high_cards.count(r) >= 3)
            is_set = len(hole_ranks) == 2 and hole_ranks[0] == hole_ranks[1]

            if is_set:
                # Set (pocket pair + one on board) - very strong, well-disguised
                base_strength = 0.85
            else:
                # Trips (one in hand + pair on board) - still very strong
                base_strength = 0.78

        # Context adjustments for two pair
        elif hand_type == "two_pair":
            # Check if both pairs use hole cards (very strong) vs one pair from board
            pairs = [r for r in set(high_cards) if high_cards.count(r) >= 2]
            if len(pairs) >= 2:
                top_pair = max(pairs)
                if top_pair >= 12:  # Ace or King high two pair
                    base_strength = 0.70
                else:
                    base_strength = 0.65

        # Adjust for board texture danger
        board_danger = self.assess_board_danger(community_cards)
        if board_danger > 0.3 and hand_rank <= 5:  # Dangerous board for weaker hands
            base_strength *= (1.0 - board_danger * 0.2)

        return min(base_strength, 1.0)

    def assess_board_danger(self, community_cards: List[CardInput]) -> float:
        """Assess how dangerous/coordinated the board is"""
        if len(community_cards) < 3:
            return 0.0

        parsed = [self.parse_card(c) for c in community_cards]
        ranks = [self.CARD_VALUES[r] for r, s in parsed]
        suits = [s for r, s in parsed]

        danger = 0.0

        # Flush danger
        suit_counts = Counter(suits)
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 3:
            danger += 0.3
        if max_suit_count >= 4:
            danger += 0.3

        # Straight danger
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 3:
            for i in range(len(unique_ranks) - 2):
                if unique_ranks[i+2] - unique_ranks[i] <= 4:
                    danger += 0.2
                    break

        # Paired board danger
        rank_counts = Counter(ranks)
        if 2 in rank_counts.values():
            danger += 0.15
        if 3 in rank_counts.values():
            danger += 0.3

        return min(danger, 1.0)
    
    def parse_card(self, card: CardInput) -> Tuple[str, str]:
        """
        Parse a card coming from various sources into (rank, suit_digit).
        Supports engine JSON objects, tuples, and classic string formats.
        """
        if isinstance(card, dict):
            rank = self._normalize_rank(card.get("rank"))
            suit = self._normalize_suit(card.get("suit"))
            return rank, suit

        if isinstance(card, (tuple, list)) and len(card) >= 2:
            rank = self._normalize_rank(card[0])
            suit = self._normalize_suit(card[1])
            return rank, suit

        if isinstance(card, str):
            text = card.strip()
            if not text:
                return "X", "x"
            if len(text) >= 2 and text[0].isdigit() and text[1].isdigit():
                # Handle formats like "10h"
                rank_part = text[:-1]
                suit_part = text[-1]
            else:
                rank_part = text[0]
                suit_part = text[1] if len(text) > 1 else ""
            return self._normalize_rank(rank_part), self._normalize_suit(suit_part)

        return "X", "x"

    def _normalize_rank(self, raw_rank: Any) -> str:
        """Normalize rank representations to single-character format."""
        if raw_rank is None:
            return "X"

        rank_str = str(raw_rank).upper()
        if rank_str in self.CARD_VALUES:
            return rank_str

        # Convert alternate representations
        if rank_str in {"10", "T10"}:
            return "T"
        if rank_str == "1":  # Ace sometimes encoded as 1
            return "A"

        return rank_str[:1] if rank_str else "X"

    def _normalize_suit(self, raw_suit: Any) -> str:
        """Normalize suits to single-letter notation."""
        if raw_suit is None:
            return "x"

        suit_str = str(raw_suit).upper()
        suit_map = {
            "HEART": "h",
            "HEARTS": "h",
            "DIAMOND": "d",
            "DIAMONDS": "d",
            "CLUB": "c",
            "CLUBS": "c",
            "SPADE": "s",
            "SPADES": "s",
            "H": "h",
            "D": "d",
            "C": "c",
            "S": "s"
        }

        return suit_map.get(suit_str, suit_str.lower()[:1] if suit_str else "x")
