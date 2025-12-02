from typing import Dict, List
from collections import defaultdict

class OpponentTracker:
    # Tracks opponents and profiles
    
    def __init__(self):
        # Track stats for each opponent
        self.player_stats = defaultdict(lambda: {
            "hands_played": 0,
            "hands_won": 0,
            "total_bets": 0,
            "total_raises": 0,
            "total_calls": 0,
            "total_folds": 0,
            "aggressive_actions": 0,
            "passive_actions": 0,
            "showdown_hands": [],
            "preflop_raises": 0,
            "preflop_calls": 0,
            "preflop_folds": 0,
            "continuation_bets": 0,
            "continuation_bet_opportunities": 0,
            "bluff_attempts": 0,
            "value_bets": 0,
            "chips_won": 0,
            "chips_lost": 0,
            "current_chips": 0,
            "bet_sizes": [],
            "street_aggression": {"FLOP": 0, "TURN": 0, "RIVER": 0},
            "street_calls": {"FLOP": 0, "TURN": 0, "RIVER": 0},
            "street_folds": {"FLOP": 0, "TURN": 0, "RIVER": 0}
        })
        
        self.last_aggressor = None
        self.last_phase = None
    
    def observe_state(self, state: Dict, players: List[Dict]):
        # Update opponent models
        phase = state.get("phase", "UNKNOWN")
        current_player = state.get("currentPlayer")
        
        # Track phase transitions
        if phase != self.last_phase:
            self.last_phase = phase
        
        # Update player chip counts
        for player in players:
            player_id = player.get("id")
            chips = player.get("chips", 0)
            self.player_stats[player_id]["current_chips"] = chips
        
        # Detect betting actions
        if current_player:
            self.last_aggressor = current_player
    
    def record_action(self, player_id: str, action: str, amount: int = 0, phase: str = "UNKNOWN"):
        # Record player action
        stats = self.player_stats[player_id]

        # Track bet sizing
        if amount > 0:
            stats["bet_sizes"].append(amount)

        # Track street aggression/calls/folds
        if phase in ["FLOP", "TURN", "RIVER"]:
            if action in ["raise", "bet"]:
                stats["street_aggression"][phase] += 1
            elif action == "call":
                stats["street_calls"][phase] += 1
            elif action == "fold":
                stats["street_folds"][phase] += 1

        if action == "fold":
            stats["total_folds"] += 1
            stats["passive_actions"] += 1
            if phase == "PREFLOP":
                stats["preflop_folds"] += 1

        elif action == "call":
            stats["total_calls"] += 1
            stats["passive_actions"] += 1
            if phase == "PREFLOP":
                stats["preflop_calls"] += 1

        elif action == "check":
            stats["passive_actions"] += 1

        elif action == "raise" or action == "bet":
            stats["total_raises"] += 1
            stats["total_bets"] += amount
            stats["aggressive_actions"] += 1
            if phase == "PREFLOP":
                stats["preflop_raises"] += 1
    
    def record_hand_result(self, player_id: str, won: bool, chips_delta: int):
        # Record hand result
        stats = self.player_stats[player_id]
        stats["hands_played"] += 1

        if won:
            stats["hands_won"] += 1
            stats["chips_won"] += chips_delta
        else:
            stats["chips_lost"] += abs(chips_delta)

        # Track showdown hands for range analysis
        # This requires passing hand info, so add a placeholder for future integration
        # stats["showdown_hands"].append(hand_info)  # hand_info: dict with cards, board, action history
    
    def get_opponent_profiles(self, players: List[Dict]) -> List[Dict]:
        # Get opponent profiles
        profiles = []
        
        for player in players:
            player_id = player.get("id")
            stats = self.player_stats[player_id]
            
            # Calculate player tendencies
            profile = {
                "id": player_id,
                "player_type": self.classify_player(stats),
                "aggression_factor": self.calculate_aggression_factor(stats),
                "vpip": self.calculate_vpip(stats),
                "pfr": self.calculate_pfr(stats),
                "fold_to_cbet": self.calculate_fold_to_cbet(stats),
                "chips": player.get("chips", 0),
                "is_short_stack": player.get("chips", 0) < 500,  # Arbitrary threshold
                "is_big_stack": player.get("chips", 0) > 2000,
            }
            
            profiles.append(profile)
        
        return profiles
    
    def classify_player(self, stats: Dict) -> str:
        # Classify player type
        hands = stats["hands_played"]
        
        if hands < 5:
            return "unknown"
        
        aggression = self.calculate_aggression_factor(stats)
        vpip = self.calculate_vpip(stats)
        
        # Tight vs Loose threshold
        is_tight = vpip < 0.18
        is_loose = vpip > 0.40

        # Aggressive vs Passive threshold
        is_aggressive = aggression > 2.0
        is_passive = aggression < 0.8

        # Nuanced player types
        if is_tight and is_aggressive:
            return "TAG"  # Tight-Aggressive (strong player)
        elif is_tight and is_passive:
            return "nit"  # Very tight-passive
        elif is_loose and is_aggressive:
            return "maniac"  # Very loose-aggressive
        elif is_loose and is_passive:
            return "fish"  # Loose-Passive (weak player)
        elif is_tight:
            return "rock"  # Tight-Passive
        elif is_loose:
            return "LAG"  # Loose-Aggressive
        else:
            return "unknown"
    
    def calculate_aggression_factor(self, stats: Dict) -> float:
        # Aggression factor
        aggressive = stats["aggressive_actions"]
        passive = max(stats["total_calls"], 1)  # Avoid division by zero
        
        return aggressive / passive
    
    def calculate_vpip(self, stats: Dict) -> float:
        # VPIP stat
        hands = stats["hands_played"]
        if hands == 0:
            return 0.0
        
        # Estimate hands where they put money in
        voluntary_actions = stats["total_calls"] + stats["total_raises"]
        
        # Rough VPIP estimate
        return min(voluntary_actions / max(hands, 1), 1.0)
    
    def calculate_pfr(self, stats: Dict) -> float:
        # PFR stat
        hands = stats["hands_played"]
        if hands == 0:
            return 0.0
        
        return stats["preflop_raises"] / max(hands, 1)
    
    def calculate_fold_to_cbet(self, stats: Dict) -> float:
        # Fold to c-bet stat
        # This would require more detailed tracking
        # For now, return a default value
        total_folds = stats["total_folds"]
        total_actions = (stats["total_folds"] + stats["total_calls"] + 
                        stats["total_raises"])
        
        if total_actions == 0:
            return 0.5  # Default
        
        return total_folds / total_actions
    
    def get_player_tendency(self, player_id: str) -> Dict:
        # Get player tendency
        stats = self.player_stats[player_id]
        
        return {
            "player_type": self.classify_player(stats),
            "aggression_factor": self.calculate_aggression_factor(stats),
            "vpip": self.calculate_vpip(stats),
            "pfr": self.calculate_pfr(stats),
            "win_rate": self.calculate_win_rate(stats),
            "hands_played": stats["hands_played"],
            "confidence": min(stats["hands_played"] / 20.0, 1.0)  # Need 20+ hands for good read
        }
    
    def calculate_win_rate(self, stats: Dict) -> float:
        # Win rate
        hands = stats["hands_played"]
        if hands == 0:
            return 0.0
        
        return stats["hands_won"] / hands
    
    def suggest_adjustment(self, opponent_profile: Dict) -> Dict:
        # Suggest strategy adjustment
        player_type = opponent_profile.get("player_type", "unknown")
        
        adjustments = {
            "TAG": {
                "description": "Strong player - play tight, avoid bluffs",
                "fold_more": True,
                "bluff_less": True,
                "value_bet_thin": False,
                "respect_raises": True
            },
            "rock": {
                "description": "Tight passive - steal blinds, fold to aggression",
                "fold_more": False,
                "bluff_more": True,
                "value_bet_thin": True,
                "respect_raises": True
            },
            "LAG": {
                "description": "Loose aggressive - call down lighter, trap more",
                "fold_more": False,
                "bluff_less": True,
                "value_bet_thin": True,
                "respect_raises": False
            },
            "fish": {
                "description": "Loose passive - value bet heavily, avoid bluffs",
                "fold_more": False,
                "bluff_less": True,
                "value_bet_thin": True,
                "respect_raises": False
            },
            "unknown": {
                "description": "Not enough data - play GTO",
                "fold_more": False,
                "bluff_less": False,
                "value_bet_thin": False,
                "respect_raises": True
            }
        }
        
        return adjustments.get(player_type, adjustments["unknown"])
    
    def reset_hand_tracking(self):
        # Reset for new hand
        self.last_aggressor = None
    
    def get_table_dynamics(self, players: List[Dict]) -> Dict:
        # Table dynamics
        profiles = self.get_opponent_profiles(players)
        
        active_players = [p for p in players if not p.get("folded", False)]
        num_active = len(active_players)
        
        # Count player types
        player_types = [p["player_type"] for p in profiles]
        
        avg_aggression = sum(p["aggression_factor"] for p in profiles) / max(len(profiles), 1)
        
        return {
            "num_active_players": num_active,
            "average_aggression": avg_aggression,
            "table_style": "aggressive" if avg_aggression > 2.0 else "passive",
            "player_types": player_types,
            "recommendation": self.get_table_recommendation(avg_aggression, num_active)
        }
    
    def get_table_recommendation(self, avg_aggression: float, num_players: int) -> str:
        # Table recommendation
        if avg_aggression > 2.5:
            return "Very aggressive table - tighten up, trap with strong hands"
        elif avg_aggression > 1.5:
            return "Moderately aggressive - play solid TAG strategy"
        else:
            return "Passive table - be aggressive, steal pots"
