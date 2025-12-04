from typing import Dict, List, Optional
from poker_bot.agents import GTOAgent, ExploiterAgent, DefenderAgent
from poker_bot.evaluation import OpponentTracker


class MetaController:
    """
    Meta-Controller: The Brain of the Ensemble System

    Purpose: Select which specialist agent to use based on:
    - Opponent type (detected via OpponentTracker)
    - Game situation (chip stacks, tournament phase)
    - Confidence in opponent classification

    Strategy Selection:
    - Agent A (GTO): vs strong/unknown opponents, minimize loss
    - Agent B (Exploiter): vs weak opponents, maximize profit
    - Agent C (Defender): vs aggressive opponents, minimize variance
    - Voting Ensemble: when uncertain (<75% confidence)

    Based on:
    - Pluribus architecture (blueprint + multiple strategies)
    - Libratus architecture (meta-strategy controller)
    - EnsembleCard research (rule + CFR + NFSP ensemble)
    """

    def __init__(self, opponent_tracker: OpponentTracker):
        """
        Initialize meta-controller with three specialist agents.

        Args:
            opponent_tracker: OpponentTracker instance for opponent classification
        """
        self.agent_gto = GTOAgent()
        self.agent_exploiter = ExploiterAgent()
        self.agent_defender = DefenderAgent()

        self.opponent_tracker = opponent_tracker

        # Agent selection stats (for learning which agents work best)
        self.agent_usage = {
            "gto": 0,
            "exploiter": 0,
            "defender": 0,
            "ensemble": 0
        }

        self.agent_results = {
            "gto": {"wins": 0, "losses": 0},
            "exploiter": {"wins": 0, "losses": 0},
            "defender": {"wins": 0, "losses": 0},
            "ensemble": {"wins": 0, "losses": 0}
        }

        # Confidence threshold for agent selection
        self.min_confidence = 0.75  # Use ensemble if below this

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
               current_bet: int,
               our_player_id: str) -> Dict:
        """
        Meta-decision: Select optimal agent and get decision.

        Args:
            All standard game state parameters
            our_player_id: Our player ID (to exclude from opponent analysis)

        Returns:
            Decision dict with 'action', optional 'amount', and metadata
        """

        # 1. Assess game situation
        stack_situation = self._assess_stack_situation(our_chips, opponent_profiles)

        # 2. Classify primary opponent (if heads-up or focused on one player)
        primary_opponent, opp_type, confidence = self._identify_primary_opponent(
            opponent_profiles, our_player_id
        )

        # 3. Analyze table dynamics
        avg_aggression, avg_vpip = self._analyze_table_dynamics(opponent_profiles)

        # 4. Select agent based on situation
        selected_agent, agent_name = self._select_agent(
            opp_type, confidence, stack_situation,
            avg_aggression, avg_vpip, num_players
        )

        # 5. Get decision from selected agent (or ensemble)
        if agent_name == "ensemble":
            decision = self._ensemble_vote(
                hand_cards, community_cards, hand_strength, phase,
                pot, to_call, our_chips, position, num_players,
                opponent_profiles, current_bet, confidence
            )
        else:
            decision = selected_agent.decide(
                hand_cards, community_cards, hand_strength, phase,
                pot, to_call, our_chips, position, num_players,
                opponent_profiles, current_bet
            )

        # 6. Add metadata for logging/analysis
        decision["meta"] = {
            "agent": agent_name,
            "opponent_type": opp_type,
            "confidence": confidence,
            "stack_situation": stack_situation
        }

        # 7. Track agent usage
        self.agent_usage[agent_name] += 1

        return decision

    def _select_agent(self, opp_type: str, confidence: float,
                      stack_situation: str, avg_aggression: float,
                      avg_vpip: float, num_players: int) -> tuple:
        """
        Select optimal agent based on situation.

        Returns:
            (agent_object, agent_name)
        """

        # LOW CONFIDENCE: Use ensemble
        if confidence < self.min_confidence:
            return None, "ensemble"

        # CHIP STACK OVERRIDES
        # Protect chip lead with defensive play
        if stack_situation == "chip_leader":
            return self.agent_defender, "defender"

        # Urgency when short-stacked - exploit aggressively
        if stack_situation == "short_stack":
            return self.agent_exploiter, "exploiter"

        # OPPONENT TYPE BASED SELECTION
        # Exploit weak opponents (Agent B)
        if opp_type in ["random", "weak", "fish", "rock", "nit", "rule_based", "hand_strength_only"]:
            return self.agent_exploiter, "exploiter"

        # Defend vs aggressive (Agent C)
        if opp_type in ["aggressive", "LAG", "maniac"]:
            return self.agent_defender, "defender"

        # GTO baseline vs strong/unknown (Agent A)
        if opp_type in ["TAG", "gto", "cfr", "strong_rl", "unknown", "hybrid"]:
            return self.agent_gto, "gto"

        # TABLE DYNAMICS OVERRIDE
        # Very aggressive table -> defensive
        if avg_aggression > 2.5:
            return self.agent_defender, "defender"

        # Very loose table -> exploit
        if avg_vpip > 0.45:
            return self.agent_exploiter, "exploiter"

        # Default: GTO baseline
        return self.agent_gto, "gto"

    def _ensemble_vote(self,
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
                       current_bet: int,
                       confidence: float) -> Dict:
        """
        Voting ensemble when uncertain about opponent type.

        All three agents vote, and we combine their recommendations.
        """

        # Get votes from all three agents
        vote_gto = self.agent_gto.decide(
            hand_cards, community_cards, hand_strength, phase,
            pot, to_call, our_chips, position, num_players,
            opponent_profiles, current_bet
        )

        vote_exploiter = self.agent_exploiter.decide(
            hand_cards, community_cards, hand_strength, phase,
            pot, to_call, our_chips, position, num_players,
            opponent_profiles, current_bet
        )

        vote_defender = self.agent_defender.decide(
            hand_cards, community_cards, hand_strength, phase,
            pot, to_call, our_chips, position, num_players,
            opponent_profiles, current_bet
        )

        # Weight votes by agent reliability
        weights = {
            "gto": 0.4,       # Always reliable
            "exploiter": 0.4,  # High upside
            "defender": 0.2    # Conservative fallback
        }

        # Voting methods:
        # 1. Action voting (fold/call/raise)
        # 2. Amount averaging (bet sizing)

        actions = [
            vote_gto.get("action"),
            vote_exploiter.get("action"),
            vote_defender.get("action")
        ]

        # Count votes for each action
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Select action with most votes
        final_action = max(action_counts, key=action_counts.get)

        # If raise, average the bet amounts with weights
        if final_action == "raise":
            amounts = [
                vote_gto.get("amount", 0),
                vote_exploiter.get("amount", 0),
                vote_defender.get("amount", 0)
            ]
            weighted_amounts = [
                amounts[0] * weights["gto"],
                amounts[1] * weights["exploiter"],
                amounts[2] * weights["defender"]
            ]
            final_amount = sum(weighted_amounts)
        else:
            final_amount = 0

        return {
            "action": final_action,
            "amount": int(final_amount),
            "confidence": confidence,
            "method": "ensemble_vote",
            "votes": {
                "gto": vote_gto.get("action"),
                "exploiter": vote_exploiter.get("action"),
                "defender": vote_defender.get("action")
            }
        }

    def _identify_primary_opponent(self, opponent_profiles: List[Dict],
                                    our_player_id: str) -> tuple:
        """
        Identify primary opponent to focus exploitation on.

        Returns:
            (opponent_id, opponent_type, confidence)
        """

        if not opponent_profiles:
            return None, "unknown", 0.0

        # If heads-up, easy choice
        if len(opponent_profiles) == 1:
            opp = opponent_profiles[0]
            return (
                opp.get("id"),
                opp.get("player_type", "unknown"),
                self._get_classification_confidence(opp)
            )

        # Multi-way: focus on weakest opponent (most exploitable)
        exploitability_scores = []
        for opp in opponent_profiles:
            if opp.get("id") != our_player_id:
                score = self._calculate_exploitability(opp)
                exploitability_scores.append((opp, score))

        # Sort by exploitability (highest first)
        exploitability_scores.sort(key=lambda x: x[1], reverse=True)

        if exploitability_scores:
            primary_opp = exploitability_scores[0][0]
            return (
                primary_opp.get("id"),
                primary_opp.get("player_type", "unknown"),
                self._get_classification_confidence(primary_opp)
            )

        return None, "unknown", 0.0

    def _calculate_exploitability(self, opponent_profile: Dict) -> float:
        """
        Calculate how exploitable an opponent is.

        Higher score = more exploitable.
        """
        player_type = opponent_profile.get("player_type", "unknown")

        # Exploitability scores by type
        exploitability = {
            "random": 1.0,
            "fish": 0.9,
            "rock": 0.85,
            "nit": 0.85,
            "rule_based": 0.8,
            "hand_strength_only": 0.9,
            "weak": 0.9,
            "maniac": 0.7,
            "LAG": 0.6,
            "TAG": 0.3,
            "gto": 0.1,
            "cfr": 0.1,
            "strong_rl": 0.2,
            "unknown": 0.5,
            "hybrid": 0.4
        }

        return exploitability.get(player_type, 0.5)

    def _get_classification_confidence(self, opponent_profile: Dict) -> float:
        """
        Get confidence in opponent classification.

        Based on hands played and consistency.
        """
        # Would come from opponent_tracker
        # For now, use placeholder
        return opponent_profile.get("confidence", 0.5)

    def _assess_stack_situation(self, our_chips: int,
                                 opponent_profiles: List[Dict]) -> str:
        """
        Assess our chip stack situation.

        Returns: "chip_leader", "comfortable", "short_stack", "desperate"
        """

        if not opponent_profiles:
            if our_chips > 700:
                return "comfortable"
            elif our_chips > 400:
                return "moderate"
            else:
                return "short_stack"

        # Calculate average opponent stack
        opponent_chips = [p.get("chips", 1000) for p in opponent_profiles]
        avg_opponent_chips = sum(opponent_chips) / len(opponent_chips)
        max_opponent_chips = max(opponent_chips)

        # Determine our situation
        if our_chips > max_opponent_chips * 1.5:
            return "chip_leader"
        elif our_chips > avg_opponent_chips * 1.2:
            return "comfortable"
        elif our_chips > 400:
            return "moderate"
        elif our_chips > 200:
            return "short_stack"
        else:
            return "desperate"

    def _analyze_table_dynamics(self, opponent_profiles: List[Dict]) -> tuple:
        """
        Analyze overall table dynamics.

        Returns:
            (avg_aggression, avg_vpip)
        """

        if not opponent_profiles:
            return 1.0, 0.25  # Default

        total_aggression = 0
        total_vpip = 0
        count = len(opponent_profiles)

        for profile in opponent_profiles:
            total_aggression += profile.get("aggression_factor", 1.0)
            total_vpip += profile.get("vpip", 0.25)

        return (
            total_aggression / count,
            total_vpip / count
        )

    def record_hand_result(self, agent_name: str, won: bool):
        """Record result of hand for agent performance tracking."""
        if agent_name in self.agent_results:
            if won:
                self.agent_results[agent_name]["wins"] += 1
            else:
                self.agent_results[agent_name]["losses"] += 1

    def get_agent_stats(self) -> Dict:
        """Get statistics on agent usage and performance."""
        stats = {
            "usage": self.agent_usage,
            "results": self.agent_results,
            "win_rates": {}
        }

        for agent in ["gto", "exploiter", "defender", "ensemble"]:
            wins = self.agent_results[agent]["wins"]
            losses = self.agent_results[agent]["losses"]
            total = wins + losses
            if total > 0:
                stats["win_rates"][agent] = wins / total
            else:
                stats["win_rates"][agent] = 0.0

        return stats
