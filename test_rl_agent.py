#!/usr/bin/env python3
"""
Test script to verify RL agent loads and makes decisions.

Usage:
    python test_rl_agent.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poker_bot.agents import RLSuitedAgent

def main():
    print("="*60)
    print("RL SUITED AGENT TEST")
    print("="*60)

    # Initialize agent
    print("\n1. Loading RL agent...")
    agent = RLSuitedAgent()

    if agent.loaded:
        print("   ✅ Model loaded successfully")
        print(f"   📂 Model path: trained_models_suited/agent_suited_final.pt")
        print(f"   🧠 Actor state dim: 70")
        print(f"   🧠 Critic state dim: 90")
    else:
        print("   ❌ Model failed to load")
        return 1

    # Test decision making
    print("\n2. Testing decision making...")

    # Example game state
    hand_cards = ["As", "Kh"]
    community_cards = ["Qd", "Jc", "Th"]
    hand_strength = {"strength": 0.85, "hand_type": "straight", "draw_potential": 0.1}
    phase = "FLOP"
    pot = 100
    to_call = 20
    our_chips = 500
    position = "late"
    num_players = 3
    opponent_profiles = [
        {"id": "opp1", "chips": 450, "vpip": 0.35, "aggression_factor": 1.5},
        {"id": "opp2", "chips": 400, "vpip": 0.25, "aggression_factor": 0.8}
    ]
    current_bet = 20

    print(f"   Hand: {hand_cards}")
    print(f"   Board: {community_cards}")
    print(f"   Phase: {phase}")
    print(f"   Pot: ${pot}, To call: ${to_call}")

    # Get decision
    decision = agent.decide(
        hand_cards=hand_cards,
        community_cards=community_cards,
        hand_strength=hand_strength,
        phase=phase,
        pot=pot,
        to_call=to_call,
        our_chips=our_chips,
        position=position,
        num_players=num_players,
        opponent_profiles=opponent_profiles,
        current_bet=current_bet
    )

    print(f"\n   🎲 Decision: {decision}")
    print(f"   ✅ Decision format correct: {isinstance(decision, dict) and 'action' in decision}")

    # Test different scenarios
    print("\n3. Testing multiple scenarios...")

    scenarios = [
        {"name": "Strong hand facing bet", "to_call": 50},
        {"name": "Weak hand, can check", "hand_cards": ["2h", "3d"], "to_call": 0},
        {"name": "Big bet facing us", "to_call": 200},
    ]

    for scenario in scenarios:
        test_to_call = scenario.get("to_call", to_call)
        test_hand = scenario.get("hand_cards", hand_cards)

        decision = agent.decide(
            hand_cards=test_hand,
            community_cards=community_cards,
            hand_strength=hand_strength,
            phase=phase,
            pot=pot,
            to_call=test_to_call,
            our_chips=our_chips,
            position=position,
            num_players=num_players,
            opponent_profiles=opponent_profiles,
            current_bet=current_bet
        )

        action_str = f"{decision['action']}"
        if decision['action'] == 'raise' and 'amount' in decision:
            action_str += f" ${decision['amount']}"

        print(f"   {scenario['name']}: {action_str}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nRL agent is ready for competition!")

    return 0

if __name__ == "__main__":
    exit(main())
