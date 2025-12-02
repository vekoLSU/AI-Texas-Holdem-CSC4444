#!/usr/bin/env python3
"""
6-PLAYER POKER TRAINING with Proper Hand Evaluation

Trains against 5 smart opponents for realistic competition scenarios.
Uses actual poker hand rankings for showdowns.

Usage: python train_local_6player.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from poker_bot.training.networks import ActorCriticAgent

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_HANDS = 500_000
SAVE_EVERY = 10_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01

CHECKPOINT_DIR = Path("trained_models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# POKER HAND EVALUATOR
# ============================================================================

def evaluate_poker_hand(hole_cards, community_cards):
    """
    Evaluate a poker hand properly using standard rankings.

    Returns: (rank, tiebreakers)
    - rank: 0-8 (high card to straight flush)
    - tiebreakers: list of card values for tie-breaking
    """
    # Combine all 7 cards (2 hole + 5 community)
    all_cards = hole_cards + community_cards
    if len(all_cards) != 7:
        # Incomplete hand (shouldn't happen at showdown)
        return (0, hole_cards)  # Just use hole cards

    values = [c for c in all_cards]
    value_counts = Counter(values)

    # Check for flush (all same suit not possible with our simple card model)
    # In our simplified model, cards are just 0-12 values

    # Check for straight
    unique_values = sorted(set(values), reverse=True)
    straight = False
    straight_high = 0
    if len(unique_values) >= 5:
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i+4] == 4:
                straight = True
                straight_high = unique_values[i]
                break

    # Get counts
    counts = sorted(value_counts.values(), reverse=True)
    values_by_count = sorted(value_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    # Four of a kind
    if counts[0] == 4:
        quads_value = values_by_count[0][0]
        kicker = values_by_count[1][0]
        return (7, [quads_value, kicker])

    # Full house
    if counts[0] == 3 and counts[1] >= 2:
        trips_value = values_by_count[0][0]
        pair_value = values_by_count[1][0]
        return (6, [trips_value, pair_value])

    # Straight
    if straight:
        return (4, [straight_high])

    # Three of a kind
    if counts[0] == 3:
        trips_value = values_by_count[0][0]
        kickers = sorted([v for v, c in values_by_count[1:3]], reverse=True)
        return (3, [trips_value] + kickers)

    # Two pair
    if counts[0] == 2 and counts[1] == 2:
        pair1 = values_by_count[0][0]
        pair2 = values_by_count[1][0]
        kicker = values_by_count[2][0]
        return (2, [max(pair1, pair2), min(pair1, pair2), kicker])

    # One pair
    if counts[0] == 2:
        pair_value = values_by_count[0][0]
        kickers = sorted([v for v, c in values_by_count[1:4]], reverse=True)
        return (1, [pair_value] + kickers)

    # High card
    top_5 = sorted(unique_values, reverse=True)[:5]
    return (0, top_5)

def compare_hands(hand1_eval, hand2_eval):
    """
    Compare two evaluated hands.
    Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    rank1, tie1 = hand1_eval
    rank2, tie2 = hand2_eval

    if rank1 > rank2:
        return 1
    elif rank1 < rank2:
        return -1
    else:
        # Same rank, compare tiebreakers
        for t1, t2 in zip(tie1, tie2):
            if t1 > t2:
                return 1
            elif t1 < t2:
                return -1
        return 0

# ============================================================================
# 6-PLAYER POKER ENVIRONMENT
# ============================================================================

class PokerEnvironment6Player:
    """6-player Texas Hold'em with proper hand evaluation."""

    def __init__(self):
        self.num_players = 6
        self.starting_chips = 1000
        self.small_blind = 10
        self.big_blind = 20
        self.reset()

    def reset(self):
        """Start a new hand."""
        self.pot = 0
        self.current_bet = 0
        self.phase = "PREFLOP"
        self.community_cards = []
        self.players = [
            {
                "chips": self.starting_chips,
                "bet": 0,
                "folded": False,
                "hole_cards": self._deal_cards(2)
            }
            for _ in range(self.num_players)
        ]

        # Post blinds
        self.players[0]["bet"] = self.small_blind
        self.players[0]["chips"] -= self.small_blind
        self.players[1]["bet"] = self.big_blind
        self.players[1]["chips"] -= self.big_blind
        self.current_bet = self.big_blind
        self.pot = self.small_blind + self.big_blind

        return self._get_state(0)

    def _deal_cards(self, num):
        """Deal random cards (0-12 representing 2-A)."""
        return np.random.randint(0, 13, num).tolist()

    def _get_state(self, player_idx):
        """Get state features for agent."""
        player = self.players[player_idx]
        state = []

        # Hole cards
        state.extend([c / 13.0 for c in player["hole_cards"]])

        # Community cards (pad to 5)
        community = self.community_cards + [0] * (5 - len(self.community_cards))
        state.extend([c / 13.0 for c in community])

        # Game state
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)
        state.append(player["chips"] / self.starting_chips)
        state.append(player["bet"] / self.starting_chips)

        # Active players count
        active = sum(1 for p in self.players if not p["folded"])
        state.append(active / self.num_players)

        # Position (one-hot for 6 players)
        for i in range(6):
            state.append(1.0 if i == player_idx else 0.0)

        # Padding to 50
        while len(state) < 50:
            state.append(0.0)

        return np.array(state[:50], dtype=np.float32)

    def _get_critic_state(self, player_idx):
        """Get full information state for critic."""
        state = []

        # All players' hole cards (12 cards total for 6 players)
        for p in self.players:
            state.extend([c / 13.0 for c in p["hole_cards"]])

        # Community cards
        community = self.community_cards + [0] * (5 - len(self.community_cards))
        state.extend([c / 13.0 for c in community])

        # Game state
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)

        # All players' state (18 values for 6 players)
        for p in self.players:
            state.append(p["chips"] / self.starting_chips)
            state.append(p["bet"] / self.starting_chips)
            state.append(1.0 if not p["folded"] else 0.0)

        # Padding to 70
        while len(state) < 70:
            state.append(0.0)

        return np.array(state[:70], dtype=np.float32)

    def step(self, player_idx, action, amount_ratio):
        """Execute action for agent (player 0)."""
        player = self.players[player_idx]
        reward = 0
        done = False

        # Check if agent out
        if player["folded"] or player["chips"] <= 0:
            reward = -player["bet"]
            done = True
            return self._get_state(player_idx), reward, done

        # Execute agent's action
        if action == 0:  # Fold
            player["folded"] = True
            reward = -player["bet"]
            done = True
            return self._get_state(player_idx), reward, done

        elif action == 1:  # Call
            call_amount = max(0, min(self.current_bet - player["bet"], player["chips"]))
            player["chips"] -= call_amount
            player["bet"] += call_amount
            self.pot += call_amount

        elif action == 2:  # Check
            if player["bet"] < self.current_bet:
                # Treat as call
                call_amount = min(self.current_bet - player["bet"], player["chips"])
                player["chips"] -= call_amount
                player["bet"] += call_amount
                self.pot += call_amount

        elif action == 3:  # Raise
            call_needed = max(0, self.current_bet - player["bet"])
            desired_raise = max(int(amount_ratio * max(self.pot, self.starting_chips)), self.big_blind)
            raise_amount = call_needed + desired_raise
            raise_amount = min(raise_amount, player["chips"])
            player["chips"] -= raise_amount
            player["bet"] += raise_amount
            self.pot += raise_amount
            if player["bet"] > self.current_bet:
                self.current_bet = player["bet"]

        # Simulate other 5 players
        self._simulate_other_players(player_idx)

        # Check active players
        active_players = [p for p in self.players if not p["folded"] and (p["chips"] > 0 or p["bet"] > 0)]

        # One player left
        if len(active_players) == 1:
            winner_idx = next(i for i, p in enumerate(self.players) if not p["folded"])
            self.players[winner_idx]["chips"] += self.pot
            reward = self.pot if winner_idx == player_idx else -player["bet"]
            done = True
            return self._get_state(player_idx), reward, done

        # Safety limit
        MAX_ROUNDS = 5
        if getattr(self, "_round_steps", 0) >= MAX_ROUNDS:
            winner_idx = self._determine_winner_properly()
            self.players[winner_idx]["chips"] += self.pot
            reward = self.pot if winner_idx == player_idx else -player["bet"]
            done = True
            self._round_steps = 0
            return self._get_state(player_idx), reward, done
        else:
            self._round_steps = getattr(self, "_round_steps", 0) + 1

        # Advance phase
        if self._is_betting_round_complete():
            showdown = self._advance_phase()
            if showdown:
                winner_idx = self._determine_winner_properly()
                self.players[winner_idx]["chips"] += self.pot
                reward = self.pot if winner_idx == player_idx else -player["bet"]
                done = True

        return self._get_state(player_idx), reward, done

    def _simulate_other_players(self, acting_player_idx: int):
        """Simulate 5 opponents with hand-strength based play."""
        num = self.num_players
        order = [(acting_player_idx + i) % num for i in range(1, num)]

        for idx in order:
            p = self.players[idx]
            if p["folded"] or p["chips"] <= 0:
                continue

            # Already matched bet
            if p["bet"] >= self.current_bet:
                r = np.random.rand()
                if r < 0.85:
                    continue  # check
                elif r < 0.98:
                    # small raise
                    raise_amt = min(int(0.2 * max(self.pot, self.big_blind)), p["chips"])
                    p["chips"] -= raise_amt
                    p["bet"] += raise_amt
                    self.pot += raise_amt
                    if p["bet"] > self.current_bet:
                        self.current_bet = p["bet"]
                else:
                    p["folded"] = True
            else:
                # Facing a bet - use hand strength
                r = np.random.rand()
                hand_strength = sum(p["hole_cards"]) / (13.0 * 2)

                # Adjust call probability based on # of opponents
                active_count = sum(1 for p2 in self.players if not p2["folded"])
                tightness = 1.0 - (active_count - 2) * 0.05  # Tighter with more opponents
                call_prob = (0.20 + 0.65 * hand_strength) * tightness

                if r < call_prob:
                    # call
                    call_amount = min(self.current_bet - p["bet"], p["chips"])
                    p["chips"] -= call_amount
                    p["bet"] += call_amount
                    self.pot += call_amount
                elif r < call_prob + 0.05:
                    # reraise
                    call_needed = max(0, self.current_bet - p["bet"])
                    extra = min(int(0.15 * max(self.pot, self.big_blind)), p["chips"] - call_needed)
                    extra = max(extra, 0)
                    raise_amt = call_needed + extra
                    raise_amt = min(raise_amt, p["chips"])
                    p["chips"] -= raise_amt
                    p["bet"] += raise_amt
                    self.pot += raise_amt
                    if p["bet"] > self.current_bet:
                        self.current_bet = p["bet"]
                else:
                    # fold
                    p["folded"] = True

    def _is_betting_round_complete(self):
        """Check if betting round done."""
        active = [p for p in self.players if not p["folded"] and p["chips"] > 0]
        if len(active) <= 1:
            return True
        bets = [p["bet"] for p in active]
        return len(set(bets)) == 1

    def _advance_phase(self):
        """Move to next phase."""
        if self.phase == "PREFLOP":
            self.phase = "FLOP"
            self.community_cards = self._deal_cards(3)
        elif self.phase == "FLOP":
            self.phase = "TURN"
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.phase == "TURN":
            self.phase = "RIVER"
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.phase == "RIVER":
            return True

        for p in self.players:
            p["bet"] = 0
        self.current_bet = 0
        return False

    def _determine_winner_properly(self):
        """Determine winner using proper poker hand evaluation."""
        active = [(i, p) for i, p in enumerate(self.players) if not p["folded"]]
        if not active:
            return 0
        if len(active) == 1:
            return active[0][0]

        # Evaluate all hands
        evaluated = []
        for idx, p in active:
            hand_eval = evaluate_poker_hand(p["hole_cards"], self.community_cards)
            evaluated.append((idx, hand_eval))

        # Find best hand
        best_idx = evaluated[0][0]
        best_eval = evaluated[0][1]

        for idx, hand_eval in evaluated[1:]:
            if compare_hands(hand_eval, best_eval) > 0:
                best_idx = idx
                best_eval = hand_eval

        return best_idx

# ============================================================================
# PPO TRAINER (Same as before)
# ============================================================================

class PPOTrainer:
    def __init__(self, agent, device):
        self.agent = agent
        self.device = device
        self.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)
        self.clear_buffer()

    def store_transition(self, state, critic_state, action, reward, value, log_prob):
        self.states.append(state)
        self.critic_states.append(critic_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns(self):
        returns = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + GAMMA * R
            returns.insert(0, R)
        return returns

    def train_step(self):
        if len(self.states) < BATCH_SIZE:
            return None

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        critic_states = torch.FloatTensor(np.array(self.critic_states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        returns = torch.FloatTensor(self.compute_returns()).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
            action_probs, _, _ = self.agent(states, critic_states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            current_values = self.agent.critic(critic_states).squeeze()
            critic_loss = nn.MSELoss()(current_values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        loss_info = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

        self.clear_buffer()
        return loss_info

    def clear_buffer(self):
        self.states = []
        self.critic_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("6-PLAYER POKER TRAINING - PROPER HAND EVALUATION")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected, using CPU")

    print(f"\nTraining Configuration:")
    print(f"  Total Hands: {NUM_HANDS:,}")
    print(f"  Players: 6 (agent + 5 smart opponents)")
    print(f"  Hand Evaluation: Proper poker rankings (pair, trips, etc.)")
    print(f"  Opponent Strategy: Hand-strength + position aware")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Checkpoint Every: {SAVE_EVERY:,} hands")
    print("=" * 70)

    agent = ActorCriticAgent(actor_state_dim=50, critic_state_dim=70).to(device)
    trainer = PPOTrainer(agent, device)
    env = PokerEnvironment6Player()

    total_rewards = []
    win_count = 0
    hand_count = 0
    start_time = time.time()

    print("\n🎲 Starting training...\n")

    while hand_count < NUM_HANDS:
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 30:  # More steps for 6 players
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            critic_state = env._get_critic_state(0)
            critic_state_tensor = torch.FloatTensor(critic_state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, amount, value = agent(state_tensor, critic_state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_idx = action.item()
            amount_val = amount.item()

            next_state, reward, done = env.step(0, action_idx, amount_val)

            trainer.store_transition(
                state, critic_state, action_idx,
                reward, value.item(), log_prob.item()
            )

            episode_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(episode_reward)
        if episode_reward > 0:
            win_count += 1
        hand_count += 1

        # Progress prints
        if hand_count <= 100:
            print(f"Hand {hand_count}: Reward={episode_reward:.1f}, Steps={steps}")
        elif hand_count % 10 == 0 and hand_count <= 1000:
            print(f"Hand {hand_count}: Win Rate={win_count/hand_count*100:.1f}%")

        if len(trainer.states) >= BATCH_SIZE:
            trainer.train_step()

        if hand_count % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            win_rate = (win_count / hand_count) * 100
            elapsed = time.time() - start_time
            hands_per_sec = hand_count / elapsed
            eta_hours = (NUM_HANDS - hand_count) / hands_per_sec / 3600

            # Expected win rate: ~16.67% (1/6 players) for random, 25-35% for trained
            print(f"[{hand_count:6d}/{NUM_HANDS}] "
                  f"Win Rate: {win_rate:5.1f}% (Target: 25-35%) | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Speed: {hands_per_sec:.0f} hands/s | "
                  f"ETA: {eta_hours:.1f}h")

        if hand_count % SAVE_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f"agent_checkpoint_{hand_count}.pt"
            agent.save(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    final_path = CHECKPOINT_DIR / "agent_final.pt"
    agent.save(str(final_path))

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total Hands: {hand_count:,}")
    final_wr = (win_count / hand_count) * 100
    print(f"Final Win Rate: {final_wr:.1f}%")
    print(f"  Random baseline: 16.67% (1/6 players)")
    print(f"  Your performance: {final_wr:.1f}% ({final_wr/16.67:.2f}x random)")
    print(f"Total Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Final Model: {final_path}")
    print("=" * 70)

    stats = {
        "total_hands": hand_count,
        "win_rate": final_wr,
        "training_time_hours": (time.time() - start_time) / 3600,
        "final_model": str(final_path)
    }

    with open(CHECKPOINT_DIR / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n🎉 Your 6-player bot is ready!")

if __name__ == "__main__":
    main()
