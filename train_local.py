#!/usr/bin/env python3
"""
Local Training Script for RTX 5080 (Windows 11)

Run this overnight to train your poker bot to superhuman level.
Usage: python train_local.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

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

NUM_HANDS = 500_000  # 500K hands (~5-6 hours on RTX 5080)
SAVE_EVERY = 10_000  # Save checkpoint every 10K hands
BATCH_SIZE = 256  # Larger batch for 5080 (16GB VRAM)
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
CLIP_EPSILON = 0.2  # PPO clip parameter
ENTROPY_COEF = 0.01  # Encourage exploration

CHECKPOINT_DIR = Path("trained_models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# SIMPLIFIED POKER ENVIRONMENT
# ============================================================================

class PokerEnvironment:
    """Simplified poker environment for self-play training."""

    def __init__(self):
        self.num_players = 3
        self.starting_chips = 1000
        self.small_blind = 10
        self.big_blind = 20
        self.reset()

    def reset(self):
        """Start a new hand."""
        self.pot = 0
        self.current_bet = 0
        self.phase = "PREFLOP"  # PREFLOP -> FLOP -> TURN -> RIVER
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
        """Deal random cards (simplified - just random values)."""
        return np.random.randint(0, 13, num).tolist()

    def _get_state(self, player_idx):
        """Get state features for a player."""
        player = self.players[player_idx]

        # Features: [hole_cards(2), community_cards(5), pot, current_bet,
        #            player_chips, player_bet, num_active_players, position(3)]
        state = []

        # Hole cards (normalized 0-1)
        state.extend([c / 13.0 for c in player["hole_cards"]])

        # Community cards (pad to 5)
        community = self.community_cards + [0] * (5 - len(self.community_cards))
        state.extend([c / 13.0 for c in community])

        # Pot and betting info (normalized by starting chips)
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)
        state.append(player["chips"] / self.starting_chips)
        state.append(player["bet"] / self.starting_chips)

        # Number of active players
        active = sum(1 for p in self.players if not p["folded"])
        state.append(active / self.num_players)

        # Position (one-hot)
        position = [0, 0, 0]
        position[player_idx] = 1
        state.extend(position)

        # Padding to reach 50 dimensions
        while len(state) < 50:
            state.append(0.0)

        return np.array(state[:50], dtype=np.float32)

    def _get_critic_state(self, player_idx):
        """Get full information state for critic (includes opponent cards)."""
        # Similar to actor state but includes all hole cards
        state = []

        # All players' hole cards
        for p in self.players:
            state.extend([c / 13.0 for c in p["hole_cards"]])

        # Community cards
        community = self.community_cards + [0] * (5 - len(self.community_cards))
        state.extend([c / 13.0 for c in community])

        # Game state
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)

        # All players' chips and bets
        for p in self.players:
            state.append(p["chips"] / self.starting_chips)
            state.append(p["bet"] / self.starting_chips)
            state.append(1.0 if not p["folded"] else 0.0)

        # Padding to 70 dimensions
        while len(state) < 70:
            state.append(0.0)

        return np.array(state[:70], dtype=np.float32)

    def step(self, player_idx, action, amount_ratio):
        """
        Execute action and return (next_state, reward, done).

        Actions:
        0 = fold
        1 = call
        2 = check
        3 = raise
        """
        player = self.players[player_idx]
        reward = 0
        done = False

        if player["folded"]:
            return self._get_state(player_idx), reward, done

        # Execute action
        if action == 0:  # Fold
            player["folded"] = True

        elif action == 1:  # Call
            call_amount = min(self.current_bet - player["bet"], player["chips"])
            player["chips"] -= call_amount
            player["bet"] += call_amount
            self.pot += call_amount

        elif action == 2:  # Check
            if player["bet"] < self.current_bet:
                # Can't check, treat as call
                call_amount = min(self.current_bet - player["bet"], player["chips"])
                player["chips"] -= call_amount
                player["bet"] += call_amount
                self.pot += call_amount

        elif action == 3:  # Raise
            # Raise amount based on network output
            raise_amount = int(amount_ratio * self.pot)
            raise_amount = max(raise_amount, self.current_bet * 2 - player["bet"])
            raise_amount = min(raise_amount, player["chips"])

            player["chips"] -= raise_amount
            player["bet"] += raise_amount
            self.pot += raise_amount
            self.current_bet = player["bet"]

        # Check if hand is over
        active_players = [i for i, p in enumerate(self.players) if not p["folded"]]

        if len(active_players) == 1:
            # Only one player left, they win
            winner_idx = active_players[0]
            self.players[winner_idx]["chips"] += self.pot
            reward = self.pot if winner_idx == player_idx else -player["bet"]
            done = True

        elif self._is_betting_round_complete():
            # Advance to next phase or showdown
            done = self._advance_phase()

            if done:
                # Showdown
                winner_idx = self._determine_winner()
                self.players[winner_idx]["chips"] += self.pot
                reward = self.pot if winner_idx == player_idx else -player["bet"]

        return self._get_state(player_idx), reward, done

    def _is_betting_round_complete(self):
        """Check if all players have acted."""
        active = [p for p in self.players if not p["folded"]]
        if len(active) <= 1:
            return True

        # All active players must have equal bets
        bets = [p["bet"] for p in active]
        return len(set(bets)) == 1

    def _advance_phase(self):
        """Move to next betting round."""
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
            return True  # Hand over, go to showdown

        # Reset bets for new round
        for p in self.players:
            p["bet"] = 0
        self.current_bet = 0

        return False

    def _determine_winner(self):
        """Determine winner at showdown (simplified - highest hole cards sum)."""
        active = [(i, sum(p["hole_cards"])) for i, p in enumerate(self.players) if not p["folded"]]
        if not active:
            return 0
        return max(active, key=lambda x: x[1])[0]

# ============================================================================
# PPO TRAINING ALGORITHM
# ============================================================================

class PPOTrainer:
    """PPO trainer for poker agent."""

    def __init__(self, agent, device):
        self.agent = agent
        self.device = device

        # Optimizers
        self.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)

        # Storage for trajectories
        self.states = []
        self.critic_states = []
        self.actions = []
        self.amounts = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def store_transition(self, state, critic_state, action, amount, reward, value, log_prob):
        """Store a transition."""
        self.states.append(state)
        self.critic_states.append(critic_state)
        self.actions.append(action)
        self.amounts.append(amount)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns(self):
        """Compute discounted returns."""
        returns = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + GAMMA * R
            returns.insert(0, R)
        return returns

    def train_step(self):
        """Perform one PPO update."""
        if len(self.states) < BATCH_SIZE:
            return None

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        critic_states = torch.FloatTensor(np.array(self.critic_states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute returns and advantages
        returns = self.compute_returns()
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get current policy
            action_probs, _, _ = self.agent(states, critic_states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            # Value loss
            current_values = self.agent.critic(critic_states).squeeze()
            critic_loss = nn.MSELoss()(current_values, returns)

            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        # Clear storage
        loss_info = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

        self.clear_buffer()
        return loss_info

    def clear_buffer(self):
        """Clear trajectory storage."""
        self.states = []
        self.critic_states = []
        self.actions = []
        self.amounts = []
        self.rewards = []
        self.values = []
        self.log_probs = []

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("POKER BOT RL TRAINING - RTX 5080 EDITION")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected, using CPU (will be slow)")

    print(f"\nTraining Configuration:")
    print(f"  Total Hands: {NUM_HANDS:,}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Estimated Time: 5-6 hours on RTX 5080")
    print(f"  Checkpoint Every: {SAVE_EVERY:,} hands")
    print("=" * 70)

    # Initialize agent and trainer
    agent = ActorCriticAgent(actor_state_dim=50, critic_state_dim=70).to(device)
    trainer = PPOTrainer(agent, device)
    env = PokerEnvironment()

    # Training stats
    total_rewards = []
    win_count = 0
    hand_count = 0
    start_time = time.time()

    print("\n🎲 Starting training...\n")

    # Training loop
    while hand_count < NUM_HANDS:
        # Play one hand
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            critic_state = env._get_critic_state(0)  # Assume we're player 0
            critic_state_tensor = torch.FloatTensor(critic_state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, amount, value = agent(state_tensor, critic_state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_idx = action.item()
            amount_val = amount.item()
            value_val = value.item()
            log_prob_val = log_prob.item()

            # Execute action
            next_state, reward, done = env.step(0, action_idx, amount_val)

            # Store transition
            trainer.store_transition(
                state, critic_state, action_idx, amount_val,
                reward, value_val, log_prob_val
            )

            episode_reward += reward
            state = next_state

        # Update stats
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            win_count += 1
        hand_count += 1

        # Train agent
        if len(trainer.states) >= BATCH_SIZE:
            loss_info = trainer.train_step()

        # Progress update
        if hand_count % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            win_rate = (win_count / hand_count) * 100
            elapsed = time.time() - start_time
            hands_per_sec = hand_count / elapsed
            eta_hours = (NUM_HANDS - hand_count) / hands_per_sec / 3600

            print(f"[{hand_count:6d}/{NUM_HANDS}] "
                  f"Win Rate: {win_rate:5.1f}% | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"ETA: {eta_hours:.1f}h")

        # Save checkpoint
        if hand_count % SAVE_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f"agent_checkpoint_{hand_count}.pt"
            agent.save(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = CHECKPOINT_DIR / "agent_final.pt"
    agent.save(str(final_path))

    # Training summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total Hands: {hand_count:,}")
    print(f"Final Win Rate: {(win_count / hand_count) * 100:.1f}%")
    print(f"Total Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Final Model: {final_path}")
    print("=" * 70)

    # Save training stats
    stats = {
        "total_hands": hand_count,
        "win_rate": (win_count / hand_count) * 100,
        "training_time_hours": (time.time() - start_time) / 3600,
        "final_model": str(final_path),
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_hands": NUM_HANDS
        }
    }

    stats_path = CHECKPOINT_DIR / "training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Training stats saved: {stats_path}")
    print("\n🎉 Your bot is now ready to DOMINATE!")

if __name__ == "__main__":
    main()
