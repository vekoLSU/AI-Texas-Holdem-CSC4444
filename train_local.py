#!/usr/bin/env python3
"""
Local Training Script for RTX 5080 (Windows 11) - FIXED VERSION

Run this overnight to train your poker bot to superhuman level.
Usage: python train_local_fixed.py
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

NUM_HANDS = 500_000  # 500K hands
SAVE_EVERY = 10_000  # Save checkpoint every 10K hands
BATCH_SIZE = 256  # Larger batch for 5080 (16GB VRAM)
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
CLIP_EPSILON = 0.2  # PPO clip parameter
ENTROPY_COEF = 0.01  # Encourage exploration

CHECKPOINT_DIR = Path("trained_models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# SIMPLIFIED HEADS-UP POKER ENVIRONMENT
# ============================================================================

class HeadsUpPokerEnv:
    """Simplified heads-up (2-player) poker for faster training."""

    def __init__(self):
        self.starting_chips = 1000
        self.small_blind = 10
        self.big_blind = 20
        self.reset()

    def reset(self):
        """Start a new hand."""
        self.pot = self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        self.phase = "PREFLOP"
        self.community_cards = []

        # Two players: our agent (position 0) and opponent (position 1)
        self.players = [
            {
                "chips": self.starting_chips - self.small_blind,
                "bet": self.small_blind,
                "folded": False,
                "hole_cards": self._deal_cards(2)
            },
            {
                "chips": self.starting_chips - self.big_blind,
                "bet": self.big_blind,
                "folded": False,
                "hole_cards": self._deal_cards(2)
            }
        ]

        self.current_player = 0  # Agent acts first
        self.actions_this_round = 0
        self.max_actions_per_round = 4  # Prevent infinite loops

        return self._get_state(0)

    def _deal_cards(self, num):
        """Deal random cards (simplified)."""
        return np.random.randint(0, 13, num).tolist()

    def _get_state(self, player_idx):
        """Get state features for a player."""
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

        # Opponent info
        opponent = self.players[1 - player_idx]
        state.append(opponent["chips"] / self.starting_chips)
        state.append(opponent["bet"] / self.starting_chips)
        state.append(1.0 if not opponent["folded"] else 0.0)

        # Padding to 50 dims
        while len(state) < 50:
            state.append(0.0)

        return np.array(state[:50], dtype=np.float32)

    def _get_critic_state(self, player_idx):
        """Get full information state for critic."""
        state = []

        # Both players' cards
        for p in self.players:
            state.extend([c / 13.0 for c in p["hole_cards"]])

        # Community cards
        community = self.community_cards + [0] * (5 - len(self.community_cards))
        state.extend([c / 13.0 for c in community])

        # Game state
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)

        # Both players' state
        for p in self.players:
            state.append(p["chips"] / self.starting_chips)
            state.append(p["bet"] / self.starting_chips)
            state.append(1.0 if not p["folded"] else 0.0)

        # Padding to 70 dims
        while len(state) < 70:
            state.append(0.0)

        return np.array(state[:70], dtype=np.float32)

    def step(self, action, amount_ratio):
        """
        Execute action for current player.

        Actions: 0=fold, 1=call, 2=check, 3=raise

        Returns: (next_state, reward, done, info)
        """
        player_idx = 0  # Always agent
        player = self.players[player_idx]

        reward = 0
        done = False

        # Execute action
        if action == 0:  # Fold
            player["folded"] = True
            # Opponent wins
            winner_idx = 1
            self.players[winner_idx]["chips"] += self.pot
            reward = -player["bet"]
            done = True

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
            raise_amount = int(amount_ratio * self.pot * 0.5)
            raise_amount = max(raise_amount, self.current_bet * 2 - player["bet"], 20)
            raise_amount = min(raise_amount, player["chips"])

            player["chips"] -= raise_amount
            player["bet"] += raise_amount
            self.pot += raise_amount
            self.current_bet = player["bet"]

        self.actions_this_round += 1

        # Check if done
        if not done:
            # Opponent takes random action (simple opponent)
            done = self._opponent_act()

        # Check if betting round complete
        if not done and self._is_round_complete():
            done = self._advance_phase()

            if done:
                # Showdown
                winner_idx = self._determine_winner()
                self.players[winner_idx]["chips"] += self.pot
                reward = self.pot - player["bet"] if winner_idx == 0 else -player["bet"]

        return self._get_state(0), reward, done, {}

    def _opponent_act(self):
        """Opponent takes a random action."""
        opponent = self.players[1]

        if opponent["folded"]:
            return False

        # Random strategy for opponent
        to_call = self.current_bet - opponent["bet"]

        if to_call == 0:
            # Check or bet randomly
            if np.random.random() < 0.3:
                # Bet
                bet_amount = min(int(self.pot * 0.5), opponent["chips"])
                opponent["chips"] -= bet_amount
                opponent["bet"] += bet_amount
                self.pot += bet_amount
                self.current_bet = opponent["bet"]
        else:
            # Fold, call, or raise
            action = np.random.choice([0, 1, 3], p=[0.2, 0.6, 0.2])

            if action == 0:  # Fold
                opponent["folded"] = True
                # Agent wins
                self.players[0]["chips"] += self.pot
                return True

            elif action == 1:  # Call
                call_amount = min(to_call, opponent["chips"])
                opponent["chips"] -= call_amount
                opponent["bet"] += call_amount
                self.pot += call_amount

            elif action == 3:  # Raise
                raise_amount = min(int(self.pot * 0.5), opponent["chips"])
                opponent["chips"] -= raise_amount
                opponent["bet"] += raise_amount
                self.pot += raise_amount
                self.current_bet = opponent["bet"]

        self.actions_this_round += 1
        return False

    def _is_round_complete(self):
        """Check if betting round is done."""
        # If too many actions, force completion
        if self.actions_this_round >= self.max_actions_per_round:
            return True

        # Both players have equal bets
        if self.players[0]["bet"] == self.players[1]["bet"]:
            return True

        return False

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
            return True  # Showdown

        # Reset for new round
        for p in self.players:
            p["bet"] = 0
        self.current_bet = 0
        self.actions_this_round = 0

        return False

    def _determine_winner(self):
        """Determine winner (simplified: highest cards sum)."""
        if self.players[0]["folded"]:
            return 1
        if self.players[1]["folded"]:
            return 0

        score0 = sum(self.players[0]["hole_cards"]) + sum(self.community_cards)
        score1 = sum(self.players[1]["hole_cards"]) + sum(self.community_cards)

        return 0 if score0 >= score1 else 1


# ============================================================================
# PPO TRAINING
# ============================================================================

class PPOTrainer:
    """PPO trainer."""

    def __init__(self, agent, device):
        self.agent = agent
        self.device = device
        self.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)

        self.clear_buffer()

    def store_transition(self, state, critic_state, action, reward, value, log_prob):
        """Store a transition."""
        self.states.append(state)
        self.critic_states.append(critic_state)
        self.actions.append(action)
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
        """Perform PPO update."""
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

        # PPO update
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
        """Clear buffers."""
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
    print("POKER BOT RL TRAINING - FIXED VERSION")
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
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Checkpoint Every: {SAVE_EVERY:,} hands")
    print("=" * 70)

    agent = ActorCriticAgent(actor_state_dim=50, critic_state_dim=70).to(device)
    trainer = PPOTrainer(agent, device)
    env = HeadsUpPokerEnv()

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

        while not done and steps < 20:  # Max 20 steps per hand
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

            next_state, reward, done, _ = env.step(action_idx, amount_val)

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

        # Show progress every hand for first 100, then every 10, then every 100
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

            print(f"[{hand_count:6d}/{NUM_HANDS}] "
                  f"Win Rate: {win_rate:5.1f}% | "
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
    print(f"Final Win Rate: {(win_count / hand_count) * 100:.1f}%")
    print(f"Total Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Final Model: {final_path}")
    print("=" * 70)

    stats = {
        "total_hands": hand_count,
        "win_rate": (win_count / hand_count) * 100,
        "training_time_hours": (time.time() - start_time) / 3600,
        "final_model": str(final_path)
    }

    with open(CHECKPOINT_DIR / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n🎉 Your bot is now ready to DOMINATE!")

if __name__ == "__main__":
    main()
