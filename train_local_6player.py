#!/usr/bin/env python3
"""
6-Player Poker Training with FULL SUIT ENCODING

This version trains with suit information so the bot understands flushes.
State dim: 70 for actor (was 50), 90 for critic (was 70)

Usage: python train_6player_with_suits.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

CHECKPOINT_DIR = Path("trained_models_suited")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# NEURAL NETWORKS WITH SUIT ENCODING
# ============================================================================

class ActorNetwork(nn.Module):
    """Actor network with suit-aware state (70 dims instead of 50)."""
    def __init__(self, state_dim=70):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.action_head = nn.Linear(64, 4)  # fold, call, check, raise
        self.amount_head = nn.Linear(64, 1)  # bet sizing
        
    def forward(self, x):
        features = self.network(x)
        action_probs = torch.softmax(self.action_head(features), dim=-1)
        amount = torch.sigmoid(self.amount_head(features))
        return action_probs, amount

class CriticNetwork(nn.Module):
    """Critic network with suit-aware full info (90 dims instead of 70)."""
    def __init__(self, state_dim=90):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class ActorCriticAgent(nn.Module):
    def __init__(self, actor_state_dim=70, critic_state_dim=90):
        super().__init__()
        self.actor = ActorNetwork(state_dim=actor_state_dim)
        self.critic = CriticNetwork(state_dim=critic_state_dim)
        
    def forward(self, actor_state, critic_state):
        action_probs, amount = self.actor(actor_state)
        value = self.critic(critic_state)
        return action_probs, amount, value
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

# ============================================================================
# POKER HAND EVALUATOR (with suits)
# ============================================================================

def evaluate_poker_hand(hole_cards, community_cards):
    """
    Evaluate poker hand with FULL suit support.
    
    Args:
        hole_cards: [(rank, suit), (rank, suit)]
        community_cards: [(rank, suit), ...]
    
    Returns: (hand_rank, tiebreakers)
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) != 7:
        return (0, [c[0] for c in hole_cards])
    
    ranks = [c[0] for c in all_cards]
    suits = [c[1] for c in all_cards]
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    # Check for flush
    is_flush = max(suit_counts.values()) >= 5
    flush_suit = None
    if is_flush:
        flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
        flush_cards = sorted([c[0] for c in all_cards if c[1] == flush_suit], reverse=True)[:5]
    
    # Check for straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                is_straight = True
                straight_high = unique_ranks[i]
                break
        # Check for A-2-3-4-5 wheel
        if set([12, 0, 1, 2, 3]).issubset(set(ranks)):
            is_straight = True
            straight_high = 3  # 5-high straight
    
    # Straight flush
    if is_flush and is_straight:
        # Check if straight cards are all same suit
        straight_cards = []
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                candidate = unique_ranks[i:i+5]
                if all(any(c[0] == r and c[1] == flush_suit for c in all_cards) for r in candidate):
                    return (8, [unique_ranks[i]])
        # Check wheel straight flush
        if set([12, 0, 1, 2, 3]).issubset(set(ranks)):
            wheel_cards = [12, 0, 1, 2, 3]
            if all(any(c[0] == r and c[1] == flush_suit for c in all_cards) for r in wheel_cards):
                return (8, [3])
    
    counts = sorted(rank_counts.values(), reverse=True)
    ranks_by_count = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Four of a kind
    if counts[0] == 4:
        quads = ranks_by_count[0][0]
        kicker = ranks_by_count[1][0]
        return (7, [quads, kicker])
    
    # Full house
    if counts[0] == 3 and counts[1] >= 2:
        trips = ranks_by_count[0][0]
        pair = ranks_by_count[1][0]
        return (6, [trips, pair])
    
    # Flush
    if is_flush:
        return (5, flush_cards)
    
    # Straight
    if is_straight:
        return (4, [straight_high])
    
    # Three of a kind
    if counts[0] == 3:
        trips = ranks_by_count[0][0]
        kickers = sorted([r for r, c in ranks_by_count[1:3]], reverse=True)
        return (3, [trips] + kickers)
    
    # Two pair
    if counts[0] == 2 and counts[1] == 2:
        pair1, pair2 = ranks_by_count[0][0], ranks_by_count[1][0]
        kicker = ranks_by_count[2][0]
        return (2, [max(pair1, pair2), min(pair1, pair2), kicker])
    
    # One pair
    if counts[0] == 2:
        pair = ranks_by_count[0][0]
        kickers = sorted([r for r, c in ranks_by_count[1:4]], reverse=True)
        return (1, [pair] + kickers)
    
    # High card
    return (0, sorted(unique_ranks, reverse=True)[:5])

def compare_hands(hand1_eval, hand2_eval):
    """Compare two evaluated hands."""
    rank1, tie1 = hand1_eval
    rank2, tie2 = hand2_eval
    
    if rank1 != rank2:
        return 1 if rank1 > rank2 else -1
    
    for t1, t2 in zip(tie1, tie2):
        if t1 != t2:
            return 1 if t1 > t2 else -1
    return 0

# ============================================================================
# 6-PLAYER POKER ENVIRONMENT (with suits)
# ============================================================================

class PokerEnvironment6Player:
    """6-player Texas Hold'em with FULL suit support."""
    
    def __init__(self):
        self.num_players = 6
        self.starting_chips = 1000
        self.small_blind = 10
        self.big_blind = 20
        self.reset()
    
    def reset(self):
        """Start new hand."""
        self.pot = 0
        self.current_bet = 0
        self.phase = "PREFLOP"
        self.community_cards = []
        
        # Deal cards with suits
        self.deck = [(r, s) for r in range(13) for s in range(4)]
        np.random.shuffle(self.deck)
        self.deck_idx = 0
        
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
        """Deal cards from deck."""
        cards = self.deck[self.deck_idx:self.deck_idx + num]
        self.deck_idx += num
        return cards
    
    def _get_state(self, player_idx):
        """
        Get suit-aware state for actor (70 dims).
        
        Encoding:
        - 2 hole cards: rank/13.0, suit/4.0 (4 dims)
        - 5 community cards: rank/13.0, suit/4.0 (10 dims)
        - Pot, bet info (same as before)
        - Position, phase encoding
        - Padding to 70
        """
        player = self.players[player_idx]
        state = []
        
        # Hole cards with suits (4 dims: rank1, suit1, rank2, suit2)
        for rank, suit in player["hole_cards"]:
            state.append(rank / 13.0)
            state.append(suit / 4.0)
        
        # Community cards with suits (10 dims: up to 5 cards)
        community = self.community_cards + [(0, 0)] * (5 - len(self.community_cards))
        for rank, suit in community:
            state.append(rank / 13.0)
            state.append(suit / 4.0)
        
        # Game state (same as before)
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)
        state.append(player["chips"] / self.starting_chips)
        state.append(player["bet"] / self.starting_chips)
        
        # Active players
        active = sum(1 for p in self.players if not p["folded"])
        state.append(active / self.num_players)
        
        # Position (one-hot encoding)
        for i in range(6):
            state.append(1.0 if i == player_idx else 0.0)
        
        # Phase encoding
        phase_map = {"PREFLOP": 0.2, "FLOP": 0.4, "TURN": 0.6, "RIVER": 0.8}
        state.append(phase_map.get(self.phase, 0.0))
        
        # Hand strength (precomputed for faster training)
        if len(self.community_cards) >= 3:
            hand_eval = evaluate_poker_hand(player["hole_cards"], self.community_cards)
            state.append(hand_eval[0] / 8.0)  # Normalize hand rank
        else:
            # Preflop: use high card value
            state.append(max(player["hole_cards"][0][0], player["hole_cards"][1][0]) / 13.0)
        
        # Suited indicator
        state.append(1.0 if player["hole_cards"][0][1] == player["hole_cards"][1][1] else 0.0)
        
        # Padding to 70
        while len(state) < 70:
            state.append(0.0)
        
        return np.array(state[:70], dtype=np.float32)
    
    def _get_critic_state(self, player_idx):
        """Get full info state for critic (90 dims)."""
        state = []
        
        # All players' hole cards (24 dims: 6 players * 4 dims each)
        for p in self.players:
            for rank, suit in p["hole_cards"]:
                state.append(rank / 13.0)
                state.append(suit / 4.0)
        
        # Community cards (10 dims)
        community = self.community_cards + [(0, 0)] * (5 - len(self.community_cards))
        for rank, suit in community:
            state.append(rank / 13.0)
            state.append(suit / 4.0)
        
        # Game state
        state.append(self.pot / self.starting_chips)
        state.append(self.current_bet / self.starting_chips)
        
        # All players' state (18 dims: 6 players * 3 dims each)
        for p in self.players:
            state.append(p["chips"] / self.starting_chips)
            state.append(p["bet"] / self.starting_chips)
            state.append(1.0 if not p["folded"] else 0.0)
        
        # Phase encoding
        phase_map = {"PREFLOP": 0.2, "FLOP": 0.4, "TURN": 0.6, "RIVER": 0.8}
        state.append(phase_map.get(self.phase, 0.0))
        
        # Padding to 90
        while len(state) < 90:
            state.append(0.0)
        
        return np.array(state[:90], dtype=np.float32)
    
    def step(self, player_idx, action, amount_ratio):
        """Execute action."""
        player = self.players[player_idx]
        reward = 0
        done = False
        info = {"won_pot": False}
        
        if player["folded"] or player["chips"] <= 0:
            reward = -player["bet"]
            done = True
            return self._get_state(player_idx), reward, done, info
        
        # Execute action
        if action == 0:  # Fold
            player["folded"] = True
            reward = -player["bet"]
            done = True
            return self._get_state(player_idx), reward, done, info
        
        elif action == 1:  # Call
            call_amount = max(0, min(self.current_bet - player["bet"], player["chips"]))
            player["chips"] -= call_amount
            player["bet"] += call_amount
            self.pot += call_amount
        
        elif action == 2:  # Check
            if player["bet"] < self.current_bet:
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
        
        # Simulate opponents
        self._simulate_other_players(player_idx)
        
        # Check for winner
        active_players = [p for p in self.players if not p["folded"] and (p["chips"] > 0 or p["bet"] > 0)]
        
        if len(active_players) == 1:
            winner_idx = next(i for i, p in enumerate(self.players) if not p["folded"])
            self.players[winner_idx]["chips"] += self.pot
            reward = self.pot if winner_idx == player_idx else -player["bet"]
            done = True
            info["won_pot"] = (winner_idx == player_idx)
            return self._get_state(player_idx), reward, done, info
        
        # Advance phase
        if self._is_betting_round_complete():
            showdown = self._advance_phase()
            if showdown:
                winner_idx = self._determine_winner()
                self.players[winner_idx]["chips"] += self.pot
                reward = self.pot if winner_idx == player_idx else -player["bet"]
                done = True
                info["won_pot"] = (winner_idx == player_idx)
        
        return self._get_state(player_idx), reward, done, info
    
    def _simulate_other_players(self, acting_player_idx):
        """Simulate 5 opponents with hand-strength based play."""
        num = self.num_players
        order = [(acting_player_idx + i) % num for i in range(1, num)]
        
        for idx in order:
            p = self.players[idx]
            if p["folded"] or p["chips"] <= 0:
                continue
            
            # Calculate actual hand strength with suits
            if len(self.community_cards) >= 3:
                hand_eval = evaluate_poker_hand(p["hole_cards"], self.community_cards)
                hand_strength = hand_eval[0] / 8.0  # Normalize to [0, 1]
            else:
                # Preflop: high card + suited bonus
                high_card = max(p["hole_cards"][0][0], p["hole_cards"][1][0])
                suited_bonus = 0.1 if p["hole_cards"][0][1] == p["hole_cards"][1][1] else 0
                hand_strength = (high_card / 13.0) + suited_bonus
            
            # Already matched bet
            if p["bet"] >= self.current_bet:
                r = np.random.rand()
                if r < 0.85:
                    continue  # check
                elif r < 0.98:
                    raise_amt = min(int(0.2 * max(self.pot, self.big_blind)), p["chips"])
                    p["chips"] -= raise_amt
                    p["bet"] += raise_amt
                    self.pot += raise_amt
                    if p["bet"] > self.current_bet:
                        self.current_bet = p["bet"]
                else:
                    p["folded"] = True
            else:
                # Facing bet - use hand strength
                r = np.random.rand()
                active_count = sum(1 for p2 in self.players if not p2["folded"])
                tightness = 1.0 - (active_count - 2) * 0.05
                call_prob = (0.20 + 0.65 * hand_strength) * tightness
                
                if r < call_prob:
                    call_amount = min(self.current_bet - p["bet"], p["chips"])
                    p["chips"] -= call_amount
                    p["bet"] += call_amount
                    self.pot += call_amount
                elif r < call_prob + 0.05:
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
            self.community_cards.extend(self._deal_cards(1))
        elif self.phase == "TURN":
            self.phase = "RIVER"
            self.community_cards.extend(self._deal_cards(1))
        elif self.phase == "RIVER":
            return True
        
        for p in self.players:
            p["bet"] = 0
        self.current_bet = 0
        return False
    
    def _determine_winner(self):
        """Determine winner with full suit-aware evaluation."""
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
# PPO TRAINER (same as before)
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
        
        self.clear_buffer()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
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
    print("6-PLAYER POKER TRAINING - WITH SUIT ENCODING")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    
    print(f"\nConfiguration:")
    print(f"  Hands: {NUM_HANDS:,}")
    print(f"  Players: 6")
    print(f"  Actor State: 70 dims (WITH SUITS)")
    print(f"  Critic State: 90 dims (WITH SUITS)")
    print(f"  Hand Eval: Full poker with flushes")
    print("=" * 70)
    
    agent = ActorCriticAgent(actor_state_dim=70, critic_state_dim=90).to(device)
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
        
        while not done and steps < 30:
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
            
            next_state, reward, done, info = env.step(0, action_idx, amount_val)
            
            trainer.store_transition(
                state, critic_state, action_idx,
                reward, value.item(), log_prob.item()
            )
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
        if info.get("won_pot", False):
            win_count += 1
        hand_count += 1
        
        if len(trainer.states) >= BATCH_SIZE:
            trainer.train_step()
        
        if hand_count % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            win_rate = (win_count / hand_count) * 100
            elapsed = time.time() - start_time
            hands_per_sec = hand_count / elapsed
            eta_hours = (NUM_HANDS - hand_count) / hands_per_sec / 3600
            
            print(f"[{hand_count:6d}/{NUM_HANDS}] "
                  f"Win Rate: {win_rate:5.1f}% (Baseline: 16.7%) | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Speed: {hands_per_sec:.0f} hands/s | "
                  f"ETA: {eta_hours:.1f}h")
        
        if hand_count % SAVE_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f"agent_suited_{hand_count}.pt"
            agent.save(str(checkpoint_path))
            print(f"💾 Checkpoint: {checkpoint_path}")
    
    final_path = CHECKPOINT_DIR / "agent_suited_final.pt"
    agent.save(str(final_path))
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total Hands: {hand_count:,}")
    final_wr = (win_count / hand_count) * 100
    print(f"Final Win Rate: {final_wr:.1f}%")
    print(f"Baseline: 16.67% | Performance: {final_wr/16.67:.2f}x")
    print(f"Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Model: {final_path}")
    print("=" * 70)
    
    stats = {
        "total_hands": hand_count,
        "win_rate": final_wr,
        "training_time_hours": (time.time() - start_time) / 3600,
        "model_path": str(final_path),
        "suit_aware": True
    }
    
    with open(CHECKPOINT_DIR / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()