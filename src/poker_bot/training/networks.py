"""
Actor-Critic Neural Network Architectures for Poker RL Training

Based on:
- Actor-Critic architecture from modern RL research
- Partial information for actor (realistic game conditions)
- Full information for critic (optimal training signal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Policy network for action selection.

    Architecture:
    - Input: Game state features (50 dims)
    - Hidden: 256 -> 128 -> 64
    - Output heads:
      - Action probabilities (4: fold, call, check, raise)
      - Bet amount (continuous 0-1, scaled to pot)
    """

    def __init__(self, state_dim: int = 50, hidden_dims: Tuple[int, int, int] = (256, 128, 64)):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Action head: discrete action selection
        self.action_head = nn.Linear(hidden_dims[2], 4)  # fold, call, check, raise

        # Amount head: continuous bet sizing
        self.amount_head = nn.Linear(hidden_dims[2], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State features [batch_size, state_dim]

        Returns:
            action_probs: Action probabilities [batch_size, 4]
            amount: Bet amount ratio [batch_size, 1] (0-1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Action probabilities (softmax over fold/call/check/raise)
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Bet amount (sigmoid to 0-1 range)
        amount = torch.sigmoid(self.amount_head(x))

        return action_probs, amount

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """
        Sample action from policy.

        Args:
            state: State features
            deterministic: If True, take argmax action (for evaluation)

        Returns:
            action: Discrete action (0=fold, 1=call, 2=check, 3=raise)
            amount: Bet amount ratio (0-1)
        """
        with torch.no_grad():
            action_probs, amount = self.forward(state)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from categorical distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

            amount_value = amount.item()

        return action, amount_value


class CriticNetwork(nn.Module):
    """
    Value network for state evaluation.

    Architecture:
    - Input: Full game state (including opponent cards for training)
    - Hidden: 256 -> 128 -> 64
    - Output: State value estimate
    """

    def __init__(self, state_dim: int = 70, hidden_dims: Tuple[int, int, int] = (256, 128, 64)):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Value head
        self.value_head = nn.Linear(hidden_dims[2], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Full state features [batch_size, state_dim]

        Returns:
            value: State value estimate [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_head(x)

        return value


class ActorCriticAgent(nn.Module):
    """
    Combined Actor-Critic agent.

    Wraps actor and critic networks for easy training and inference.
    """

    def __init__(self, actor_state_dim: int = 50, critic_state_dim: int = 70):
        super().__init__()

        self.actor = ActorNetwork(state_dim=actor_state_dim)
        self.critic = CriticNetwork(state_dim=critic_state_dim)

    def forward(self, actor_state: torch.Tensor, critic_state: torch.Tensor):
        """
        Forward pass for both networks.

        Args:
            actor_state: Partial information state for actor
            critic_state: Full information state for critic

        Returns:
            action_probs, amount, value
        """
        action_probs, amount = self.actor(actor_state)
        value = self.critic(critic_state)

        return action_probs, amount, value

    def get_action_and_value(self, actor_state: torch.Tensor, critic_state: torch.Tensor,
                             deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action and value estimate.

        Args:
            actor_state: Partial information state
            critic_state: Full information state
            deterministic: If True, use argmax action

        Returns:
            action, amount, value
        """
        action, amount = self.actor.get_action(actor_state, deterministic)
        value = self.critic(critic_state).item()

        return action, amount, value

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


# Helper function to count parameters
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test network architectures
    print("Testing Actor-Critic Networks...")

    agent = ActorCriticAgent(actor_state_dim=50, critic_state_dim=70)

    print(f"Actor parameters: {count_parameters(agent.actor):,}")
    print(f"Critic parameters: {count_parameters(agent.critic):,}")
    print(f"Total parameters: {count_parameters(agent):,}")

    # Test forward pass
    batch_size = 32
    actor_state = torch.randn(batch_size, 50)
    critic_state = torch.randn(batch_size, 70)

    action_probs, amount, value = agent(actor_state, critic_state)

    print(f"\nOutput shapes:")
    print(f"  Action probs: {action_probs.shape}")
    print(f"  Amount: {amount.shape}")
    print(f"  Value: {value.shape}")

    # Test action sampling
    action, amount_val = agent.actor.get_action(actor_state[0:1])
    print(f"\nSampled action: {action}, amount: {amount_val:.3f}")

    print("\n✅ Networks initialized successfully!")
