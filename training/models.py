from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PokerActor(nn.Module):
    """
    Policy network that outputs an action distribution and a normalized
    bet-size suggestion (0-1 range representing pot fraction).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.bet_size_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = self.dropout(F.relu(self.fc2(x)))
        action_logits = self.action_head(x)
        bet_fraction = torch.sigmoid(self.bet_size_head(x)).squeeze(-1)
        return action_logits, bet_fraction


class PokerCritic(nn.Module):
    """Value network that estimates expected reward for a given state."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value_head(x).squeeze(-1)

