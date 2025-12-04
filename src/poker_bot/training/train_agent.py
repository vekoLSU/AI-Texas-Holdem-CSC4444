from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import ACTION_TO_INDEX, DecisionDataset
from .models import PokerActor, PokerCritic
from .state_encoder import StateEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train poker decision models.")
    parser.add_argument("--log-path", default="logs/training_decisions.jsonl", help="Path to decision log file.")
    parser.add_argument("--model-dir", default="models", help="Directory to store trained weights.")
    parser.add_argument("--agent-name", default="ensemble", help="Name used when saving checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Device to use (cpu, cuda, mps). Auto-detected if omitted.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    encoder = StateEncoder()
    dataset = DecisionDataset(args.log_path, encoder=encoder)
    if len(dataset) == 0:
        raise SystemExit("No training data found. Run the bot with training logs enabled first.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    num_actions = max(ACTION_TO_INDEX.values()) + 1
    actor = PokerActor(input_dim=encoder.feature_size, output_dim=num_actions).to(device)
    critic = PokerCritic(input_dim=encoder.feature_size).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        actor.train()
        critic.train()
        action_loss_total = 0.0
        amount_loss_total = 0.0
        value_loss_total = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            states = batch["state"].to(device)
            action_targets = batch["action"].to(device)
            amount_targets = batch["amount"].to(device)
            reward_targets = batch["reward"].to(device)

            optimizer.zero_grad(set_to_none=True)
            action_logits, bet_fraction = actor(states)
            action_loss = F.cross_entropy(action_logits, action_targets)
            amount_loss = F.mse_loss(bet_fraction, amount_targets)
            value_pred = critic(states)
            value_loss = F.mse_loss(value_pred, reward_targets)

            loss = action_loss + amount_loss + 0.5 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()

            action_loss_total += action_loss.item()
            amount_loss_total += amount_loss.item()
            value_loss_total += value_loss.item()

        num_batches = max(len(dataloader), 1)
        print(
            f"Epoch {epoch}: "
            f"action_loss={action_loss_total / num_batches:.4f}, "
            f"amount_loss={amount_loss_total / num_batches:.4f}, "
            f"value_loss={value_loss_total / num_batches:.4f}"
        )

        torch.save(actor.state_dict(), model_dir / f"{args.agent_name}_actor.pt")
        torch.save(critic.state_dict(), model_dir / f"{args.agent_name}_critic.pt")

    print(f"Training complete. Models saved to {model_dir.resolve()}")


if __name__ == "__main__":
    train()

