from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .state_encoder import StateEncoder

ACTION_TO_INDEX: Dict[str, int] = {
    "fold": 0,
    "check": 1,
    "call": 2,
    "raise": 3,
    "all-in": 4,
    "all_in": 4,
}
INDEX_TO_ACTION: Dict[int, str] = {idx: action for action, idx in ACTION_TO_INDEX.items()}


class DecisionDataset(Dataset):
    """
    Torch dataset that loads logged decisions and outcomes to produce
    supervised learning targets for the actor/critic networks.
    """

    def __init__(
        self,
        log_path: str = "logs/training_decisions.jsonl",
        encoder: Optional[StateEncoder] = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.encoder = encoder or StateEncoder()
        self._records: List[Dict[str, Any]] = []
        self._rewards: List[float] = []
        self._load()

    def _load(self) -> None:
        if not self.log_path.exists():
            return

        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "decision" not in record or "outcome" not in record:
                    continue
                self._records.append(record)
                self._rewards.append(self._compute_reward(record))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self._records[index]
        reward = torch.tensor(self._rewards[index], dtype=torch.float32)
        state_tensor = torch.from_numpy(self.encoder.encode(record))

        decision = record.get("decision", {})
        action_name = str(decision.get("action", "fold")).lower()
        action_idx = ACTION_TO_INDEX.get(action_name, ACTION_TO_INDEX["fold"])

        amount = float(decision.get("amount", 0.0))
        pot = float(record.get("state", {}).get("pot", 1.0)) or 1.0
        amount_fraction = max(0.0, min(amount / pot, 1.0))

        return {
            "state": state_tensor,
            "action": torch.tensor(action_idx, dtype=torch.long),
            "amount": torch.tensor(amount_fraction, dtype=torch.float32),
            "reward": reward,
        }

    def _compute_reward(self, record: Dict[str, Any]) -> float:
        outcome = record.get("outcome", {})
        chips_delta = outcome.get("chips_delta")
        if chips_delta is not None:
            return float(chips_delta) / max(self.encoder.max_stack, 1.0)

        pot = float(outcome.get("pot", 0.0))
        won = bool(outcome.get("won"))
        baseline = pot if won else -pot
        return baseline / max(self.encoder.max_pot, 1.0)

