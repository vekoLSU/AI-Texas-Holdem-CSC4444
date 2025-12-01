from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

PHASES: List[str] = ["WAITING", "PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
POSITIONS: List[str] = ["early", "middle", "late", "heads-up", "button", "unknown"]


class StateEncoder:
    """
    Transforms logged decision data into fixed-size numeric feature vectors
    that can be consumed by PyTorch models.
    """

    def __init__(
        self,
        *,
        max_stack: float = 4000.0,
        max_pot: float = 4000.0,
        max_players: int = 9,
    ) -> None:
        self.max_stack = max_stack
        self.max_pot = max_pot
        self.max_players = max_players
        self._phase_to_idx = {phase: idx for idx, phase in enumerate(PHASES)}
        self._position_to_idx = {pos: idx for idx, pos in enumerate(POSITIONS)}
        self._feature_size = 10 + len(PHASES) + len(POSITIONS)

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def encode(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Convert a single logged decision into a numeric feature vector.
        """
        hand_strength = record.get("hand_strength", {})
        state = record.get("state", {})
        opponents = record.get("opponents", [])

        features: List[float] = [
            float(hand_strength.get("strength", 0.0)),
            float(hand_strength.get("draw_potential", 0.0)),
            self._normalize(state.get("pot", 0)),
            self._normalize(state.get("to_call", 0)),
            self._normalize(state.get("our_chips", 0), self.max_stack),
            self._normalize(state.get("num_players", 0), self.max_players),
            self._normalize(state.get("current_bet", 0)),
        ]

        opponent_summary = self._summarize_opponents(opponents)
        features.extend(
            [
                opponent_summary["avg_aggression"],
                opponent_summary["avg_vpip"],
                opponent_summary["avg_stack_ratio"],
            ]
        )

        features.extend(self._one_hot_phase(record.get("phase", "UNKNOWN")))
        features.extend(self._one_hot_position(record.get("position", "unknown")))

        return np.asarray(features, dtype=np.float32)

    def encode_batch(self, records: Iterable[Dict[str, Any]]) -> np.ndarray:
        """Vectorize a list of decisions."""
        return np.stack([self.encode(record) for record in records])

    def _summarize_opponents(self, opponents: List[Dict[str, Any]]) -> Dict[str, float]:
        if not opponents:
            return {
                "avg_aggression": 0.0,
                "avg_vpip": 0.0,
                "avg_stack_ratio": 0.0,
            }

        agg_total = 0.0
        vpip_total = 0.0
        stack_total = 0.0

        for opponent in opponents:
            agg_total += float(opponent.get("aggression_factor", 0.0))
            vpip_total += float(opponent.get("vpip", 0.0))
            stack_total += float(opponent.get("chips", 0.0))

        count = max(len(opponents), 1)
        return {
            "avg_aggression": self._clamp(agg_total / count / 5.0),
            "avg_vpip": self._clamp(vpip_total / count),
            "avg_stack_ratio": self._normalize(stack_total / count, self.max_stack),
        }

    def _one_hot_phase(self, phase: str) -> List[float]:
        vec = [0.0] * len(PHASES)
        idx = self._phase_to_idx.get(phase.upper())
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _one_hot_position(self, position: str) -> List[float]:
        vec = [0.0] * len(POSITIONS)
        normalized_position = position.lower()
        idx = self._position_to_idx.get(normalized_position)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _normalize(self, value: float, limit: Optional[float] = None) -> float:
        if limit is None:
            limit = self.max_pot
        if limit <= 0:
            return 0.0
        return self._clamp(float(value) / float(limit))

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(value, maximum))

