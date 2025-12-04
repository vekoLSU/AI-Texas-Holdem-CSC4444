from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional


class TrainingRecorder:
    """
    Collects per-decision data so that it can be used for supervised or
    reinforcement learning later on.

    The recorder keeps decisions in memory until the hand outcome is known,
    then writes a JSON line per decision that includes the final result.
    """

    def __init__(
        self,
        log_path: str = "logs/training_decisions.jsonl",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_decision(
        self,
        *,
        hand_id: int,
        phase: str,
        position: str,
        hand_strength: Dict[str, Any],
        state_snapshot: Dict[str, Any],
        opponent_profiles: List[Dict[str, Any]],
        decision: Dict[str, Any],
        agent_meta: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a single decision until the hand outcome is known."""
        if not self.enabled:
            return

        entry = {
            "hand_id": int(hand_id),
            "phase": phase,
            "position": position,
            "hand_strength": hand_strength,
            "state": state_snapshot,
            "opponents": opponent_profiles,
            "decision": {
                "action": decision.get("action"),
                "amount": decision.get("amount", 0),
            },
            "agent": agent_meta or {},
            "timestamp": self._utc_now(),
        }

        if extra:
            entry["extra"] = extra

        self._pending[str(hand_id)].append(entry)

    def record_outcome(
        self,
        hand_id: int,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Attach an outcome to all pending decisions for the given hand and
        persist them to disk.
        """
        if not self.enabled:
            return

        records = self._pending.pop(str(hand_id), [])
        if not records:
            return

        outcome_with_id = {"hand_id": int(hand_id), **outcome}
        self._write_records(records, outcome_with_id)

    def flush(self) -> None:
        """Force pending decisions to be written even if no outcome arrived."""
        if not self.enabled or not self._pending:
            return

        fallback_outcome = {
            "won": False,
            "pot": 0,
            "chips_delta": 0,
            "note": "forced_flush_no_outcome",
        }

        records: List[Dict[str, Any]] = []
        for hand_id, entries in self._pending.items():
            for entry in entries:
                entry.setdefault("timestamp", self._utc_now())
                records.append(entry)

        self._pending.clear()

        # Write with fallback outcome so the data is still usable later.
        self._write_records(records, fallback_outcome)

    def _write_records(
        self,
        records: List[Dict[str, Any]],
        outcome: Dict[str, Any],
    ) -> None:
        timestamp = self._utc_now()
        with self.log_path.open("a", encoding="utf-8") as handle:
            for record in records:
                record.setdefault("timestamp", timestamp)
                record["outcome"] = outcome
                handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.flush()
        except Exception:
            # Never raise during GC
            pass

