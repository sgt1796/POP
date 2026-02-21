from __future__ import annotations

import json
import os
from typing import Dict, List


STATUS_TRANSITIONS: Dict[str, List[str]] = {
    "draft": ["validated", "rejected"],
    "validated": ["approval_required", "rejected"],
    "approval_required": ["approved", "rejected"],
    "approved": ["activated", "rejected"],
    "rejected": [],
    "activated": ["approved"],  # allow deactivation to approved state
}


class ApprovalStateMachine:
    """Simple state machine for dynamic tool approval lifecycle."""

    def transition(self, current: str, target: str) -> str:
        if current == target:
            return target
        allowed = STATUS_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(f"Invalid tool status transition: {current} -> {target}")
        return target


def write_review_artifact(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
