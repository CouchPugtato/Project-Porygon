from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RuntimeMessage:
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"))


def battle_start(battle_id: str, fmt: str, is_doubles: bool) -> RuntimeMessage:
    return RuntimeMessage(
        {"type": "battle_start", "battle_id": battle_id, "format": fmt, "is_doubles": is_doubles}
    )


def request_message(battle_id: str, request_id: int, payload: Dict[str, Any]) -> RuntimeMessage:
    return RuntimeMessage(
        {"type": "request", "battle_id": battle_id, "request_id": request_id, "payload": payload}
    )


def event_message(battle_id: str, seq: int, line: str) -> RuntimeMessage:
    return RuntimeMessage({"type": "event", "battle_id": battle_id, "seq": seq, "line": line})


def terminal_message(battle_id: str, result: str, reward: float) -> RuntimeMessage:
    return RuntimeMessage({"type": "terminal", "battle_id": battle_id, "result": result, "reward": reward})


def battle_end(battle_id: str) -> RuntimeMessage:
    return RuntimeMessage({"type": "battle_end", "battle_id": battle_id})


def decision_message(
    battle_id: str,
    request_id: int,
    action: int,
    command: str,
    accepted: bool | None = None,
    reason: str = "",
) -> RuntimeMessage:
    msg_type = "decision_proposed"
    if accepted is True:
        msg_type = "decision_accepted"
    elif accepted is False:
        msg_type = "decision_rejected"
    payload: Dict[str, Any] = {
        "type": msg_type,
        "battle_id": battle_id,
        "request_id": request_id,
        "action": action,
        "command": command,
    }
    if reason:
        payload["message"] = reason
    return RuntimeMessage(payload)
