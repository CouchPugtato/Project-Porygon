from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from communicator.ipc import LearnerProcess
    from communicator.protocol import battle_end, battle_start, event_message, request_message, terminal_message
    from communicator.showdown_client import ShowdownGateway, default_showdown_uri
else:
    from .ipc import LearnerProcess
    from .protocol import battle_end, battle_start, event_message, request_message, terminal_message
    from .showdown_client import ShowdownGateway, default_showdown_uri


async def capture_mode(out_path: Path) -> None:
    gateway = ShowdownGateway(default_showdown_uri())
    await gateway.connect()
    seq = 0

    async def on_message(message: str) -> None:
        nonlocal seq
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for raw_line in message.splitlines():
                if raw_line.startswith(">battle-"):
                    battle_id = raw_line[1:]
                    handle.write(battle_start(battle_id, "unknown", True).to_json() + "\n")
                elif "|request|" in raw_line:
                    room, payload = raw_line.split("|request|", 1)
                    handle.write(request_message(room.lstrip(">"), seq, json.loads(payload)).to_json() + "\n")
                elif raw_line.startswith("|win|") or raw_line.startswith("|tie|"):
                    result = "win" if raw_line.startswith("|win|") else "draw"
                    handle.write(terminal_message("unknown", result, 1.0 if result == "win" else 0.0).to_json() + "\n")
                elif raw_line.startswith("|"):
                    handle.write(event_message("unknown", seq, raw_line).to_json() + "\n")
                seq += 1

    await gateway.run(on_message)


async def live_mode(learner_command: list[str], replay_path: Path | None) -> None:
    gateway = ShowdownGateway(default_showdown_uri())
    learner = LearnerProcess(learner_command, replay_path)
    await learner.start()
    await gateway.connect()
    seq = 0

    async def on_message(message: str) -> None:
        nonlocal seq
        current_battle = "unknown"
        for raw_line in message.splitlines():
            if raw_line.startswith(">battle-"):
                current_battle = raw_line[1:]
                await learner.send(battle_start(current_battle, "unknown", True).payload)
            elif "|request|" in raw_line:
                room, payload = raw_line.split("|request|", 1)
                current_battle = room.lstrip(">")
                await learner.send(request_message(current_battle, seq, json.loads(payload)).payload)
            elif raw_line.startswith("|win|") or raw_line.startswith("|tie|"):
                result = "win" if raw_line.startswith("|win|") else "draw"
                await learner.send(terminal_message(current_battle, result, 1.0 if result == "win" else 0.0).payload)
                await learner.send(battle_end(current_battle).payload)
            elif raw_line.startswith("|"):
                await learner.send(event_message(current_battle, seq, raw_line).payload)
            seq += 1

    async def action_loop() -> None:
        while True:
            msg = await learner.read_message()
            if not msg:
                return
            if msg.get("type") == "action":
                battle_id = msg["battle_id"]
                await gateway.send(f"{battle_id}|{msg['command']}")

    await asyncio.gather(gateway.run(on_message), action_loop())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "capture"], default="live")
    parser.add_argument("--replay-path", default="matches/runtime_capture.jsonl")
    parser.add_argument("--learner-command", nargs="+", default=["./showdown_client", "--runtime"])
    args = parser.parse_args()

    if args.mode == "capture":
        asyncio.run(capture_mode(Path(args.replay_path)))
    else:
        asyncio.run(live_mode(args.learner_command, Path(args.replay_path)))


if __name__ == "__main__":
    main()
