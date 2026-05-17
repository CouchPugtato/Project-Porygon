from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from communicator.ipc import LearnerProcess
    from communicator.protocol import battle_end, battle_start, event_message, request_message, terminal_message
    from communicator.showdown_client import ShowdownEvent, ShowdownGateway, default_showdown_uri, infer_is_doubles
else:
    from .ipc import LearnerProcess
    from .protocol import battle_end, battle_start, event_message, request_message, terminal_message
    from .showdown_client import ShowdownEvent, ShowdownGateway, default_showdown_uri, infer_is_doubles


def battle_label(battle_id: str) -> str:
    return battle_id if battle_id else "unknown-battle"


def event_summary(event: ShowdownEvent) -> str | None:
    line = event.line
    parts = line.split("|")
    if len(parts) < 2:
        return None
    tag = parts[1]
    if tag == "turn" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} turn {parts[2]}"
    if tag == "move" and len(parts) >= 5:
        return f"[battle] {battle_label(event.room_id)} move: {parts[2]} used {parts[3]} on {parts[4]}"
    if tag == "switch" and len(parts) >= 4:
        return f"[battle] {battle_label(event.room_id)} switch: {parts[2]} -> {parts[3]}"
    if tag == "faint" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} faint: {parts[2]}"
    if tag == "-weather" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} weather: {parts[2]}"
    if tag == "-fieldstart" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} field start: {parts[2]}"
    if tag == "-fieldend" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} field end: {parts[2]}"
    if tag == "-sidestart" and len(parts) >= 4:
        return f"[battle] {battle_label(event.room_id)} side start: {parts[2]} {parts[3]}"
    if tag == "-sideend" and len(parts) >= 4:
        return f"[battle] {battle_label(event.room_id)} side end: {parts[2]} {parts[3]}"
    if tag == "win" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} winner: {parts[2]}"
    if tag == "tie":
        return f"[battle] {battle_label(event.room_id)} result: tie"
    return None


class ReplayWriter:
    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path

    def append(self, payload: str) -> None:
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


async def capture_mode(out_path: Path, fmt: str, username: str) -> None:
    gateway = ShowdownGateway(default_showdown_uri(), username=username, fmt=fmt)
    writer = ReplayWriter(out_path)
    started_battles: set[str] = set()
    finished_battles = 0
    seq = 0

    print(f"[capture] writing replay records to {out_path}")

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal finished_battles, seq

        if event.room_id.startswith("battle-") and event.room_id not in started_battles:
            started_battles.add(event.room_id)
            writer.append(
                battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).to_json()
            )
            print(f"[capture] started {event.room_id}")
            await gateway.send_room_command(event.room_id, "/timer on")
            print(f"[capture] timer enabled for {event.room_id}")

        summary = event_summary(event)
        if summary:
            print(summary)

        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            writer.append(request_message(event.room_id, seq, json.loads(payload)).to_json())
            print(f"[capture] request {seq} for {battle_label(event.room_id)}")
        elif event.line.startswith("|win|") or event.line.startswith("|tie|"):
            result = "win" if event.line.startswith("|win|") else "draw"
            reward = 1.0 if result == "win" else 0.0
            writer.append(terminal_message(event.room_id, result, reward).to_json())
            writer.append(battle_end(event.room_id).to_json())
            finished_battles += 1
            print(f"[capture] finished {event.room_id} result={result} total_matches_captured={finished_battles}")
        else:
            writer.append(event_message(event.room_id, seq, event.line).to_json())

        seq += 1

    await gateway.connect()
    await gateway.run(on_event)


async def live_mode(learner_command: list[str], replay_path: Path | None, fmt: str, username: str) -> None:
    gateway = ShowdownGateway(default_showdown_uri(), username=username, fmt=fmt)
    learner = LearnerProcess(learner_command, replay_path)
    await learner.start()
    print(f"[live] started learner: {' '.join(learner_command)}")
    seq = 0

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal seq
        summary = event_summary(event)
        if summary:
            print(summary)
        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            await learner.send(request_message(event.room_id, seq, json.loads(payload)).payload)
            print(f"[live] request {seq} for {battle_label(event.room_id)}")
        elif event.line.startswith("|win|") or event.line.startswith("|tie|"):
            result = "win" if event.line.startswith("|win|") else "draw"
            reward = 1.0 if result == "win" else 0.0
            await learner.send(terminal_message(event.room_id, result, reward).payload)
            await learner.send(battle_end(event.room_id).payload)
            print(f"[live] battle ended {event.room_id} result={result}")
        else:
            if event.room_id.startswith("battle-") and event.line.startswith("|init|battle"):
                await learner.send(
                    battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).payload
                )
                print(f"[live] battle started {event.room_id}")
                await gateway.send_room_command(event.room_id, "/timer on")
                print(f"[live] timer enabled for {event.room_id}")
            await learner.send(event_message(event.room_id, seq, event.line).payload)

        seq += 1

    async def action_loop() -> None:
        while True:
            msg = await learner.read_message()
            if not msg:
                print("[live] learner exited")
                return
            if msg.get("type") == "ready":
                print(f"[live] learner ready: {msg}")
            elif msg.get("type") == "action":
                battle_id = msg["battle_id"]
                print(f"[live] action for {battle_id}: {msg['command']}")
                await gateway.send_room_command(battle_id, msg["command"])
            elif msg.get("type") == "error":
                print(f"[live] learner error: {msg}")

    await gateway.connect()
    await asyncio.gather(gateway.run(on_event), action_loop())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "capture"], default="live")
    parser.add_argument("--replay-path", default="matches/runtime_capture.jsonl")
    parser.add_argument("--learner-command", nargs="+", default=["./showdown_client", "--runtime"])
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--username", default="")
    args = parser.parse_args()

    if args.mode == "capture":
        asyncio.run(capture_mode(Path(args.replay_path), args.format, args.username))
    else:
        asyncio.run(live_mode(args.learner_command, Path(args.replay_path), args.format, args.username))


if __name__ == "__main__":
    main()
