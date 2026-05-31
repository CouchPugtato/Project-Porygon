from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from itertools import product
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
    if tag == "error" and len(parts) >= 3:
        return f"[battle] {battle_label(event.room_id)} error: {parts[2]}"
    return None


def is_fainted_condition(condition: str) -> bool:
    return "fnt" in condition.lower()


def target_options(target: str, slot_index: int) -> list[str]:
    if target in {"normal", "adjacentFoe", "any"}:
        return [" 1", " 2"]
    if target == "adjacentAlly":
        return [" -2" if slot_index == 0 else " -1"]
    if target == "adjacentAllyOrSelf":
        return [" -1" if slot_index == 0 else " -2", " -2" if slot_index == 0 else " -1"]
    return [""]


def legal_switch_indices(request_payload: dict) -> list[int]:
    side = request_payload.get("side", {})
    pokemon = side.get("pokemon", [])
    legal: list[int] = []
    for idx, mon in enumerate(pokemon, start=1):
        if mon.get("active"):
            continue
        if is_fainted_condition(str(mon.get("condition", ""))):
            continue
        legal.append(idx)
    return legal


def unfainted_active_slots(request_payload: dict) -> list[int]:
    side = request_payload.get("side", {})
    pokemon = side.get("pokemon", [])
    living: list[int] = []
    for idx, mon in enumerate(pokemon):
        if not mon.get("active"):
            continue
        if is_fainted_condition(str(mon.get("condition", ""))):
            continue
        living.append(idx)
    return living


def living_active_request_slots(request_payload: dict) -> list[int]:
    active = request_payload.get("active", [])
    living: list[int] = []
    for idx, slot in enumerate(active):
        if slot.get("fainted"):
            continue
        living.append(idx)
    return living


def force_switch_flags(request_payload: dict, active_count: int) -> list[bool]:
    raw = request_payload.get("forceSwitch")
    flags = [False] * max(active_count, 1)
    if isinstance(raw, list):
        for idx, value in enumerate(raw[: len(flags)]):
            flags[idx] = bool(value)
    return flags


def slot_action_options(request_payload: dict, slot_index: int) -> list[str]:
    active = request_payload.get("active", [])
    if slot_index >= len(active):
        return []

    slot = active[slot_index]
    options: list[str] = []
    force_switch = False
    force_switch_arr = request_payload.get("forceSwitch")
    if isinstance(force_switch_arr, list) and slot_index < len(force_switch_arr):
        force_switch = bool(force_switch_arr[slot_index])
    force_switch = force_switch or bool(slot.get("forceSwitch"))

    switches = legal_switch_indices(request_payload)
    trapped = bool(slot.get("trapped"))

    if not force_switch:
        for move_index, move in enumerate(slot.get("moves", []), start=1):
            if move.get("disabled"):
                continue
            suffixes = target_options(str(move.get("target", "")), slot_index)
            for suffix in suffixes:
                options.append(f"move {move_index}{suffix}")

    if force_switch or not trapped:
        for switch_index in switches:
            options.append(f"switch {switch_index}")

    return options


def fallback_commands_for_request(request_payload: dict) -> list[str]:
    if request_payload.get("teamPreview"):
        switches = legal_switch_indices(request_payload)
        return [f"/choose team {idx}" for idx in switches]

    active = request_payload.get("active", [])
    inferred_slots = len(active)
    raw_force_switch = request_payload.get("forceSwitch")
    if isinstance(raw_force_switch, list) and len(raw_force_switch) > inferred_slots:
        inferred_slots = len(raw_force_switch)
    if inferred_slots <= 0:
        inferred_slots = 1

    force_flags = force_switch_flags(request_payload, inferred_slots)
    if any(force_flags):
        switches = legal_switch_indices(request_payload)
        if inferred_slots == 1:
            return [f"/choose switch {idx}" for idx in switches]
        if force_flags == [True, False]:
            commands: list[str] = []
            for idx in switches:
                commands.append(f"/choose switch {idx}")
                commands.append(f"/choose switch {idx}, pass")
            return commands
        if force_flags == [False, True]:
            commands = []
            for idx in switches:
                commands.append(f"/choose pass, switch {idx}")
                commands.append(f"/choose switch {idx}")
            return commands
        commands: list[str] = []
        if len(switches) == 1:
            only = switches[0]
            commands.append(f"/choose switch {only}")
            commands.append(f"/choose switch {only}, pass")
            commands.append(f"/choose pass, switch {only}")
            return commands
        for first, second in product(switches, switches):
            if first == second:
                continue
            commands.append(f"/choose switch {first}, switch {second}")
        return commands

    if not active:
        return []

    living_request_slots = living_active_request_slots(request_payload)
    living_active_count = len(living_request_slots) or len(unfainted_active_slots(request_payload))
    if living_active_count <= 1:
        slot_index = living_request_slots[0] if living_request_slots else 0
        slot1 = slot_action_options(request_payload, slot_index)
        return [f"/choose {part}" for part in slot1]

    slot1 = slot_action_options(request_payload, 0)
    if len(active) == 1:
        return [f"/choose {part}" for part in slot1]

    slot2 = slot_action_options(request_payload, 1)
    commands: list[str] = []
    for part1, part2 in product(slot1, slot2):
        if part1.startswith("switch ") and part2.startswith("switch ") and part1 == part2:
            continue
        commands.append(f"/choose {part1}, {part2}")
    return commands


def next_fallback_command(
    request_payload: dict,
    attempted: set[str],
) -> str | None:
    candidates = fallback_commands_for_request(request_payload)
    random.shuffle(candidates)
    for candidate in candidates:
        if candidate in attempted:
            continue
        attempted.add(candidate)
        return candidate
    return None


def parse_player_line(line: str) -> tuple[str, str] | None:
    parts = line.split("|")
    if len(parts) >= 4 and parts[1] == "player":
        return parts[2], parts[3]
    return None


def parse_player_metadata(line: str) -> tuple[str, str, int | None] | None:
    parts = line.split("|")
    if len(parts) >= 4 and parts[1] == "player":
        rating = None
        if len(parts) >= 6:
            try:
                rating = int(parts[5])
            except ValueError:
                rating = None
        return parts[2], parts[3], rating
    return None


class MatchStats:
    def __init__(self, replay_path: Path) -> None:
        self._stats_path = replay_path.with_suffix(".stats.txt")
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.max_rating = 0
        self._load()

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}-{self.draws}"

    def _load(self) -> None:
        if not self._stats_path.exists():
            return
        for line in self._stats_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                parsed = int(value.strip())
            except ValueError:
                continue
            if key == "matches_played":
                self.matches_played = parsed
            elif key == "wins":
                self.wins = parsed
            elif key == "losses":
                self.losses = parsed
            elif key == "draws":
                self.draws = parsed
            elif key == "max_rating":
                self.max_rating = parsed

    def save(self) -> None:
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats_path.write_text(
            "\n".join(
                [
                    f"matches_played={self.matches_played}",
                    f"record={self.record}",
                    f"wins={self.wins}",
                    f"losses={self.losses}",
                    f"draws={self.draws}",
                    f"max_rating={self.max_rating}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def note_battle(self, result: str, bot_rating: int | None) -> None:
        self.matches_played += 1
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        else:
            self.losses += 1
        if bot_rating is not None:
            self.max_rating = max(self.max_rating, bot_rating)
        self.save()


MAX_INVALID_RETRIES_PER_REQUEST = 24


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
    stats = MatchStats(out_path)
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
            stats.note_battle(result, None)
            print(f"[capture] finished {event.room_id} result={result} total_matches_captured={finished_battles}")
            await gateway.search_next_battle()
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
    stats = MatchStats(replay_path or Path("matches/runtime_capture.jsonl"))
    seq = 0
    latest_requests: dict[str, dict] = {}
    attempted_commands: dict[str, set[str]] = {}
    latest_request_seq: dict[str, int] = {}
    request_open: dict[str, bool] = {}
    invalid_retry_count: dict[str, int] = {}
    learner_rerolls: dict[str, int] = {}
    battle_players: dict[str, dict[str, str]] = {}
    battle_ratings: dict[str, dict[str, int | None]] = {}
    announced_matchups: set[str] = set()

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal seq
        summary = event_summary(event)
        if summary:
            print(summary)
        player_info = parse_player_metadata(event.line)
        if player_info is not None:
            side, username_found, rating = player_info
            players = battle_players.setdefault(event.room_id, {})
            ratings = battle_ratings.setdefault(event.room_id, {})
            players[side] = username_found
            ratings[side] = rating
            if event.room_id not in announced_matchups and "p1" in players and "p2" in players:
                print()
                print(f"[stats] matches={stats.matches_played} record={stats.record} max_rating={stats.max_rating}")
                print(
                    f"[matchup] {players['p1']} ({ratings.get('p1', '?') if ratings.get('p1') is not None else '?'}) vs "
                    f"{players['p2']} ({ratings.get('p2', '?') if ratings.get('p2') is not None else '?'})"
                )
                announced_matchups.add(event.room_id)
        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            request_payload = json.loads(payload)
            latest_requests[event.room_id] = request_payload
            attempted_commands[event.room_id] = set()
            latest_request_seq[event.room_id] = seq
            request_open[event.room_id] = True
            invalid_retry_count[event.room_id] = 0
            learner_rerolls[event.room_id] = 0
            print(f"[live] sending request {seq} to learner for {battle_label(event.room_id)}")
            await learner.send(request_message(event.room_id, seq, request_payload).payload)
            print(f"[live] request {seq} for {battle_label(event.room_id)}")
        elif event.line.startswith("|error|") and "Invalid choice" in event.line:
            if "There's nothing to choose" in event.line:
                request_open[event.room_id] = False
                print(f"[live] request closed for {battle_label(event.room_id)} after stale choice rejection")
                return
            request_payload = latest_requests.get(event.room_id)
            if request_payload is not None and request_open.get(event.room_id, False):
                invalid_retry_count[event.room_id] = invalid_retry_count.get(event.room_id, 0) + 1
                if invalid_retry_count[event.room_id] > MAX_INVALID_RETRIES_PER_REQUEST:
                    request_open[event.room_id] = False
                    print(f"[live] request closed for {battle_label(event.room_id)} after too many invalid retries")
                    return
                tried = attempted_commands.setdefault(event.room_id, set())
                if learner_rerolls.get(event.room_id, 0) < 3 and random.random() < 0.35:
                    learner_rerolls[event.room_id] = learner_rerolls.get(event.room_id, 0) + 1
                    retry_seq = latest_request_seq.get(event.room_id, seq)
                    print(f"[live] rerolling learner choice for {battle_label(event.room_id)}")
                    await learner.send(request_message(event.room_id, retry_seq, request_payload).payload)
                else:
                    candidate = next_fallback_command(request_payload, tried)
                    if candidate is not None:
                        print(f"[live] retrying with fallback for {battle_label(event.room_id)}: {candidate}")
                        await gateway.send_room_command(event.room_id, candidate)
                    else:
                        print(
                            f"[live] fallback debug for {battle_label(event.room_id)} "
                            f"forceSwitch={request_payload.get('forceSwitch')} "
                            f"legal_switches={legal_switch_indices(request_payload)} "
                            f"living_active={unfainted_active_slots(request_payload)}"
                        )
                        print(f"[live] no fallback candidates left for {battle_label(event.room_id)}")
        elif event.line.startswith("|win|") or event.line.startswith("|tie|"):
            result = "win" if event.line.startswith("|win|") else "draw"
            reward = 1.0 if result == "win" else 0.0
            await learner.send(terminal_message(event.room_id, result, reward).payload)
            await learner.send(battle_end(event.room_id).payload)
            players = battle_players.get(event.room_id, {})
            ratings = battle_ratings.get(event.room_id, {})
            bot_side = "p2" if players.get("p2", "").startswith("Guest") or players.get("p2") == username or not username else "p1"
            bot_rating = ratings.get(bot_side)
            final_result = result
            if result == "win":
                winner_name = event.line.split("|", 2)[2] if "|" in event.line else ""
                if players.get(bot_side) != winner_name:
                    final_result = "loss"
            stats.note_battle(final_result, bot_rating)
            print(f"[live] battle ended {event.room_id} result={result}")
            request_open[event.room_id] = False
            await gateway.search_next_battle()
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
                if not request_open.get(battle_id, False):
                    print(f"[live] ignoring stale learner action for closed request {battle_id}: {msg['command']}")
                    continue
                print(f"[live] action for {battle_id}: {msg['command']}")
                attempted_commands.setdefault(battle_id, set()).add(msg["command"])
                await gateway.send_room_command(battle_id, msg["command"])
            elif msg.get("type") == "error":
                print(f"[live] learner error: {msg}")

    async def stderr_loop() -> None:
        while True:
            line = await learner.read_stderr_line()
            if line is None:
                return
            if line:
                print(f"[learner-stderr] {line}")

    await gateway.connect()
    await asyncio.gather(gateway.run(on_event), action_loop(), stderr_loop())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "capture"], default="live")
    parser.add_argument("--replay-path", default="matches/runtime_capture.jsonl")
    parser.add_argument("--learner-command", default="./showdown_client")
    parser.add_argument("--learner-args", nargs="*", default=[])
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--username", default="")
    args, learner_passthrough = parser.parse_known_args()

    if args.mode == "capture":
        asyncio.run(capture_mode(Path(args.replay_path), args.format, args.username))
    else:
        learner_args = list(args.learner_args) + list(learner_passthrough)
        if not learner_args:
            learner_args = ["--runtime"]
        learner_command = [args.learner_command] + learner_args
        asyncio.run(live_mode(learner_command, Path(args.replay_path), args.format, args.username))


if __name__ == "__main__":
    main()
