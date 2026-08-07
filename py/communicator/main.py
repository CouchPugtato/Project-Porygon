from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import sys
import time
from itertools import product
from pathlib import Path
from websockets.exceptions import ConnectionClosed

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from communicator.ipc import LearnerProcess
    from communicator.protocol import battle_end, battle_start, decision_message, event_message, request_message, terminal_message
    from communicator.protocol import action_taken_message, action_rejected_message
    from communicator.showdown_client import ShowdownEvent, ShowdownGateway, default_showdown_uri, infer_is_doubles
else:
    from .ipc import LearnerProcess
    from .protocol import (
        action_rejected_message,
        action_taken_message,
        battle_end,
        battle_start,
        decision_message,
        event_message,
        request_message,
        terminal_message,
    )
    from .showdown_client import ShowdownEvent, ShowdownGateway, default_showdown_uri, infer_is_doubles

THINK_DELAY_MIN_SECONDS = 0.8
THINK_DELAY_MAX_SECONDS = 5.0
FALLBACK_DELAY_MIN_SECONDS = 0.4
FALLBACK_DELAY_MAX_SECONDS = 1.2
DEFAULT_ARGS_PATH = Path("config/communicator.toml")
DEFAULT_REWARD_WEIGHTS_PATH = Path("config/reward_weights.toml")
DEFAULT_RECONNECT_SECONDS = 5.0
DEFAULT_LEARNER_COMMAND = r".\build-fresh\showdown_client.exe"
DEFAULT_GUEST_REFRESH_SECONDS = 0.0
PROTECT_MOVE_NAMES = {
    "protect",
    "detect",
    "spiky shield",
    "baneful bunker",
    "burning bulwark",
    "silk trap",
    "kings shield",
    "obstruct",
    "endure",
}


def load_default_args(path: Path) -> list[str]:
    if not path.exists():
        return []
    bare_flags = {"battle_agent"}
    args: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise SystemExit(f"invalid config {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if not key:
            raise SystemExit(f"invalid config {path}:{line_number}: empty key")
        flag = "--" + key.replace("_", "-")
        lowered = value_text.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
            if key in bare_flags:
                if value:
                    args.append(flag)
            else:
                args.extend([flag, "1" if value else "0"])
            continue
        if len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        args.extend([flag, value])
    return args


def load_flat_toml_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise SystemExit(f"invalid config {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if not key:
            raise SystemExit(f"invalid config {path}:{line_number}: empty key")
        if len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        values[key] = value
    return values


def reward_float(config: dict[str, str], key: str, default: float) -> float:
    raw_value = config.get(key)
    if raw_value is None or raw_value == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"invalid reward config {key}={raw_value!r}") from exc


def resolve_replay_save_path(run_name: str) -> Path:
    normalized = (run_name or "").strip()
    if not normalized:
        normalized = "runtime_capture"
    normalized_path = Path(normalized)
    if len(normalized_path.parts) > 1:
        run_dir = Path("matches") / "runs" / normalized_path.parent
        file_stem = normalized_path.name
    else:
        run_dir = Path("matches") / "runs" / normalized
        file_stem = normalized
    return run_dir / f"{file_stem}_raw.jsonl"


def resolved_server_uri(explicit_uri: str | None) -> str:
    if explicit_uri and explicit_uri.strip():
        return explicit_uri.strip()
    return default_showdown_uri()


def shutdown_file_requested(shutdown_file: Path | None) -> bool:
    return shutdown_file is not None and shutdown_file.exists()


def should_refresh_guest(username: str, guest_refresh_seconds: float, session_started_at: float | None) -> bool:
    if username:
        return False
    if guest_refresh_seconds <= 0.0 or session_started_at is None:
        return False
    return (time.monotonic() - session_started_at) >= guest_refresh_seconds


def battle_label(battle_id: str) -> str:
    return battle_id if battle_id else "unknown-battle"


async def randomized_send_delay(min_seconds: float, max_seconds: float) -> None:
    if max_seconds <= 0.0:
        return
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)


async def enable_battle_timer(gateway: ShowdownGateway | None, room_id: str, prefix: str) -> None:
    if gateway is None:
        return
    try:
        await gateway.send_room_command(room_id, "/timer on")
        print(f"[{prefix}] timer enabled for {room_id}")
    except (RuntimeError, ConnectionClosed, OSError) as exc:
        print(f"[{prefix}] timer enable skipped for {room_id}: {exc}")


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


def is_disconnect_or_forfeit_end(line: str) -> bool:
    if not line.startswith("|-message|"):
        return False
    lowered = line.lower()
    return " forfeited." in lowered or "lost due to inactivity." in lowered


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


def active_request_slots(request_payload: dict) -> list[int]:
    side = request_payload.get("side", {})
    pokemon = side.get("pokemon", [])
    request_slots: list[int] = []
    for idx, mon in enumerate(pokemon):
        if mon.get("active"):
            request_slots.append(idx)
            if len(request_slots) >= 2:
                break
    return request_slots


def living_active_request_slots(request_payload: dict) -> list[int]:
    active_slots = active_request_slots(request_payload)
    living: list[int] = []
    for request_slot, team_slot in enumerate(active_slots):
        side = request_payload.get("side", {})
        pokemon = side.get("pokemon", [])
        if team_slot >= len(pokemon):
            continue
        if is_fainted_condition(str(pokemon[team_slot].get("condition", ""))):
            continue
        living.append(request_slot)
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
    living_slots = living_active_request_slots(request_payload)
    force_switch_arr = request_payload.get("forceSwitch")
    if slot_index not in living_slots and not (
        isinstance(force_switch_arr, list) and slot_index < len(force_switch_arr) and bool(force_switch_arr[slot_index])
    ):
        return []

    slot = active[slot_index]
    options: list[str] = []
    force_switch = False
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

    if len(living_request_slots) >= 2:
        slot_indices = living_request_slots[:2]
    else:
        slot_indices = [0, 1]

    slot1 = slot_action_options(request_payload, slot_indices[0])
    if len(active) == 1 or len(slot_indices) == 1:
        return [f"/choose {part}" for part in slot1]

    slot2 = slot_action_options(request_payload, slot_indices[1])
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


def random_mode_candidates(request_payload: dict) -> list[str]:
    base_candidates = fallback_commands_for_request(request_payload)
    candidates = list(base_candidates)
    active = request_payload.get("active", [])
    slot_indices = command_slot_indices(request_payload)

    if request_payload.get("teamPreview"):
        return candidates

    for candidate in base_candidates:
        parts = parse_choose_parts(candidate)
        if not parts:
            continue
        for idx, part in enumerate(parts):
            slot_index = slot_indices[idx] if idx < len(slot_indices) else idx
            if slot_index < 0 or slot_index >= len(active):
                continue
            if not part.startswith("move "):
                continue
            if not active[slot_index].get("canTerastallize"):
                continue
            tera_parts = list(parts)
            tera_parts[idx] = f"{part} terastallize"
            tera_candidate = "/choose " + ", ".join(tera_parts)
            if tera_candidate not in candidates:
                candidates.append(tera_candidate)

    return candidates


def weighted_random_command(
    request_payload: dict,
    attempted: set[str],
) -> str | None:
    candidates = [candidate for candidate in random_mode_candidates(request_payload) if candidate not in attempted]
    if not candidates:
        return None
    if request_payload.get("teamPreview"):
        choice = random.choice(candidates)
        attempted.add(choice)
        return choice

    active = request_payload.get("active", [])
    inferred_slots = len(active)
    raw_force_switch = request_payload.get("forceSwitch")
    if isinstance(raw_force_switch, list) and len(raw_force_switch) > inferred_slots:
        inferred_slots = len(raw_force_switch)
    if inferred_slots <= 0:
        inferred_slots = 1
    if any(force_switch_flags(request_payload, inferred_slots)):
        choice = random.choice(candidates)
        attempted.add(choice)
        return choice

    weighted_candidates: list[tuple[str, int]] = []
    for candidate in candidates:
        if " terastallize" in candidate:
            weighted_candidates.append((candidate, 4))
            continue
        weight = 1
        for part in parse_choose_parts(candidate):
            if part.startswith("move "):
                weight *= 8
            elif part.startswith("switch "):
                weight *= 1
            elif part == "pass":
                weight *= 1
        weighted_candidates.append((candidate, max(weight, 1)))

    total_weight = sum(weight for _, weight in weighted_candidates)
    roll = random.randint(1, total_weight)
    cumulative = 0
    for candidate, weight in weighted_candidates:
        cumulative += weight
        if roll <= cumulative:
            attempted.add(candidate)
            return candidate

    choice = weighted_candidates[-1][0]
    attempted.add(choice)
    return choice


def command_slot_indices(request_payload: dict) -> list[int]:
    if request_payload.get("teamPreview"):
        return []

    active = request_payload.get("active", [])
    inferred_slots = len(active)
    raw_force_switch = request_payload.get("forceSwitch")
    if isinstance(raw_force_switch, list) and len(raw_force_switch) > inferred_slots:
        inferred_slots = len(raw_force_switch)
    if inferred_slots <= 0:
        inferred_slots = 1

    force_flags = force_switch_flags(request_payload, inferred_slots)
    if any(force_flags):
        if inferred_slots == 1:
            return [0]
        if force_flags == [True, False]:
            return [0, 1]
        if force_flags == [False, True]:
            return [0, 1]
        return list(range(inferred_slots))

    living_slots = living_active_request_slots(request_payload)
    if len(living_slots) <= 1:
        return [living_slots[0] if living_slots else 0]
    return living_slots[:2]


def parse_choose_parts(command: str) -> list[str]:
    if not command.startswith("/choose "):
        return []
    body = command[len("/choose ") :]
    return [part.strip() for part in body.split(",")]


def empty_action_counts() -> dict[str, int]:
    return {
        "forced_switches": 0,
        "voluntary_switches": 0,
        "moves": 0,
        "protects": 0,
        "passes": 0,
        "teras": 0,
        "move_slot_1": 0,
        "move_slot_2": 0,
        "move_slot_3": 0,
        "move_slot_4": 0,
        "switch_slot_1": 0,
        "switch_slot_2": 0,
        "switch_slot_3": 0,
        "switch_slot_4": 0,
        "switch_slot_5": 0,
        "switch_slot_6": 0,
    }


def move_name_for_part(request_payload: dict, slot_index: int, part: str) -> str:
    tokens = part.split()
    if len(tokens) < 2 or tokens[0] != "move":
        return ""
    try:
        move_index = int(tokens[1]) - 1
    except ValueError:
        return ""
    active = request_payload.get("active", [])
    if slot_index < 0 or slot_index >= len(active):
        return ""
    moves = active[slot_index].get("moves", [])
    if move_index < 0 or move_index >= len(moves):
        return ""
    return str(moves[move_index].get("move", "")).strip().lower()


def tally_command_categories(request_payload: dict, command: str, counts: dict[str, int]) -> None:
    parts = parse_choose_parts(command)
    if not parts:
        return
    slot_indices = command_slot_indices(request_payload)
    force_flags = force_switch_flags(request_payload, max(len(request_payload.get("active", [])), len(slot_indices), 1))
    for idx, part in enumerate(parts):
        if " terastallize" in part:
            counts["teras"] += 1
        if part == "pass":
            counts["passes"] += 1
            continue
        slot_index = slot_indices[idx] if idx < len(slot_indices) else idx
        forced_slot = idx < len(force_flags) and force_flags[slot_index if slot_index < len(force_flags) else idx]
        if part.startswith("switch "):
            switch_tokens = part.split()
            if len(switch_tokens) >= 2:
                try:
                    switch_slot = int(switch_tokens[1])
                    switch_key = f"switch_slot_{switch_slot}"
                    if switch_key in counts:
                        counts[switch_key] += 1
                except ValueError:
                    pass
            if forced_slot:
                counts["forced_switches"] += 1
            else:
                counts["voluntary_switches"] += 1
            continue
        if part.startswith("move "):
            move_tokens = part.split()
            if len(move_tokens) >= 2:
                try:
                    move_slot = int(move_tokens[1])
                    move_key = f"move_slot_{move_slot}"
                    if move_key in counts:
                        counts[move_key] += 1
                except ValueError:
                    pass
            move_name = move_name_for_part(request_payload, slot_index, part)
            if move_name in PROTECT_MOVE_NAMES:
                counts["protects"] += 1
            else:
                counts["moves"] += 1


def format_action_counts(prefix: str, counts: dict[str, int]) -> str:
    return (
        f"{prefix}forced_switches={counts['forced_switches']} "
        f"{prefix}voluntary_switches={counts['voluntary_switches']} "
        f"{prefix}moves={counts['moves']} "
        f"{prefix}protects={counts['protects']} "
        f"{prefix}passes={counts['passes']} "
        f"{prefix}teras={counts['teras']}"
    )


def command_uses_tera(command: str) -> bool:
    return " terastallize" in command


def parse_player_line(line: str) -> tuple[str, str] | None:
    parts = line.split("|")
    if len(parts) >= 4 and parts[1] == "player":
        return parts[2], parts[3].strip()
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
        return parts[2], parts[3].strip(), rating
    return None


def identify_bot_side(players: dict[str, str], current_username: str) -> str | None:
    if not current_username:
        return None
    for side in ("p1", "p2"):
        if players.get(side) == current_username:
            return side
    return None


class MatchStats:
    def __init__(self, replay_path: Path) -> None:
        self._stats_path = replay_path.with_suffix(".stats.txt")
        self.matches_played = 0
        self.wins = 0
        self.earned_wins = 0
        self.losses = 0
        self.draws = 0
        self.max_rating = 0
        self.total_invalid_choices = 0
        self.total_fallbacks = 0
        self.total_accepted_proposals = 0
        self.total_forced_switches = 0
        self.total_voluntary_switches = 0
        self.total_moves = 0
        self.total_protects = 0
        self.total_passes = 0
        self.total_teras = 0
        self.tera_battles = 0
        self.total_move_slot_1 = 0
        self.total_move_slot_2 = 0
        self.total_move_slot_3 = 0
        self.total_move_slot_4 = 0
        self.total_switch_slot_1 = 0
        self.total_switch_slot_2 = 0
        self.total_switch_slot_3 = 0
        self.total_switch_slot_4 = 0
        self.total_switch_slot_5 = 0
        self.total_switch_slot_6 = 0
        self._avg_turns_until_tera = 0.0
        self._load()

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}-{self.draws}"

    @property
    def avg_turns_until_tera(self) -> float:
        return self._avg_turns_until_tera

    def _load(self) -> None:
        if not self._stats_path.exists():
            return
        for line in self._stats_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            stripped = value.strip()
            try:
                if key == "matches_played":
                    self.matches_played = int(stripped)
                elif key == "wins":
                    self.wins = int(stripped)
                elif key == "earned_wins":
                    self.earned_wins = int(stripped)
                elif key == "losses":
                    self.losses = int(stripped)
                elif key == "draws":
                    self.draws = int(stripped)
                elif key == "max_rating":
                    self.max_rating = int(stripped)
                elif key == "total_invalid_choices":
                    self.total_invalid_choices = int(stripped)
                elif key == "total_fallbacks":
                    self.total_fallbacks = int(stripped)
                elif key == "total_accepted_proposals":
                    self.total_accepted_proposals = int(stripped)
                elif key == "total_forced_switches":
                    self.total_forced_switches = int(stripped)
                elif key == "total_voluntary_switches":
                    self.total_voluntary_switches = int(stripped)
                elif key == "total_moves":
                    self.total_moves = int(stripped)
                elif key == "total_protects":
                    self.total_protects = int(stripped)
                elif key == "total_passes":
                    self.total_passes = int(stripped)
                elif key == "total_teras":
                    self.total_teras = int(stripped)
                elif key == "tera_battles":
                    self.tera_battles = int(stripped)
                elif key == "total_move_slot_1":
                    self.total_move_slot_1 = int(stripped)
                elif key == "total_move_slot_2":
                    self.total_move_slot_2 = int(stripped)
                elif key == "total_move_slot_3":
                    self.total_move_slot_3 = int(stripped)
                elif key == "total_move_slot_4":
                    self.total_move_slot_4 = int(stripped)
                elif key == "total_switch_slot_1":
                    self.total_switch_slot_1 = int(stripped)
                elif key == "total_switch_slot_2":
                    self.total_switch_slot_2 = int(stripped)
                elif key == "total_switch_slot_3":
                    self.total_switch_slot_3 = int(stripped)
                elif key == "total_switch_slot_4":
                    self.total_switch_slot_4 = int(stripped)
                elif key == "total_switch_slot_5":
                    self.total_switch_slot_5 = int(stripped)
                elif key == "total_switch_slot_6":
                    self.total_switch_slot_6 = int(stripped)
                elif key == "avg_turns_until_tera":
                    self._avg_turns_until_tera = float(stripped)
            except ValueError:
                continue

    def save(self) -> None:
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats_path.write_text(
            "\n".join(
                [
                    f"matches_played={self.matches_played}",
                    f"record={self.record}",
                    f"wins={self.wins}",
                    f"earned_wins={self.earned_wins}",
                    f"losses={self.losses}",
                    f"draws={self.draws}",
                    f"max_rating={self.max_rating}",
                    f"total_invalid_choices={self.total_invalid_choices}",
                    f"total_fallbacks={self.total_fallbacks}",
                    f"total_accepted_proposals={self.total_accepted_proposals}",
                    f"total_forced_switches={self.total_forced_switches}",
                    f"total_voluntary_switches={self.total_voluntary_switches}",
                    f"total_moves={self.total_moves}",
                    f"total_protects={self.total_protects}",
                    f"total_passes={self.total_passes}",
                    f"total_teras={self.total_teras}",
                    f"tera_battles={self.tera_battles}",
                    f"total_move_slot_1={self.total_move_slot_1}",
                    f"total_move_slot_2={self.total_move_slot_2}",
                    f"total_move_slot_3={self.total_move_slot_3}",
                    f"total_move_slot_4={self.total_move_slot_4}",
                    f"total_switch_slot_1={self.total_switch_slot_1}",
                    f"total_switch_slot_2={self.total_switch_slot_2}",
                    f"total_switch_slot_3={self.total_switch_slot_3}",
                    f"total_switch_slot_4={self.total_switch_slot_4}",
                    f"total_switch_slot_5={self.total_switch_slot_5}",
                    f"total_switch_slot_6={self.total_switch_slot_6}",
                    f"avg_turns_until_tera={self.avg_turns_until_tera:.3f}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def note_battle(self, result: str, bot_rating: int | None, earned_win: bool = False) -> None:
        self.matches_played += 1
        if result == "win":
            self.wins += 1
            if earned_win:
                self.earned_wins += 1
        elif result == "draw":
            self.draws += 1
        else:
            self.losses += 1
        if bot_rating is not None:
            self.max_rating = max(self.max_rating, bot_rating)
        self.save()

    def note_live_totals(
        self,
        invalid_choices: int,
        fallbacks: int,
        accepted_proposals: int,
        action_counts: dict[str, int],
        first_tera_turn: int | None,
    ) -> None:
        self.total_invalid_choices = invalid_choices
        self.total_fallbacks = fallbacks
        self.total_accepted_proposals = accepted_proposals
        self.total_forced_switches = action_counts.get("forced_switches", 0)
        self.total_voluntary_switches = action_counts.get("voluntary_switches", 0)
        self.total_moves = action_counts.get("moves", 0)
        self.total_protects = action_counts.get("protects", 0)
        self.total_passes = action_counts.get("passes", 0)
        self.total_teras = action_counts.get("teras", 0)
        self.total_move_slot_1 = action_counts.get("move_slot_1", 0)
        self.total_move_slot_2 = action_counts.get("move_slot_2", 0)
        self.total_move_slot_3 = action_counts.get("move_slot_3", 0)
        self.total_move_slot_4 = action_counts.get("move_slot_4", 0)
        self.total_switch_slot_1 = action_counts.get("switch_slot_1", 0)
        self.total_switch_slot_2 = action_counts.get("switch_slot_2", 0)
        self.total_switch_slot_3 = action_counts.get("switch_slot_3", 0)
        self.total_switch_slot_4 = action_counts.get("switch_slot_4", 0)
        self.total_switch_slot_5 = action_counts.get("switch_slot_5", 0)
        self.total_switch_slot_6 = action_counts.get("switch_slot_6", 0)
        if first_tera_turn is not None:
            total_tera_turns = (self._avg_turns_until_tera * self.tera_battles) + first_tera_turn
            self.tera_battles += 1
            self._avg_turns_until_tera = total_tera_turns / self.tera_battles
        self.save()


MAX_INVALID_RETRIES_PER_REQUEST = 24


class ReplayWriter:
    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path

    def append(self, payload: str) -> None:
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


class BufferedReplayWriter:
    def __init__(self, out_path: Path) -> None:
        self._writer = ReplayWriter(out_path)
        self._pending_by_battle: dict[str, list[str]] = {}

    def append(self, battle_id: str, payload: str) -> None:
        self._pending_by_battle.setdefault(battle_id, []).append(payload)

    def commit_battle(self, battle_id: str) -> None:
        for payload in self._pending_by_battle.pop(battle_id, []):
            self._writer.append(payload)

    def discard_battle(self, battle_id: str) -> None:
        self._pending_by_battle.pop(battle_id, None)

    def discard_all(self, battle_ids: set[str]) -> None:
        for battle_id in list(battle_ids):
            self.discard_battle(battle_id)


async def capture_mode(
    out_path: Path,
    fmt: str,
    username: str,
    server_uri: str,
    shutdown_file: Path | None = None,
    max_games: int | None = None,
    reconnect_seconds: float = DEFAULT_RECONNECT_SECONDS,
    guest_refresh_seconds: float = DEFAULT_GUEST_REFRESH_SECONDS,
) -> None:
    gateway: ShowdownGateway | None = None
    writer = BufferedReplayWriter(out_path)
    stats = MatchStats(out_path)
    started_battles: set[str] = set()
    disconnect_or_forfeit_end: dict[str, bool] = {}
    finished_battles = 0
    seq = 0
    stop_requested = False
    shutdown_after_current = False
    refresh_requested = False
    session_started_at: float | None = None

    print(f"[capture] writing replay records to {out_path}")

    async def request_shutdown() -> None:
        nonlocal stop_requested, shutdown_after_current
        if gateway is not None:
            await gateway.cancel_search()
        if started_battles:
            if not shutdown_after_current:
                shutdown_after_current = True
                print("[capture] shutdown requested; will stop after current battles finish")
            return
        print("[capture] shutdown requested with no active battle; stopping now")
        stop_requested = True
        if gateway is not None:
            await gateway.close()

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal finished_battles, seq, stop_requested, shutdown_after_current, refresh_requested

        if event.room_id.startswith("battle-"):
            if event.room_id not in started_battles:
                started_battles.add(event.room_id)
                disconnect_or_forfeit_end[event.room_id] = False
                writer.append(
                    event.room_id,
                    battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).to_json(),
                )
                print(f"[capture] started {event.room_id}")
                assert gateway is not None
                await enable_battle_timer(gateway, event.room_id, "capture")

        summary = event_summary(event)
        if summary:
            print(summary)

        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            writer.append(event.room_id, request_message(event.room_id, seq, json.loads(payload)).to_json())
            print(f"[capture] request {seq} for {battle_label(event.room_id)}")
        elif is_disconnect_or_forfeit_end(event.line):
            disconnect_or_forfeit_end[event.room_id] = True
            writer.append(event.room_id, event_message(event.room_id, seq, event.line).to_json())
        elif event.line.startswith("|win|") or event.line.startswith("|tie|"):
            result = "win" if event.line.startswith("|win|") else "draw"
            reward = 1.0 if result == "win" else 0.0
            disconnected_or_forfeited = disconnect_or_forfeit_end.get(event.room_id, False)
            if disconnected_or_forfeited:
                reward = 0.0
            writer.append(event.room_id, terminal_message(event.room_id, result, reward).to_json())
            writer.append(event.room_id, battle_end(event.room_id).to_json())
            writer.commit_battle(event.room_id)
            finished_battles += 1
            stats.note_battle(result, None, earned_win=(result == "win" and not disconnected_or_forfeited))
            print(f"[capture] finished {event.room_id} result={result} total_matches_captured={finished_battles}")
            started_battles.discard(event.room_id)
            disconnect_or_forfeit_end.pop(event.room_id, None)
            if max_games is not None and finished_battles >= max_games:
                print(f"[capture] reached max games ({max_games}), stopping")
                stop_requested = True
                assert gateway is not None
                await gateway.close()
            elif shutdown_after_current:
                print("[capture] current battle finished; stopping")
                stop_requested = True
                assert gateway is not None
                await gateway.close()
            elif should_refresh_guest(username, guest_refresh_seconds, session_started_at):
                print(f"[capture] refreshing guest session after {guest_refresh_seconds:.1f}s")
                refresh_requested = True
                assert gateway is not None
                await gateway.close()
            else:
                assert gateway is not None
                await gateway.search_next_battle()
        else:
            writer.append(event.room_id, event_message(event.room_id, seq, event.line).to_json())

        seq += 1

    async def shutdown_watcher() -> None:
        while not stop_requested:
            if shutdown_file_requested(shutdown_file):
                await request_shutdown()
                return
            await asyncio.sleep(1.0)

    while not stop_requested:
        gateway = ShowdownGateway(server_uri, username=username, fmt=fmt)
        try:
            await gateway.connect()
            session_started_at = time.monotonic()
            await asyncio.gather(gateway.run(on_event), shutdown_watcher())
            if refresh_requested and not stop_requested:
                refresh_requested = False
                print("[capture] guest session refreshed; reconnecting now")
                continue
            if not stop_requested:
                print("[capture] gateway closed cleanly")
                break
        except (OSError, ConnectionClosed, asyncio.TimeoutError, TimeoutError) as exc:
            if stop_requested or shutdown_after_current:
                break
            writer.discard_all(started_battles)
            started_battles.clear()
            disconnect_or_forfeit_end.clear()
            print(f"[capture] connection lost: {exc}; reconnecting in {reconnect_seconds:.1f}s")
            await asyncio.sleep(reconnect_seconds)


async def random_mode(
    replay_path: Path,
    fmt: str,
    username: str,
    server_uri: str,
    challenge_target: str = "",
    accept_challenges: bool = False,
    accept_challenge_from: str = "",
    shutdown_file: Path | None = None,
    max_games: int | None = None,
    reconnect_seconds: float = DEFAULT_RECONNECT_SECONDS,
    guest_refresh_seconds: float = DEFAULT_GUEST_REFRESH_SECONDS,
) -> None:
    gateway: ShowdownGateway | None = None
    reward_config = load_flat_toml_values(DEFAULT_REWARD_WEIGHTS_PATH)
    reward_win = reward_float(reward_config, "terminal_win", 1.0)
    reward_loss = reward_float(reward_config, "terminal_loss", -1.0)
    reward_draw = reward_float(reward_config, "terminal_draw", 0.0)
    reward_disconnect_or_forfeit = reward_float(reward_config, "terminal_disconnect_or_forfeit", 0.0)
    writer = BufferedReplayWriter(replay_path)
    stats = MatchStats(replay_path)
    seq = 0
    latest_requests: dict[str, dict] = {}
    latest_request_identity: dict[str, tuple[int | None, str]] = {}
    latest_request_seq: dict[str, int] = {}
    attempted_commands: dict[str, set[str]] = {}
    pending_decisions: dict[str, dict] = {}
    request_open: dict[str, bool] = {}
    invalid_retry_count: dict[str, int] = {}
    battle_players: dict[str, dict[str, str]] = {}
    battle_ratings: dict[str, dict[str, int | None]] = {}
    disconnect_or_forfeit_end: dict[str, bool] = {}
    announced_matchups: set[str] = set()
    finished_battles = 0
    invalid_choice_count = 0
    fallback_used_count = 0
    accepted_count = 0
    total_action_counts = empty_action_counts()
    battle_current_turn: dict[str, int] = {}
    battle_first_tera_turn: dict[str, int | None] = {}
    active_battles: set[str] = set()
    stop_after_current_battle = asyncio.Event()
    stop_requested = False
    refresh_requested = False
    session_started_at: float | None = None

    async def request_shutdown() -> None:
        nonlocal stop_requested
        if gateway is not None:
            await gateway.cancel_search()
        if active_battles:
            if not stop_after_current_battle.is_set():
                stop_after_current_battle.set()
                print("[random] shutdown requested; will stop after the current battle finishes")
            return
        print("[random] shutdown requested with no active battle; stopping now")
        stop_requested = True
        if gateway is not None:
            await gateway.close()

    loop = asyncio.get_running_loop()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(_signum: int, _frame: object) -> None:
        if shutdown_file is not None:
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(request_shutdown()))

    signal.signal(signal.SIGINT, handle_sigint)

    async def shutdown_watcher() -> None:
        nonlocal stop_requested
        while not stop_requested:
            if shutdown_file_requested(shutdown_file):
                await request_shutdown()
                return
            await asyncio.sleep(1.0)

    async def send_random_choice(room_id: str, request_payload: dict, initial: bool) -> None:
        nonlocal fallback_used_count
        candidate = weighted_random_command(request_payload, attempted_commands.setdefault(room_id, set()))
        if candidate is None:
            print(
                f"[random] no legal random candidates left for {battle_label(room_id)} "
                f"forceSwitch={request_payload.get('forceSwitch')} legal_switches={legal_switch_indices(request_payload)}"
            )
            request_open[room_id] = False
            return
        if not initial:
            fallback_used_count += 1
        if command_uses_tera(candidate) and battle_first_tera_turn.get(room_id) is None:
            battle_first_tera_turn[room_id] = battle_current_turn.get(room_id, 0)
        tally_command_categories(request_payload, candidate, total_action_counts)
        pending_decisions[room_id] = {
            "request_id": latest_request_seq.get(room_id, seq),
            "action": -1,
            "action2": -1,
            "command": candidate,
        }
        print(f"[random] action for {battle_label(room_id)}: {candidate}")
        await randomized_send_delay(
            THINK_DELAY_MIN_SECONDS if initial else FALLBACK_DELAY_MIN_SECONDS,
            THINK_DELAY_MAX_SECONDS if initial else FALLBACK_DELAY_MAX_SECONDS,
        )
        assert gateway is not None
        await gateway.send_room_command(room_id, candidate)

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal seq, finished_battles, invalid_choice_count, accepted_count, stop_requested, refresh_requested
        if event.room_id.startswith("battle-"):
            newly_active = event.room_id not in active_battles
            active_battles.add(event.room_id)
            disconnect_or_forfeit_end.setdefault(event.room_id, False)
            if newly_active:
                battle_current_turn[event.room_id] = 0
                battle_first_tera_turn[event.room_id] = None
                writer.append(
                    event.room_id,
                    battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).to_json(),
                )
                print(f"[random] battle started {event.room_id}")
                assert gateway is not None
                await enable_battle_timer(gateway, event.room_id, "random")
        summary = event_summary(event)
        if summary:
            print(summary)
        if (
            event.room_id in pending_decisions
            and not event.line.startswith("|error|")
            and not event.line.startswith("|request|")
        ):
            pending = pending_decisions.pop(event.room_id)
            accepted_count += 1
            writer.append(
                event.room_id,
                action_taken_message(
                    event.room_id,
                    pending["request_id"],
                    pending["action"],
                    pending.get("action2", -1),
                    pending["command"],
                ).to_json(),
            )
        player_info = parse_player_metadata(event.line)
        if player_info is not None:
            side, username_found, rating = player_info
            players = battle_players.setdefault(event.room_id, {})
            ratings = battle_ratings.setdefault(event.room_id, {})
            players[side] = username_found
            ratings[side] = rating
            if event.room_id not in announced_matchups and "p1" in players and "p2" in players:
                print()
                print(
                    f"[stats] matches={stats.matches_played} record={stats.record} earned_wins={stats.earned_wins} max_rating={stats.max_rating} "
                    f"moves={stats.total_moves} protects={stats.total_protects} "
                    f"passes={stats.total_passes} teras={stats.total_teras} "
                    f"avg_turns_until_tera={stats.avg_turns_until_tera:.2f}"
                )
                print(
                    f"[matchup] {players['p1']} ({ratings.get('p1', '?') if ratings.get('p1') is not None else '?'}) vs "
                    f"{players['p2']} ({ratings.get('p2', '?') if ratings.get('p2') is not None else '?'})"
                )
                announced_matchups.add(event.room_id)
        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            request_payload = json.loads(payload)
            request_identity = (
                request_payload.get("rqid") if isinstance(request_payload, dict) else None,
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")),
            )
            if request_open.get(event.room_id, False) and latest_request_identity.get(event.room_id) == request_identity:
                print(f"[random] duplicate request ignored for {battle_label(event.room_id)}")
                return
            latest_requests[event.room_id] = request_payload
            latest_request_identity[event.room_id] = request_identity
            latest_request_seq[event.room_id] = seq
            attempted_commands[event.room_id] = set()
            request_open[event.room_id] = True
            invalid_retry_count[event.room_id] = 0
            writer.append(event.room_id, request_message(event.room_id, seq, request_payload).to_json())
            print(f"[random] request {seq} for {battle_label(event.room_id)}")
            await send_random_choice(event.room_id, request_payload, initial=True)
        elif event.line.startswith("|error|") and "Invalid choice" in event.line:
            invalid_choice_count += 1
            pending = pending_decisions.pop(event.room_id, None)
            if pending is not None:
                writer.append(
                    event.room_id,
                    action_rejected_message(
                        event.room_id,
                        pending["request_id"],
                        pending["action"],
                        pending.get("action2", -1),
                        pending["command"],
                        reason=event.line,
                    ).to_json(),
                )
            request_payload = latest_requests.get(event.room_id)
            if "There's nothing to choose" in event.line:
                request_open[event.room_id] = False
                print(f"[random] request closed for {battle_label(event.room_id)} after stale choice rejection")
                return
            if request_payload is not None and request_open.get(event.room_id, False):
                invalid_retry_count[event.room_id] = invalid_retry_count.get(event.room_id, 0) + 1
                if invalid_retry_count[event.room_id] > MAX_INVALID_RETRIES_PER_REQUEST:
                    request_open[event.room_id] = False
                    print(f"[random] request closed for {battle_label(event.room_id)} after too many invalid retries")
                    return
                await send_random_choice(event.room_id, request_payload, initial=False)
        elif event.line.startswith("|win|") or event.line.startswith("|tie|"):
            players = battle_players.get(event.room_id, {})
            ratings = battle_ratings.get(event.room_id, {})
            winner_name = event.line.split("|", 2)[2] if event.line.startswith("|win|") and "|" in event.line else ""
            bot_side = identify_bot_side(players, gateway.current_username)
            bot_rating = ratings.get(bot_side)
            if event.line.startswith("|tie|"):
                final_result = "draw"
                reward = reward_draw
            elif bot_side is not None and players.get(bot_side) == winner_name:
                final_result = "win"
                reward = reward_win
            else:
                final_result = "loss"
                reward = reward_loss
            disconnected_or_forfeited = disconnect_or_forfeit_end.get(event.room_id, False)
            if disconnected_or_forfeited:
                reward = reward_disconnect_or_forfeit
            writer.append(event.room_id, terminal_message(event.room_id, final_result, reward).to_json())
            writer.append(event.room_id, battle_end(event.room_id).to_json())
            writer.commit_battle(event.room_id)
            stats.note_battle(
                final_result,
                bot_rating,
                earned_win=(final_result == "win" and not disconnected_or_forfeited),
            )
            stats.note_live_totals(
                invalid_choice_count,
                fallback_used_count,
                accepted_count,
                total_action_counts,
                battle_first_tera_turn.pop(event.room_id, None),
            )
            print(f"[random] battle ended {event.room_id} result={final_result}")
            request_open[event.room_id] = False
            latest_request_identity.pop(event.room_id, None)
            latest_request_seq.pop(event.room_id, None)
            pending_decisions.pop(event.room_id, None)
            disconnect_or_forfeit_end.pop(event.room_id, None)
            battle_current_turn.pop(event.room_id, None)
            active_battles.discard(event.room_id)
            finished_battles += 1
            if max_games is not None and finished_battles >= max_games:
                print(f"[random] reached max games ({max_games}), stopping")
                stop_requested = True
                assert gateway is not None
                await gateway.close()
            elif should_refresh_guest(username, guest_refresh_seconds, session_started_at):
                print(f"[random] refreshing guest session after {guest_refresh_seconds:.1f}s")
                refresh_requested = True
                assert gateway is not None
                await gateway.close()
            elif stop_after_current_battle.is_set():
                print("[random] current battle finished; stopping")
                stop_requested = True
                assert gateway is not None
                await gateway.close()
            else:
                assert gateway is not None
                await gateway.search_next_battle()
        else:
            if is_disconnect_or_forfeit_end(event.line):
                disconnect_or_forfeit_end[event.room_id] = True
            elif event.line.startswith("|turn|"):
                parts = event.line.split("|")
                if len(parts) >= 3:
                    try:
                        battle_current_turn[event.room_id] = int(parts[2])
                    except ValueError:
                        pass
            writer.append(event.room_id, event_message(event.room_id, seq, event.line).to_json())
        seq += 1

    try:
        while not stop_requested:
            gateway = ShowdownGateway(
                server_uri,
                username=username,
                fmt=fmt,
                challenge_target=challenge_target,
                accept_challenges=accept_challenges,
                accept_challenge_from=accept_challenge_from,
            )
            try:
                await gateway.connect()
                session_started_at = time.monotonic()
                await asyncio.gather(gateway.run(on_event), shutdown_watcher())
                if refresh_requested and not stop_requested:
                    refresh_requested = False
                    print("[random] guest session refreshed; reconnecting now")
                    continue
                if not stop_requested:
                    print("[random] gateway closed cleanly")
                    break
            except (OSError, ConnectionClosed, asyncio.TimeoutError, TimeoutError) as exc:
                if stop_requested or stop_after_current_battle.is_set():
                    break
                writer.discard_all(active_battles)
                active_battles.clear()
                latest_requests.clear()
                latest_request_identity.clear()
                latest_request_seq.clear()
                attempted_commands.clear()
                pending_decisions.clear()
                request_open.clear()
                invalid_retry_count.clear()
                disconnect_or_forfeit_end.clear()
                battle_current_turn.clear()
                battle_first_tera_turn.clear()
                print(f"[random] connection lost: {exc}; reconnecting in {reconnect_seconds:.1f}s")
                await asyncio.sleep(reconnect_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


async def live_mode(
    learner_command: list[str],
    replay_path: Path | None,
    fmt: str,
    username: str,
    server_uri: str,
    challenge_target: str = "",
    accept_challenges: bool = False,
    accept_challenge_from: str = "",
    shutdown_file: Path | None = None,
    max_games: int | None = None,
    reconnect_seconds: float = DEFAULT_RECONNECT_SECONDS,
    guest_refresh_seconds: float = DEFAULT_GUEST_REFRESH_SECONDS,
) -> None:
    gateway: ShowdownGateway | None = None
    learner: LearnerProcess | None = None
    reward_config = load_flat_toml_values(DEFAULT_REWARD_WEIGHTS_PATH)
    reward_win = reward_float(reward_config, "terminal_win", 1.0)
    reward_loss = reward_float(reward_config, "terminal_loss", -1.0)
    reward_draw = reward_float(reward_config, "terminal_draw", 0.0)
    reward_disconnect_or_forfeit = reward_float(reward_config, "terminal_disconnect_or_forfeit", 0.0)
    writer = BufferedReplayWriter(replay_path) if replay_path is not None else None
    stats = MatchStats(replay_path or Path("matches/runtime_capture.jsonl"))
    seq = 0
    latest_requests: dict[str, dict] = {}
    latest_request_identity: dict[str, tuple[int | None, str]] = {}
    attempted_commands: dict[str, set[str]] = {}
    latest_request_seq: dict[str, int] = {}
    request_open: dict[str, bool] = {}
    invalid_retry_count: dict[str, int] = {}
    learner_rerolls: dict[str, int] = {}
    pending_decisions: dict[str, dict] = {}
    battle_players: dict[str, dict[str, str]] = {}
    battle_ratings: dict[str, dict[str, int | None]] = {}
    disconnect_or_forfeit_end: dict[str, bool] = {}
    announced_matchups: set[str] = set()
    invalid_choice_count = 0
    fallback_used_count = 0
    learner_proposal_accepted_count = 0
    finished_battles = 0
    total_action_counts = empty_action_counts()
    battle_current_turn: dict[str, int] = {}
    battle_first_tera_turn: dict[str, int | None] = {}
    battle_invalid_choice_count: dict[str, int] = {}
    battle_fallback_used_count: dict[str, int] = {}
    battle_accepted_proposal_count: dict[str, int] = {}
    battle_action_counts: dict[str, dict[str, int]] = {}
    active_battles: set[str] = set()
    stop_after_current_battle = asyncio.Event()
    learner_stopping = asyncio.Event()
    stop_requested = False
    refresh_requested = False
    session_started_at: float | None = None

    async def request_shutdown() -> None:
        nonlocal stop_requested
        if gateway is not None:
            await gateway.cancel_search()
        if active_battles:
            if not stop_after_current_battle.is_set():
                stop_after_current_battle.set()
                print("[live] shutdown requested; will stop after the current battle finishes")
            return
        print("[live] shutdown requested with no active battle; stopping now")
        stop_requested = True
        learner_stopping.set()
        if gateway is not None:
            await gateway.close()
        if learner is not None:
            await learner.terminate()

    loop = asyncio.get_running_loop()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(_signum: int, _frame: object) -> None:
        if shutdown_file is not None:
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(request_shutdown()))

    signal.signal(signal.SIGINT, handle_sigint)

    async def shutdown_watcher() -> None:
        while not stop_requested:
            if shutdown_file_requested(shutdown_file):
                await request_shutdown()
                return
            await asyncio.sleep(1.0)

    async def on_event(event: ShowdownEvent) -> None:
        nonlocal seq, invalid_choice_count, fallback_used_count, learner_proposal_accepted_count, finished_battles, stop_requested, refresh_requested
        if event.room_id.startswith("battle-"):
            newly_active = event.room_id not in active_battles
            active_battles.add(event.room_id)
            disconnect_or_forfeit_end.setdefault(event.room_id, False)
            if newly_active:
                battle_current_turn[event.room_id] = 0
                battle_first_tera_turn[event.room_id] = None
                battle_invalid_choice_count[event.room_id] = 0
                battle_fallback_used_count[event.room_id] = 0
                battle_accepted_proposal_count[event.room_id] = 0
                battle_action_counts[event.room_id] = empty_action_counts()
                if writer is not None:
                    writer.append(
                        event.room_id,
                        battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).to_json(),
                    )
                assert learner is not None
                await learner.send(
                    battle_start(event.room_id, fmt, infer_is_doubles(event.room_id, fmt)).payload
                )
                print(f"[live] battle started {event.room_id}")
                assert gateway is not None
                await enable_battle_timer(gateway, event.room_id, "live")
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
                print(
                    f"[stats] matches={stats.matches_played} record={stats.record} earned_wins={stats.earned_wins} max_rating={stats.max_rating} "
                    f"moves={stats.total_moves} protects={stats.total_protects} "
                    f"passes={stats.total_passes} teras={stats.total_teras} "
                    f"avg_turns_until_tera={stats.avg_turns_until_tera:.2f}"
                )
                print(
                    f"[matchup] {players['p1']} ({ratings.get('p1', '?') if ratings.get('p1') is not None else '?'}) vs "
                    f"{players['p2']} ({ratings.get('p2', '?') if ratings.get('p2') is not None else '?'})"
                )
                announced_matchups.add(event.room_id)
        if (
            event.room_id in pending_decisions
            and not event.line.startswith("|error|")
            and not event.line.startswith("|request|")
        ):
            pending = pending_decisions.pop(event.room_id)
            learner_proposal_accepted_count += 1
            battle_accepted_proposal_count[event.room_id] = battle_accepted_proposal_count.get(event.room_id, 0) + 1
            if writer is not None:
                writer.append(
                    event.room_id,
                    action_taken_message(
                        event.room_id,
                        pending["request_id"],
                        pending["action"],
                        pending.get("action2", -1),
                        pending["command"],
                    ).to_json(),
                )
            await learner.send(
                decision_message(
                    event.room_id,
                    pending["request_id"],
                    pending["action"],
                    pending.get("action2", -1),
                    pending["command"],
                    accepted=True,
                ).payload
            )
        if event.line.startswith("|request|"):
            payload = event.line.split("|request|", 1)[1]
            request_payload = json.loads(payload)
            request_identity = (
                request_payload.get("rqid") if isinstance(request_payload, dict) else None,
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")),
            )
            if request_open.get(event.room_id, False) and latest_request_identity.get(event.room_id) == request_identity:
                print(f"[live] duplicate request ignored for {battle_label(event.room_id)}")
                return
            pending = pending_decisions.pop(event.room_id, None)
            if pending is not None:
                learner_proposal_accepted_count += 1
                battle_accepted_proposal_count[event.room_id] = battle_accepted_proposal_count.get(event.room_id, 0) + 1
                if writer is not None:
                    writer.append(
                        event.room_id,
                        action_taken_message(
                            event.room_id,
                            pending["request_id"],
                            pending["action"],
                            pending.get("action2", -1),
                            pending["command"],
                        ).to_json(),
                    )
                await learner.send(
                    decision_message(
                        event.room_id,
                        pending["request_id"],
                        pending["action"],
                        pending.get("action2", -1),
                        pending["command"],
                        accepted=True,
                    ).payload
                )
            latest_requests[event.room_id] = request_payload
            latest_request_identity[event.room_id] = request_identity
            attempted_commands[event.room_id] = set()
            latest_request_seq[event.room_id] = seq
            request_open[event.room_id] = True
            invalid_retry_count[event.room_id] = 0
            learner_rerolls[event.room_id] = 0
            print(f"[live] sending request {seq} to learner for {battle_label(event.room_id)}")
            if writer is not None:
                writer.append(event.room_id, request_message(event.room_id, seq, request_payload).to_json())
            assert learner is not None
            await learner.send(request_message(event.room_id, seq, request_payload).payload)
            print(f"[live] request {seq} for {battle_label(event.room_id)}")
        elif event.line.startswith("|error|") and "Invalid choice" in event.line:
            invalid_choice_count += 1
            battle_invalid_choice_count[event.room_id] = battle_invalid_choice_count.get(event.room_id, 0) + 1
            pending = pending_decisions.pop(event.room_id, None)
            if pending is not None:
                if writer is not None:
                    writer.append(
                        event.room_id,
                        action_rejected_message(
                            event.room_id,
                            pending["request_id"],
                            pending["action"],
                            pending.get("action2", -1),
                            pending["command"],
                            reason=event.line,
                        ).to_json(),
                    )
                await learner.send(
                    decision_message(
                        event.room_id,
                        pending["request_id"],
                        pending["action"],
                        pending.get("action2", -1),
                        pending["command"],
                        accepted=False,
                        reason=event.line,
                    ).payload
                )
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
                        fallback_used_count += 1
                        battle_fallback_used_count[event.room_id] = battle_fallback_used_count.get(event.room_id, 0) + 1
                        print(f"[live] retrying with fallback for {battle_label(event.room_id)}: {candidate}")
                        tally_command_categories(
                            request_payload,
                            candidate,
                            battle_action_counts.setdefault(event.room_id, empty_action_counts()),
                        )
                        tally_command_categories(request_payload, candidate, total_action_counts)
                        pending_decisions[event.room_id] = {
                            "request_id": latest_request_seq.get(event.room_id, seq),
                            "action": -1,
                            "action2": -1,
                            "command": candidate,
                        }
                        if writer is not None:
                            writer.append(
                                event.room_id,
                                decision_message(
                                    event.room_id,
                                    latest_request_seq.get(event.room_id, seq),
                                    -1,
                                    -1,
                                    candidate,
                                ).to_json(),
                            )
                        assert learner is not None
                        await learner.send(
                            decision_message(
                                event.room_id,
                                latest_request_seq.get(event.room_id, seq),
                                -1,
                                -1,
                                candidate,
                            ).payload
                        )
                        await randomized_send_delay(FALLBACK_DELAY_MIN_SECONDS, FALLBACK_DELAY_MAX_SECONDS)
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
            players = battle_players.get(event.room_id, {})
            ratings = battle_ratings.get(event.room_id, {})
            winner_name = event.line.split("|", 2)[2] if event.line.startswith("|win|") and "|" in event.line else ""
            bot_side = identify_bot_side(players, gateway.current_username)
            bot_rating = ratings.get(bot_side)
            if event.line.startswith("|tie|"):
                final_result = "draw"
                reward = reward_draw
            elif bot_side is not None and players.get(bot_side) == winner_name:
                final_result = "win"
                reward = reward_win
            else:
                final_result = "loss"
                reward = reward_loss
            disconnected_or_forfeited = disconnect_or_forfeit_end.get(event.room_id, False)
            if disconnected_or_forfeited:
                reward = reward_disconnect_or_forfeit
            if writer is not None:
                writer.append(event.room_id, terminal_message(event.room_id, final_result, reward).to_json())
                writer.append(event.room_id, battle_end(event.room_id).to_json())
                writer.commit_battle(event.room_id)
            assert learner is not None
            await learner.send(terminal_message(event.room_id, final_result, reward).payload)
            await learner.send(battle_end(event.room_id).payload)
            stats.note_battle(
                final_result,
                bot_rating,
                earned_win=(final_result == "win" and not disconnected_or_forfeited),
            )
            stats.note_live_totals(
                invalid_choice_count,
                fallback_used_count,
                learner_proposal_accepted_count,
                total_action_counts,
                battle_first_tera_turn.pop(event.room_id, None),
            )
            print(f"[live] battle ended {event.room_id} result={final_result}")
            battle_invalid = battle_invalid_choice_count.pop(event.room_id, 0)
            battle_fallbacks = battle_fallback_used_count.pop(event.room_id, 0)
            battle_accepted = battle_accepted_proposal_count.pop(event.room_id, 0)
            battle_actions = battle_action_counts.pop(event.room_id, empty_action_counts())
            total_resolved = learner_proposal_accepted_count + fallback_used_count
            accept_rate = (learner_proposal_accepted_count / total_resolved) if total_resolved > 0 else 0.0
            battle_total_resolved = battle_accepted + battle_fallbacks
            battle_accept_rate = (battle_accepted / battle_total_resolved) if battle_total_resolved > 0 else 0.0
            print(
                f"[live] legality battle_invalid_choices={battle_invalid} "
                f"battle_fallbacks={battle_fallbacks} battle_accepted_proposals={battle_accepted} "
                f"battle_accept_rate={battle_accept_rate:.3f} "
                f"total_invalid_choices={invalid_choice_count} total_fallbacks={fallback_used_count} "
                f"total_accepted_proposals={learner_proposal_accepted_count} total_accept_rate={accept_rate:.3f}"
            )
            print(
                f"[live] action mix {format_action_counts('battle_', battle_actions)} "
                f"{format_action_counts('total_', total_action_counts)}"
            )
            request_open[event.room_id] = False
            latest_request_identity.pop(event.room_id, None)
            disconnect_or_forfeit_end.pop(event.room_id, None)
            battle_current_turn.pop(event.room_id, None)
            active_battles.discard(event.room_id)
            finished_battles += 1
            if max_games is not None and finished_battles >= max_games:
                print(f"[live] reached max games ({max_games}), stopping")
                stop_requested = True
                learner_stopping.set()
                assert gateway is not None
                await gateway.close()
                assert learner is not None
                await learner.terminate()
            elif should_refresh_guest(username, guest_refresh_seconds, session_started_at):
                print(f"[live] refreshing guest session after {guest_refresh_seconds:.1f}s")
                refresh_requested = True
                learner_stopping.set()
                assert gateway is not None
                await gateway.close()
                assert learner is not None
                await learner.terminate()
            elif stop_after_current_battle.is_set():
                print("[live] current battle finished; stopping")
                stop_requested = True
                learner_stopping.set()
                assert gateway is not None
                await gateway.close()
                assert learner is not None
                await learner.terminate()
            else:
                assert gateway is not None
                await gateway.search_next_battle()
        else:
            if is_disconnect_or_forfeit_end(event.line):
                disconnect_or_forfeit_end[event.room_id] = True
            elif event.line.startswith("|turn|"):
                parts = event.line.split("|")
                if len(parts) >= 3:
                    try:
                        battle_current_turn[event.room_id] = int(parts[2])
                    except ValueError:
                        pass
            if writer is not None:
                writer.append(event.room_id, event_message(event.room_id, seq, event.line).to_json())
            if not learner_stopping.is_set():
                assert learner is not None
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
            elif msg.get("type") == "episode_complete":
                battle_id = str(msg.get("battle_id") or "")
                if writer is not None and battle_id:
                    writer.append(battle_id, json.dumps(msg, separators=(",", ":")))
                print(f"[live] episode complete for {battle_id} steps={msg.get('count', 0)}")
            elif msg.get("type") == "action":
                battle_id = msg["battle_id"]
                if not request_open.get(battle_id, False):
                    print(f"[live] ignoring stale learner action for closed request {battle_id}: {msg['command']}")
                    continue
                print(f"[live] action for {battle_id}: {msg['command']}")
                attempted_commands.setdefault(battle_id, set()).add(msg["command"])
                request_payload = latest_requests.get(battle_id)
                if request_payload is not None:
                    if command_uses_tera(msg["command"]) and battle_first_tera_turn.get(battle_id) is None:
                        battle_first_tera_turn[battle_id] = battle_current_turn.get(battle_id, 0)
                    tally_command_categories(
                        request_payload,
                        msg["command"],
                        battle_action_counts.setdefault(battle_id, empty_action_counts()),
                    )
                    tally_command_categories(request_payload, msg["command"], total_action_counts)
                if writer is not None:
                    writer.append(
                        battle_id,
                        decision_message(
                            battle_id,
                            msg["request_id"],
                            msg.get("action", -1),
                            msg.get("action2", -1),
                            msg["command"],
                        ).to_json(),
                    )
                pending_decisions[battle_id] = {
                    "request_id": msg["request_id"],
                    "action": msg.get("action", -1),
                    "action2": msg.get("action2", -1),
                    "command": msg["command"],
                }
                await learner.send(
                    decision_message(
                        battle_id,
                        msg["request_id"],
                        msg.get("action", -1),
                        msg.get("action2", -1),
                        msg["command"],
                    ).payload
                )
                await randomized_send_delay(THINK_DELAY_MIN_SECONDS, THINK_DELAY_MAX_SECONDS)
                assert gateway is not None
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

    try:
        while not stop_requested:
            gateway = ShowdownGateway(
                server_uri,
                username=username,
                fmt=fmt,
                challenge_target=challenge_target,
                accept_challenges=accept_challenges,
                accept_challenge_from=accept_challenge_from,
            )
            learner = LearnerProcess(learner_command, None)
            await learner.start()
            print(f"[live] started learner: {' '.join(learner_command)}")
            try:
                await gateway.connect()
                session_started_at = time.monotonic()
                await asyncio.gather(gateway.run(on_event), action_loop(), stderr_loop(), shutdown_watcher())
                if refresh_requested and not stop_requested:
                    refresh_requested = False
                    learner_stopping.clear()
                    print("[live] guest session refreshed; reconnecting now")
                    continue
                if not stop_requested:
                    print("[live] gateway/learner session ended cleanly")
                    break
            except (OSError, ConnectionClosed, asyncio.TimeoutError, TimeoutError) as exc:
                if stop_requested or stop_after_current_battle.is_set():
                    await learner.terminate()
                    break
                if writer is not None:
                    writer.discard_all(active_battles)
                active_battles.clear()
                latest_requests.clear()
                latest_request_identity.clear()
                attempted_commands.clear()
                latest_request_seq.clear()
                request_open.clear()
                invalid_retry_count.clear()
                learner_rerolls.clear()
                pending_decisions.clear()
                disconnect_or_forfeit_end.clear()
                battle_current_turn.clear()
                battle_first_tera_turn.clear()
                battle_invalid_choice_count.clear()
                battle_fallback_used_count.clear()
                battle_accepted_proposal_count.clear()
                battle_action_counts.clear()
                learner_stopping.clear()
                print(f"[live] connection lost: {exc}; reconnecting in {reconnect_seconds:.1f}s")
                await learner.terminate()
                await asyncio.sleep(reconnect_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "capture", "random"], default="live")
    parser.add_argument("--replay-save", default="runtime_capture")
    parser.add_argument("--learner-args", nargs="*", default=[])
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--username", default="")
    parser.add_argument("--server-uri", default="")
    parser.add_argument("--challenge-target", default="")
    parser.add_argument("--accept-challenges", type=int, choices=[0, 1], default=0)
    parser.add_argument("--accept-challenge-from", default="")
    parser.add_argument("--shutdown-file", default="")
    parser.add_argument("--games", type=int, default=0)
    parser.add_argument("--reconnect-seconds", type=float, default=DEFAULT_RECONNECT_SECONDS)
    parser.add_argument("--guest-refresh-seconds", type=float, default=DEFAULT_GUEST_REFRESH_SECONDS)
    argv = sys.argv[1:]
    if not argv:
        argv = load_default_args(DEFAULT_ARGS_PATH)
    args, learner_passthrough = parser.parse_known_args(argv)
    max_games = args.games if args.games > 0 else None
    replay_path = resolve_replay_save_path(args.replay_save)
    server_uri = resolved_server_uri(args.server_uri)
    shutdown_file = Path(args.shutdown_file) if args.shutdown_file else None

    if args.mode == "capture":
        asyncio.run(
            capture_mode(
                replay_path,
                args.format,
                args.username,
                server_uri,
                challenge_target=args.challenge_target,
                accept_challenges=bool(args.accept_challenges),
                accept_challenge_from=args.accept_challenge_from,
                shutdown_file=shutdown_file,
                max_games=max_games,
                reconnect_seconds=args.reconnect_seconds,
                guest_refresh_seconds=args.guest_refresh_seconds,
            )
        )
    elif args.mode == "random":
        asyncio.run(
            random_mode(
                replay_path,
                args.format,
                args.username,
                server_uri,
                challenge_target=args.challenge_target,
                accept_challenges=bool(args.accept_challenges),
                accept_challenge_from=args.accept_challenge_from,
                shutdown_file=shutdown_file,
                max_games=max_games,
                reconnect_seconds=args.reconnect_seconds,
                guest_refresh_seconds=args.guest_refresh_seconds,
            )
        )
    else:
        learner_args = list(args.learner_args) + list(learner_passthrough)
        if not learner_args:
            learner_args = ["--battle-agent"]
        learner_command = [DEFAULT_LEARNER_COMMAND] + learner_args
        asyncio.run(
            live_mode(
                learner_command,
                replay_path,
                args.format,
                args.username,
                server_uri,
                challenge_target=args.challenge_target,
                accept_challenges=bool(args.accept_challenges),
                accept_challenge_from=args.accept_challenge_from,
                shutdown_file=shutdown_file,
                max_games=max_games,
                reconnect_seconds=args.reconnect_seconds,
                guest_refresh_seconds=args.guest_refresh_seconds,
            )
        )


if __name__ == "__main__":
    main()
