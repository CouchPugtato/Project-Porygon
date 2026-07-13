from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("matches") / "runs"


@dataclass
class RequestRecord:
    request_id: int | None
    payload: dict[str, Any]


@dataclass
class ActionRecord:
    request_id: int | None
    action: int | None
    action2: int | None
    command: str
    accepted: bool


@dataclass
class EventRecord:
    seq: int | None
    line: str


@dataclass
class BattleRecord:
    battle_id: str
    source_files: set[str] = field(default_factory=set)
    terminal_result: str | None = None
    reward: float | None = None
    terminal_conflict: bool = False
    requests: list[RequestRecord] = field(default_factory=list)
    actions_taken: list[ActionRecord] = field(default_factory=list)
    actions_rejected: list[ActionRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CommandPart:
    kind: str
    move_slot: int | None = None
    switch_slot: int | None = None
    target: int | None = None
    used_tera: bool = False


@dataclass
class ParsedCommand:
    raw: str
    parts: list[CommandPart]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_bool01(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("value must be 0 or 1")
    return value == "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run name under matches/runs/")
    parser.add_argument("--path", help="Replay jsonl file or directory")
    parser.add_argument("--glob", dest="glob_pattern", help="Glob for replay jsonl files")
    parser.add_argument("--side", choices=["a", "b"], default="a")
    parser.add_argument("--include-outcomes", default="win,loss")
    parser.add_argument(
        "--output",
        help="Optional JSON output path. When omitted with --run, defaults to matches/runs/<run>/<run>_diagnostics.json",
    )
    parser.add_argument("--sample-losses", type=positive_int, default=10)
    parser.add_argument("--min-battles", type=positive_int, default=1)
    parser.add_argument("--quiet", type=parse_bool01, default=False)
    parser.add_argument("--pretty", type=parse_bool01, default=True)
    return parser


def infer_file_side(path: Path) -> str | None:
    name = path.name
    if "_a_raw.jsonl" in name:
        return "a"
    if "_b_raw.jsonl" in name:
        return "b"
    return None


def resolve_output_path(base_dir: Path | None, raw_output: str | None, repo_root: Path) -> Path | None:
    if not raw_output:
        return None
    output = Path(raw_output)
    if output.is_absolute():
        return output
    if base_dir is not None:
        return base_dir / output
    return repo_root / output


def resolve_input_files(args: argparse.Namespace, repo_root: Path) -> tuple[str, list[Path], Path | None]:
    selectors = [bool(args.run), bool(args.path), bool(args.glob_pattern)]
    if sum(selectors) != 1:
        raise SystemExit("exactly one of --run, --path, or --glob must be provided")

    mode = "unknown"
    output_path: Path | None = None
    files: list[Path]
    if args.run:
        mode = "run"
        run_dir = repo_root / DEFAULT_RUNS_ROOT / args.run
        if not run_dir.exists():
            raise SystemExit(f"run dir not found: {run_dir}")
        files = sorted(run_dir.glob("worker_*_raw.jsonl"))
        if not files:
            raise SystemExit(f"no replay files found in {run_dir}")
        if args.output:
            output_path = resolve_output_path(run_dir, args.output, repo_root)
        else:
            output_path = run_dir / f"{args.run}_diagnostics.json"
    elif args.path:
        mode = "path"
        input_path = repo_root / Path(args.path)
        if not input_path.exists():
            raise SystemExit(f"path not found: {input_path}")
        if input_path.is_dir():
            files = sorted(input_path.glob("*.jsonl"))
        else:
            files = [input_path]
        if args.output:
            output_path = resolve_output_path(input_path if input_path.is_dir() else input_path.parent, args.output, repo_root)
    else:
        mode = "glob"
        files = sorted(Path(path) for path in glob.glob(str(repo_root / args.glob_pattern)))
        if args.output:
            output_path = resolve_output_path(None, args.output, repo_root)

    if not files:
        raise SystemExit("no input files matched")

    file_sides = {path: infer_file_side(path) for path in files}
    matching_side_files = [path for path in files if file_sides[path] == args.side]
    if matching_side_files:
        files = matching_side_files
    return mode, files, output_path


def parse_outcome_filter(raw_value: str) -> set[str]:
    allowed = {"win", "loss", "draw"}
    values = {piece.strip().lower() for piece in raw_value.split(",") if piece.strip()}
    if not values:
        raise SystemExit("include-outcomes produced an empty filter")
    invalid = values - allowed
    if invalid:
        raise SystemExit(f"unsupported outcomes in --include-outcomes: {', '.join(sorted(invalid))}")
    return values


def is_int_token(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def parse_choose_command(command: str) -> ParsedCommand:
    text = str(command or "").strip()
    if text.startswith("/choose "):
        text = text[len("/choose ") :]
    parts: list[CommandPart] = []
    if not text:
        return ParsedCommand(raw=command, parts=parts)
    for raw_part in text.split(","):
        tokens = raw_part.strip().split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == "move":
            move_slot: int | None = None
            target: int | None = None
            used_tera = False
            remaining = tokens[1:]
            if remaining and is_int_token(remaining[0]):
                move_slot = int(remaining[0])
                remaining = remaining[1:]
            for token in remaining:
                lowered = token.lower()
                if lowered == "terastallize":
                    used_tera = True
                elif is_int_token(token) and target is None:
                    target = int(token)
            parts.append(CommandPart(kind="move", move_slot=move_slot, target=target, used_tera=used_tera))
        elif head == "switch":
            switch_slot = int(tokens[1]) if len(tokens) >= 2 and is_int_token(tokens[1]) else None
            parts.append(CommandPart(kind="switch", switch_slot=switch_slot))
        elif head == "pass":
            parts.append(CommandPart(kind="pass"))
        else:
            parts.append(CommandPart(kind="unknown"))
    return ParsedCommand(raw=command, parts=parts)


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def median_or_zero(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def mean_or_zero(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def late_start_index(length: int) -> int:
    if length <= 0:
        return 0
    window = max(3, math.ceil(length * 0.25))
    return max(0, length - window)


def battle_bucket(length: int) -> str:
    if length <= 10:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def request_has_force_switch(payload: dict[str, Any]) -> bool:
    raw = payload.get("forceSwitch")
    if isinstance(raw, list):
        return any(bool(value) for value in raw)
    return bool(raw)


def request_active_len(payload: dict[str, Any]) -> int:
    active = payload.get("active")
    if isinstance(active, list):
        return len(active)
    return 0


def self_faint_event(line: str, side: str) -> bool:
    text = str(line or "")
    if not text.startswith("|faint|"):
        return False
    self_prefix = "p1" if side == "a" else "p2"
    return f"|faint|{self_prefix}" in text


def load_battles(files: list[Path]) -> tuple[dict[str, BattleRecord], dict[str, Any]]:
    battles: dict[str, BattleRecord] = {}
    warnings: dict[str, Any] = {
        "malformed_json_lines": 0,
        "missing_battle_id_lines": 0,
        "empty_command_actions": 0,
    }
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    warnings["malformed_json_lines"] += 1
                    continue
                battle_id = payload.get("battle_id")
                if not battle_id:
                    warnings["missing_battle_id_lines"] += 1
                    continue
                battle_id = str(battle_id)
                battle = battles.get(battle_id)
                if battle is None:
                    battle = BattleRecord(battle_id=battle_id)
                    battles[battle_id] = battle
                battle.source_files.add(path.name)
                record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
                if record_type == "request":
                    request_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                    battle.requests.append(
                        RequestRecord(
                            request_id=payload.get("request_id"),
                            payload=request_payload,
                        )
                    )
                    battle.timeline.append({"type": "request", "request_id": payload.get("request_id"), "payload": request_payload})
                elif record_type == "action_taken":
                    command = str(payload.get("command") or "")
                    if not command.strip():
                        warnings["empty_command_actions"] += 1
                    battle.actions_taken.append(
                        ActionRecord(
                            request_id=payload.get("request_id"),
                            action=payload.get("action"),
                            action2=payload.get("action2"),
                            command=command,
                            accepted=True,
                        )
                    )
                    battle.timeline.append({"type": "action_taken", "request_id": payload.get("request_id"), "command": command})
                elif record_type == "action_rejected":
                    command = str(payload.get("command") or "")
                    if not command.strip():
                        warnings["empty_command_actions"] += 1
                    battle.actions_rejected.append(
                        ActionRecord(
                            request_id=payload.get("request_id"),
                            action=payload.get("action"),
                            action2=payload.get("action2"),
                            command=command,
                            accepted=False,
                        )
                    )
                elif record_type == "event":
                    line_text = str(payload.get("line") or "")
                    battle.events.append(EventRecord(seq=payload.get("seq"), line=line_text))
                    battle.timeline.append({"type": "event", "line": line_text})
                elif record_type == "terminal":
                    result = str(payload.get("result") or "").strip().lower()
                    battle.timeline.append({"type": "terminal", "result": result})
                    if battle.terminal_result is None:
                        battle.terminal_result = result
                        reward = payload.get("reward")
                        battle.reward = float(reward) if reward is not None else None
                    elif battle.terminal_result != result:
                        battle.terminal_conflict = True
    return battles, warnings


def collect_battle_features(battle: BattleRecord, side: str) -> dict[str, Any]:
    parsed_actions = [parse_choose_command(record.command) for record in battle.actions_taken]
    request_by_id = {request.request_id: request for request in battle.requests if request.request_id is not None}
    request_count = len(battle.requests)
    accepted_action_count = len(battle.actions_taken)
    rejected_action_count = len(battle.actions_rejected)
    move_slot_counts = Counter()
    switch_slot_counts = Counter()
    target_counts = Counter()
    targetable_double_commands = 0
    same_target_double_commands = 0
    split_target_double_commands = 0
    self_target_move_count = 0
    no_target_move_count = 0
    total_move_count = 0
    total_switch_count = 0
    pure_switch_turns = 0
    pass_switch_turns = 0
    move_switch_turns = 0
    double_move_turns = 0
    single_move_turns = 0
    spread_command_count = 0
    tera_action_index: int | None = None
    repeated_command_transitions = 0
    forced_switch_turns = 0
    force_switch_request_count = 0
    reduced_active_request_count = 0
    first_force_switch_action_index: int | None = None
    fallback_within_one_turn_of_force_switch = False
    fallback_within_one_turn_of_reduced_active = False
    fallback_within_one_turn_of_self_faint = False

    late_idx = late_start_index(accepted_action_count)
    late_single_action_fallback = 0
    late_slot6_switch_count = 0
    late_actions = 0
    earlier_double_action_seen = False
    prior_command = None
    previous_has_force_switch = False
    previous_reduced_active = False

    for index, (record, parsed) in enumerate(zip(battle.actions_taken, parsed_actions), start=1):
        parts = parsed.parts
        request = request_by_id.get(record.request_id)
        request_payload = request.payload if request is not None else {}
        has_force_switch = request_has_force_switch(request_payload)
        active_len = request_active_len(request_payload)
        if has_force_switch:
            force_switch_request_count += 1
            if first_force_switch_action_index is None:
                first_force_switch_action_index = index
        if active_len > 0 and active_len < 2:
            reduced_active_request_count += 1
        if prior_command is not None and parsed.raw.strip() == prior_command:
            repeated_command_transitions += 1
        prior_command = parsed.raw.strip()

        actionable_parts = [part for part in parts if part.kind in {"move", "switch", "pass"}]
        move_parts = [part for part in parts if part.kind == "move"]
        switch_parts = [part for part in parts if part.kind == "switch"]
        pass_parts = [part for part in parts if part.kind == "pass"]
        if len(actionable_parts) >= 2:
            earlier_double_action_seen = True

        if switch_parts or pass_parts or (move_parts and switch_parts):
            forced_switch_turns += 1
        if switch_parts and not move_parts and not pass_parts:
            pure_switch_turns += 1
        if switch_parts and pass_parts:
            pass_switch_turns += 1
        if switch_parts and move_parts:
            move_switch_turns += 1
        if len(move_parts) >= 2 and not switch_parts:
            double_move_turns += 1
        if len(move_parts) == 1 and not switch_parts:
            single_move_turns += 1
        if len(move_parts) >= 2 and len({part.target for part in move_parts if part.target is not None}) > 1:
            spread_command_count += 1

        for part in move_parts:
            total_move_count += 1
            if part.move_slot is not None:
                move_slot_counts[f"slot_{part.move_slot}"] += 1
            if part.target is None:
                no_target_move_count += 1
                target_counts["none"] += 1
            elif part.target < 0:
                self_target_move_count += 1
                target_counts["self_or_ally"] += 1
            elif part.target == 1:
                target_counts["foe_1"] += 1
            elif part.target == 2:
                target_counts["foe_2"] += 1
            else:
                target_counts[f"other_{part.target}"] += 1
            if part.used_tera and tera_action_index is None:
                tera_action_index = index

        move_targets = [part.target for part in move_parts if part.target is not None and part.target > 0]
        if len(move_targets) >= 2:
            targetable_double_commands += 1
            if len(set(move_targets)) == 1:
                same_target_double_commands += 1
            else:
                split_target_double_commands += 1

        for part in switch_parts:
            total_switch_count += 1
            if part.switch_slot is not None:
                switch_slot_counts[f"slot_{part.switch_slot}"] += 1

        if index > late_idx:
            late_actions += 1
            if earlier_double_action_seen and len(move_parts) == 1 and not switch_parts:
                late_single_action_fallback += 1
                if has_force_switch or previous_has_force_switch:
                    fallback_within_one_turn_of_force_switch = True
                if (active_len > 0 and active_len < 2) or previous_reduced_active:
                    fallback_within_one_turn_of_reduced_active = True
            for part in switch_parts:
                if part.switch_slot == 6:
                    late_slot6_switch_count += 1
        previous_has_force_switch = has_force_switch
        previous_reduced_active = active_len > 0 and active_len < 2

    tera_timing = "none"
    if tera_action_index is not None:
        if tera_action_index <= 3:
            tera_timing = "early"
        elif tera_action_index <= 7:
            tera_timing = "mid"
        else:
            tera_timing = "late"

    repeated_command_rate = safe_rate(repeated_command_transitions, max(accepted_action_count - 1, 1))
    late_single_action_rate = safe_rate(late_single_action_fallback, late_actions)
    slot6_switch_rate = safe_rate(switch_slot_counts.get("slot_6", 0), total_switch_count)
    late_slot6_switch_rate = safe_rate(late_slot6_switch_count, max(total_switch_count, 1))
    target_foe1_rate = safe_rate(target_counts.get("foe_1", 0), total_move_count)
    target_foe2_rate = safe_rate(target_counts.get("foe_2", 0), total_move_count)
    ally_target_rate = safe_rate(self_target_move_count, total_move_count)
    no_target_rate = safe_rate(no_target_move_count, total_move_count)

    flags: list[str] = []
    if slot6_switch_rate >= 0.5 and total_switch_count >= 2:
        flags.append("slot6_switch_heavy")
    if target_foe1_rate >= 0.6 and total_move_count >= 4:
        flags.append("foe1_target_heavy")
    if tera_timing == "early":
        flags.append("early_tera")
    if pass_switch_turns > 0:
        flags.append("pass_switch_present")
    if battle_bucket(accepted_action_count) == "long" and late_single_action_rate > 0.3:
        flags.append("long_collapse")
    if late_single_action_rate > 0.25:
        flags.append("late_single_action_fallback")
    if fallback_within_one_turn_of_force_switch:
        flags.append("fallback_after_force_switch")

    action_index = 0
    seen_double = False
    force_switch_window = 0
    reduced_active_window = 0
    self_faint_window = 0
    for entry in battle.timeline:
        entry_type = entry.get("type")
        if entry_type == "request":
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
            if request_has_force_switch(payload):
                force_switch_window = max(force_switch_window, 2)
            active_len = request_active_len(payload)
            if active_len > 0 and active_len < 2:
                reduced_active_window = max(reduced_active_window, 2)
        elif entry_type == "event":
            if self_faint_event(str(entry.get("line") or ""), side):
                self_faint_window = max(self_faint_window, 2)
        elif entry_type == "action_taken":
            action_index += 1
            parsed = parse_choose_command(str(entry.get("command") or ""))
            parts = parsed.parts
            move_parts = [part for part in parts if part.kind == "move"]
            switch_parts = [part for part in parts if part.kind == "switch"]
            if len([part for part in parts if part.kind in {"move", "switch", "pass"}]) >= 2:
                seen_double = True
            is_late_fallback = action_index > late_idx and seen_double and len(move_parts) == 1 and not switch_parts
            if is_late_fallback:
                if force_switch_window > 0:
                    fallback_within_one_turn_of_force_switch = True
                if reduced_active_window > 0:
                    fallback_within_one_turn_of_reduced_active = True
                if self_faint_window > 0:
                    fallback_within_one_turn_of_self_faint = True
            force_switch_window = max(0, force_switch_window - 1)
            reduced_active_window = max(0, reduced_active_window - 1)
            self_faint_window = max(0, self_faint_window - 1)
    if fallback_within_one_turn_of_self_faint:
        flags.append("fallback_after_self_faint")

    return {
        "battle_id": battle.battle_id,
        "source_file": sorted(battle.source_files)[0],
        "source_files": sorted(battle.source_files),
        "multi_file": len(battle.source_files) > 1,
        "result": battle.terminal_result,
        "reward": battle.reward,
        "request_count": request_count,
        "accepted_action_count": accepted_action_count,
        "rejected_action_count": rejected_action_count,
        "move_slot_counts": dict(move_slot_counts),
        "switch_slot_counts": dict(switch_slot_counts),
        "target_counts": dict(target_counts),
        "total_move_count": total_move_count,
        "total_switch_count": total_switch_count,
        "pure_switch_turns": pure_switch_turns,
        "pass_switch_turns": pass_switch_turns,
        "move_switch_turns": move_switch_turns,
        "double_move_turns": double_move_turns,
        "single_move_turns": single_move_turns,
        "spread_command_count": spread_command_count,
        "targetable_double_commands": targetable_double_commands,
        "same_target_double_commands": same_target_double_commands,
        "split_target_double_commands": split_target_double_commands,
        "self_target_move_count": self_target_move_count,
        "no_target_move_count": no_target_move_count,
        "tera_used": tera_action_index is not None,
        "tera_action_index": tera_action_index,
        "tera_timing": tera_timing,
        "repeated_command_rate": repeated_command_rate,
        "late_single_action_fallback": late_single_action_fallback,
        "late_single_action_rate": late_single_action_rate,
        "late_slot6_switch_count": late_slot6_switch_count,
        "late_slot6_switch_rate": late_slot6_switch_rate,
        "forced_switch_turns": forced_switch_turns,
        "force_switch_request_count": force_switch_request_count,
        "reduced_active_request_count": reduced_active_request_count,
        "first_force_switch_action_index": first_force_switch_action_index,
        "slot6_switch_rate": slot6_switch_rate,
        "target_foe1_rate": target_foe1_rate,
        "target_foe2_rate": target_foe2_rate,
        "ally_target_rate": ally_target_rate,
        "no_target_rate": no_target_rate,
        "battle_length_bucket": battle_bucket(accepted_action_count),
        "fallback_within_one_turn_of_force_switch": fallback_within_one_turn_of_force_switch,
        "fallback_within_one_turn_of_reduced_active": fallback_within_one_turn_of_reduced_active,
        "fallback_within_one_turn_of_self_faint": fallback_within_one_turn_of_self_faint,
        "reason_flags": flags,
    }


def aggregate_outcome(features: list[dict[str, Any]]) -> dict[str, Any]:
    battle_count = len(features)
    request_counts = [int(feature["request_count"]) for feature in features]
    accepted_counts = [int(feature["accepted_action_count"]) for feature in features]
    rejected_counts = [int(feature["rejected_action_count"]) for feature in features]
    move_slot_counts = Counter()
    switch_slot_counts = Counter()
    target_counts = Counter()
    tera_timing_counts = Counter()
    bucket_counts = Counter()

    total_moves = 0
    total_switches = 0
    total_pure_switch_turns = 0
    total_pass_switch_turns = 0
    total_move_switch_turns = 0
    total_double_move_turns = 0
    total_spread_command_count = 0
    total_targetable_double_commands = 0
    total_same_target_double_commands = 0
    total_split_target_double_commands = 0
    total_self_target_moves = 0
    total_no_target_moves = 0
    total_forced_switch_turns = 0
    total_force_switch_requests = 0
    total_reduced_active_requests = 0

    tera_used_battles = 0
    tera_action_indices: list[int] = []
    repeated_command_rates: list[float] = []
    late_single_action_rates: list[float] = []
    target_foe1_rates: list[float] = []
    fallback_after_force_switch_count = 0
    fallback_after_reduced_active_count = 0
    fallback_after_self_faint_count = 0
    flagged_counts = Counter()

    for feature in features:
        move_slot_counts.update(feature["move_slot_counts"])
        switch_slot_counts.update(feature["switch_slot_counts"])
        target_counts.update(feature["target_counts"])
        tera_timing_counts[feature["tera_timing"]] += 1
        bucket_counts[feature["battle_length_bucket"]] += 1
        total_moves += int(feature["total_move_count"])
        total_switches += int(feature["total_switch_count"])
        total_pure_switch_turns += int(feature["pure_switch_turns"])
        total_pass_switch_turns += int(feature["pass_switch_turns"])
        total_move_switch_turns += int(feature["move_switch_turns"])
        total_double_move_turns += int(feature["double_move_turns"])
        total_spread_command_count += int(feature["spread_command_count"])
        total_targetable_double_commands += int(feature["targetable_double_commands"])
        total_same_target_double_commands += int(feature["same_target_double_commands"])
        total_split_target_double_commands += int(feature["split_target_double_commands"])
        total_self_target_moves += int(feature["self_target_move_count"])
        total_no_target_moves += int(feature["no_target_move_count"])
        total_forced_switch_turns += int(feature["forced_switch_turns"])
        total_force_switch_requests += int(feature["force_switch_request_count"])
        total_reduced_active_requests += int(feature["reduced_active_request_count"])
        if feature["tera_used"]:
            tera_used_battles += 1
        if feature["tera_action_index"] is not None:
            tera_action_indices.append(int(feature["tera_action_index"]))
        repeated_command_rates.append(float(feature["repeated_command_rate"]))
        late_single_action_rates.append(float(feature["late_single_action_rate"]))
        target_foe1_rates.append(float(feature["target_foe1_rate"]))
        if feature["fallback_within_one_turn_of_force_switch"]:
            fallback_after_force_switch_count += 1
        if feature["fallback_within_one_turn_of_reduced_active"]:
            fallback_after_reduced_active_count += 1
        if feature["fallback_within_one_turn_of_self_faint"]:
            fallback_after_self_faint_count += 1
        flagged_counts.update(feature["reason_flags"])

    move_slot_rates = {slot: safe_rate(count, total_moves) for slot, count in sorted(move_slot_counts.items())}
    switch_slot_rates = {slot: safe_rate(count, total_switches) for slot, count in sorted(switch_slot_counts.items())}
    target_patterns = {
        "foe1_bias": safe_rate(target_counts.get("foe_1", 0), total_moves),
        "foe2_bias": safe_rate(target_counts.get("foe_2", 0), total_moves),
        "self_target_rate": safe_rate(total_self_target_moves, total_moves),
        "no_explicit_target_rate": safe_rate(total_no_target_moves, total_moves),
        "same_target_double_rate": safe_rate(total_same_target_double_commands, total_targetable_double_commands),
        "split_target_rate": safe_rate(total_split_target_double_commands, total_targetable_double_commands),
        "spread_command_rate": safe_rate(total_spread_command_count, max(total_double_move_turns, 1)),
    }
    forced_switch = {
        "forced_switch_turns": total_forced_switch_turns,
        "force_switch_request_count": total_force_switch_requests,
        "reduced_active_request_count": total_reduced_active_requests,
        "pure_switch_rate": safe_rate(total_pure_switch_turns, total_forced_switch_turns),
        "pass_switch_rate": safe_rate(total_pass_switch_turns, total_forced_switch_turns),
        "move_switch_rate": safe_rate(total_move_switch_turns, total_forced_switch_turns),
        "slot6_rate": safe_rate(switch_slot_counts.get("slot_6", 0), total_switches),
        "switch_concentration": max(switch_slot_rates.values(), default=0.0),
        "fallback_within_one_turn_of_force_switch_rate": safe_rate(fallback_after_force_switch_count, battle_count),
        "fallback_within_one_turn_of_reduced_active_rate": safe_rate(fallback_after_reduced_active_count, battle_count),
        "fallback_within_one_turn_of_self_faint_rate": safe_rate(fallback_after_self_faint_count, battle_count),
    }
    tera = {
        "tera_rate": safe_rate(tera_used_battles, battle_count),
        "first_tera_action_index_avg": mean_or_zero(tera_action_indices),
        "timing_distribution": {
            key: safe_rate(tera_timing_counts.get(key, 0), battle_count)
            for key in ["early", "mid", "late", "none"]
        },
        "early_tera_rate": safe_rate(tera_timing_counts.get("early", 0), battle_count),
    }
    battle_length = {
        "avg_accepted_actions_per_battle": mean_or_zero(accepted_counts),
        "median_accepted_actions_per_battle": median_or_zero(accepted_counts),
        "avg_requests_per_battle": mean_or_zero(request_counts),
        "median_requests_per_battle": median_or_zero(request_counts),
        "bucket_rates": {key: safe_rate(bucket_counts.get(key, 0), battle_count) for key in ["short", "medium", "long"]},
    }
    loss_heuristics = {
        "late_single_action_fallback_rate": sum(late_single_action_rates) / battle_count if battle_count else 0.0,
        "repeated_foe1_bias": sum(target_foe1_rates) / battle_count if battle_count else 0.0,
        "early_tera_rate": tera["early_tera_rate"],
        "pass_switch_rate": forced_switch["pass_switch_rate"],
        "fallback_within_one_turn_of_force_switch_rate": forced_switch["fallback_within_one_turn_of_force_switch_rate"],
        "fallback_within_one_turn_of_reduced_active_rate": forced_switch["fallback_within_one_turn_of_reduced_active_rate"],
        "fallback_within_one_turn_of_self_faint_rate": forced_switch["fallback_within_one_turn_of_self_faint_rate"],
    }
    return {
        "battle_count": battle_count,
        "request_count": sum(request_counts),
        "accepted_action_count": sum(accepted_counts),
        "rejected_action_count": sum(rejected_counts),
        "avg_accepted_actions_per_battle": battle_length["avg_accepted_actions_per_battle"],
        "median_accepted_actions_per_battle": battle_length["median_accepted_actions_per_battle"],
        "avg_requests_per_battle": battle_length["avg_requests_per_battle"],
        "median_requests_per_battle": battle_length["median_requests_per_battle"],
        "move_slot_counts": dict(move_slot_counts),
        "move_slot_rates": move_slot_rates,
        "switch_slot_counts": dict(switch_slot_counts),
        "switch_slot_rates": switch_slot_rates,
        "target_counts": dict(target_counts),
        "target_patterns": target_patterns,
        "forced_switch": forced_switch,
        "tera": tera,
        "battle_length": battle_length,
        "loss_heuristics": loss_heuristics,
        "repeated_command_rate_avg": sum(repeated_command_rates) / battle_count if battle_count else 0.0,
        "reason_flag_counts": dict(flagged_counts),
    }


def build_comparisons(by_outcome: dict[str, dict[str, Any]]) -> dict[str, Any]:
    wins = by_outcome.get("win")
    losses = by_outcome.get("loss")
    if not wins or not losses:
        return {}
    return {
        "loss_minus_win": {
            "slot6_switch_rate": losses["forced_switch"]["slot6_rate"] - wins["forced_switch"]["slot6_rate"],
            "foe1_target_rate": losses["target_patterns"]["foe1_bias"] - wins["target_patterns"]["foe1_bias"],
            "early_tera_rate": losses["tera"]["early_tera_rate"] - wins["tera"]["early_tera_rate"],
            "same_target_double_rate": losses["target_patterns"]["same_target_double_rate"] - wins["target_patterns"]["same_target_double_rate"],
            "late_single_action_fallback_rate": losses["loss_heuristics"]["late_single_action_fallback_rate"] - wins["loss_heuristics"]["late_single_action_fallback_rate"],
            "fallback_within_one_turn_of_force_switch_rate": losses["loss_heuristics"]["fallback_within_one_turn_of_force_switch_rate"] - wins["loss_heuristics"]["fallback_within_one_turn_of_force_switch_rate"],
            "fallback_within_one_turn_of_reduced_active_rate": losses["loss_heuristics"]["fallback_within_one_turn_of_reduced_active_rate"] - wins["loss_heuristics"]["fallback_within_one_turn_of_reduced_active_rate"],
            "fallback_within_one_turn_of_self_faint_rate": losses["loss_heuristics"]["fallback_within_one_turn_of_self_faint_rate"] - wins["loss_heuristics"]["fallback_within_one_turn_of_self_faint_rate"],
        }
    }


def select_representative_losses(losses: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    ranked = sorted(
        losses,
        key=lambda feature: (
            len(feature["reason_flags"]),
            feature["late_single_action_rate"],
            feature["slot6_switch_rate"],
            feature["target_foe1_rate"],
            feature["accepted_action_count"],
        ),
        reverse=True,
    )
    return [
        {
            "battle_id": feature["battle_id"],
            "source_file": feature["source_file"],
            "reason_flags": feature["reason_flags"],
        }
        for feature in ranked[:sample_size]
    ]


def print_text_summary(summary: dict[str, Any], by_outcome: dict[str, dict[str, Any]], representative_losses: list[dict[str, Any]], side: str) -> None:
    print(
        f"[replay_diagnose] battles={summary['battle_count']} side={side} "
        f"wins={summary['wins']} losses={summary['losses']} draws={summary['draws']} "
        f"win_rate={summary['win_rate']:.3f}",
        flush=True,
    )
    if "win" in by_outcome or "loss" in by_outcome:
        win_tera = by_outcome.get("win", {}).get("tera", {}).get("tera_rate", 0.0)
        loss_tera = by_outcome.get("loss", {}).get("tera", {}).get("tera_rate", 0.0)
        print(f"[replay_diagnose] tera_rate win={win_tera:.2f} loss={loss_tera:.2f}", flush=True)
    loss_metrics = by_outcome.get("loss")
    if loss_metrics:
        loss_move = loss_metrics.get("move_slot_rates", {})
        print(
            "[replay_diagnose] move_slot_rates loss "
            f"slot1={loss_move.get('slot_1', 0.0):.2f} "
            f"slot2={loss_move.get('slot_2', 0.0):.2f} "
            f"slot3={loss_move.get('slot_3', 0.0):.2f} "
            f"slot4={loss_move.get('slot_4', 0.0):.2f}",
            flush=True,
        )
        loss_switch = loss_metrics.get("switch_slot_rates", {})
        print(
            "[replay_diagnose] switch_slot_rates loss "
            f"slot3={loss_switch.get('slot_3', 0.0):.2f} "
            f"slot4={loss_switch.get('slot_4', 0.0):.2f} "
            f"slot5={loss_switch.get('slot_5', 0.0):.2f} "
            f"slot6={loss_switch.get('slot_6', 0.0):.2f}",
            flush=True,
        )
        target_patterns = loss_metrics.get("target_patterns", {})
        print(
            "[replay_diagnose] target_patterns loss "
            f"foe1_bias={target_patterns.get('foe1_bias', 0.0):.2f} "
            f"self_target_rate={target_patterns.get('self_target_rate', 0.0):.2f} "
            f"spread_command_rate={target_patterns.get('spread_command_rate', 0.0):.2f}",
            flush=True,
        )
        forced_switch = loss_metrics.get("forced_switch", {})
        print(
            "[replay_diagnose] forced_switch loss "
            f"slot6_rate={forced_switch.get('slot6_rate', 0.0):.2f} "
            f"pass_switch_rate={forced_switch.get('pass_switch_rate', 0.0):.2f} "
            f"fallback_after_self_faint={forced_switch.get('fallback_within_one_turn_of_self_faint_rate', 0.0):.2f}",
            flush=True,
        )
    if representative_losses:
        print(
            "[replay_diagnose] representative_losses " + " ".join(item["battle_id"] for item in representative_losses),
            flush=True,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    outcome_filter = parse_outcome_filter(args.include_outcomes)
    mode, files, output_path = resolve_input_files(args, repo_root)
    battles, warnings = load_battles(files)

    usable_features: list[dict[str, Any]] = []
    skipped_battles = {
        "missing_terminal_count": 0,
        "conflicting_terminal_count": 0,
    }
    for battle in battles.values():
        if battle.terminal_conflict:
            skipped_battles["conflicting_terminal_count"] += 1
            continue
        if battle.terminal_result is None:
            skipped_battles["missing_terminal_count"] += 1
            continue
        if battle.terminal_result not in outcome_filter:
            continue
        usable_features.append(collect_battle_features(battle, args.side))

    if len(usable_features) < args.min_battles:
        raise SystemExit(f"usable battles {len(usable_features)} is below --min-battles {args.min_battles}")

    by_result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in usable_features:
        by_result[str(feature["result"])].append(feature)

    summary = {
        "battle_count": len(usable_features),
        "wins": len(by_result.get("win", [])),
        "losses": len(by_result.get("loss", [])),
        "draws": len(by_result.get("draw", [])),
    }
    summary["win_rate"] = safe_rate(summary["wins"], summary["wins"] + summary["losses"])

    by_outcome = {outcome: aggregate_outcome(features) for outcome, features in sorted(by_result.items())}
    comparisons = build_comparisons(by_outcome)
    representative_losses = select_representative_losses(by_result.get("loss", []), args.sample_losses)
    report = {
        "input": {
            "mode": mode,
            "run": args.run,
            "path": args.path,
            "glob": args.glob_pattern,
            "side": args.side,
            "files": [str(path) for path in files],
        },
        "summary": summary,
        "by_outcome": by_outcome,
        "comparisons": comparisons,
        "representative_losses": representative_losses,
        "skipped_battles": skipped_battles,
        "parse_warnings": warnings,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.write_text(
                json.dumps(report, indent=2 if args.pretty else None, sort_keys=args.pretty),
                encoding="utf-8",
            )
        except OSError as exc:
            raise SystemExit(f"failed to write output {output_path}: {exc}") from exc
        if not args.quiet:
            print(f"[replay_diagnose] wrote report {output_path}", flush=True)

    if not args.quiet:
        print_text_summary(summary, by_outcome, representative_losses, args.side)


if __name__ == "__main__":
    main()
