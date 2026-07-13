from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("matches") / "runs"


@dataclass
class BattleBuffer:
    battle_id: str
    lines: list[str] = field(default_factory=list)
    source_files: set[str] = field(default_factory=set)
    has_force_switch: bool = False
    has_reduced_active: bool = False
    has_post_trigger_activity: bool = False
    action_taken_after_trigger: bool = False
    request_after_trigger: bool = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Source run under matches/runs/")
    parser.add_argument("--output-run", required=True, help="Destination filtered run name under matches/runs/")
    parser.add_argument("--side", choices=["a", "b"], default="a")
    parser.add_argument("--min-lines", type=positive_int, default=1, help="Minimum number of lines to write per output shard")
    return parser


def load_battle_buffers(files: list[Path]) -> dict[str, BattleBuffer]:
    battles: dict[str, BattleBuffer] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                battle_id = payload.get("battle_id")
                if not battle_id:
                    continue
                battle_id = str(battle_id)
                battle = battles.get(battle_id)
                if battle is None:
                    battle = BattleBuffer(battle_id=battle_id)
                    battles[battle_id] = battle
                battle.lines.append(line if line.endswith("\n") else line + "\n")
                battle.source_files.add(path.name)

                record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
                if record_type == "request":
                    request_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                    force_switch = request_has_force_switch(request_payload)
                    reduced_active = request_active_len(request_payload) > 0 and request_active_len(request_payload) < 2
                    if force_switch:
                        battle.has_force_switch = True
                    if reduced_active:
                        battle.has_reduced_active = True
                    if (battle.has_force_switch or battle.has_reduced_active) and (force_switch or reduced_active):
                        continue
                    if battle.has_force_switch or battle.has_reduced_active:
                        battle.has_post_trigger_activity = True
                        battle.request_after_trigger = True
                elif record_type == "action_taken":
                    if battle.has_force_switch or battle.has_reduced_active:
                        battle.has_post_trigger_activity = True
                        battle.action_taken_after_trigger = True
                elif record_type == "terminal":
                    if battle.has_force_switch or battle.has_reduced_active:
                        battle.has_post_trigger_activity = True
    return battles


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / DEFAULT_RUNS_ROOT / args.run
    output_dir = repo_root / DEFAULT_RUNS_ROOT / args.output_run
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    all_files = sorted(run_dir.glob(f"worker_*_{args.side}_raw.jsonl"))
    if not all_files:
        raise SystemExit(f"no replay files found for side {args.side} in {run_dir}")

    battles = load_battle_buffers(all_files)
    selected_battles = {
        battle_id: battle
        for battle_id, battle in battles.items()
        if (battle.has_force_switch or battle.has_reduced_active) and battle.has_post_trigger_activity
    }
    if not selected_battles:
        raise SystemExit("no battles matched the post-faint/reduced-board filter")

    output_dir.mkdir(parents=True, exist_ok=True)
    lines_by_file: dict[str, list[str]] = defaultdict(list)
    selected_ids = set(selected_battles.keys())
    for path in all_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                battle_id = payload.get("battle_id")
                if battle_id and str(battle_id) in selected_ids:
                    lines_by_file[path.name].append(line if line.endswith("\n") else line + "\n")

    written_files = 0
    written_lines = 0
    for filename, lines in sorted(lines_by_file.items()):
        if len(lines) < args.min_lines:
            continue
        target_path = output_dir / filename
        target_path.write_text("".join(lines), encoding="utf-8")
        written_files += 1
        written_lines += len(lines)

    summary = {
        "source_run": args.run,
        "output_run": args.output_run,
        "side": args.side,
        "selected_battles": len(selected_battles),
        "written_files": written_files,
        "written_lines": written_lines,
        "battle_ids": sorted(selected_ids),
    }
    (output_dir / f"{args.output_run}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[replay_extract_postfaint] source_run={args.run} output_run={args.output_run} "
        f"side={args.side} selected_battles={len(selected_battles)} written_files={written_files} written_lines={written_lines}",
        flush=True,
    )


if __name__ == "__main__":
    main()
