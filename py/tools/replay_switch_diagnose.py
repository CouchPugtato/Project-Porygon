from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.tools.replay_diagnose import (
    DEFAULT_RUNS_ROOT,
    collect_battle_features,
    load_battles,
    parse_bool01,
    positive_int,
    resolve_input_files,
)


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run name under matches/runs/")
    parser.add_argument("--path", help="Replay jsonl file or directory")
    parser.add_argument("--glob", dest="glob_pattern", help="Glob for replay jsonl files")
    parser.add_argument("--side", choices=["a", "b"], default="a")
    parser.add_argument("--output", default="")
    parser.add_argument("--pretty", type=parse_bool01, default=True)
    parser.add_argument("--min-battles", type=positive_int, default=1)
    parser.add_argument("--include-files", type=parse_bool01, default=False)
    return parser


def bucket_name(feature: dict[str, Any]) -> str:
    if int(feature.get("force_switch_request_count", 0)) > 0:
        return "forced_switch"
    if int(feature.get("reduced_active_request_count", 0)) > 0:
        return "reduced_active"
    if int(feature.get("total_switch_count", 0)) > 0:
        return "voluntary_switch"
    return "no_switch"


def summarize_bucket(features: list[dict[str, Any]]) -> dict[str, Any]:
    total_switch_count = sum(int(feature.get("total_switch_count", 0)) for feature in features)
    total_slot6_switches = sum(int(feature.get("switch_slot_counts", {}).get("slot_6", 0)) for feature in features)
    total_force_switch_requests = sum(int(feature.get("force_switch_request_count", 0)) for feature in features)
    total_reduced_active_requests = sum(int(feature.get("reduced_active_request_count", 0)) for feature in features)
    switch_slot_counts = Counter()
    for feature in features:
        switch_slot_counts.update(feature.get("switch_slot_counts", {}))
    switch_slot_rates = {
        slot: safe_rate(count, total_switch_count) for slot, count in sorted(switch_slot_counts.items())
    }
    return {
        "battle_count": len(features),
        "total_switch_count": total_switch_count,
        "force_switch_request_count": total_force_switch_requests,
        "reduced_active_request_count": total_reduced_active_requests,
        "slot6_switch_rate": safe_rate(total_slot6_switches, total_switch_count),
        "switch_slot_counts": dict(switch_slot_counts),
        "switch_slot_rates": switch_slot_rates,
    }


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    _, files, _ = resolve_input_files(args, repo_root)
    battles, _ = load_battles(files)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for battle in battles.values():
        feature = collect_battle_features(battle, args.side)
        result = str(feature.get("result") or "unknown")
        grouped[result][bucket_name(feature)].append(feature)

    summary: dict[str, Any] = {
        "side": args.side,
        "file_count": len(files),
        "buckets": {},
    }
    if args.include_files:
        summary["files"] = [str(path) for path in files]
    for result, buckets in sorted(grouped.items()):
        result_summary: dict[str, Any] = {}
        for name, features in sorted(buckets.items()):
            if len(features) < args.min_battles:
                continue
            result_summary[name] = summarize_bucket(features)
        if result_summary:
            summary["buckets"][result] = result_summary

    render_indent = 2 if args.pretty else None
    output_path: Path | None = None
    if args.output:
        explicit_output = Path(args.output)
        if explicit_output.is_absolute():
            output_path = explicit_output
        elif args.run:
            output_path = repo_root / DEFAULT_RUNS_ROOT / args.run / explicit_output
        elif output_path is None:
            output_path = repo_root / explicit_output
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=render_indent, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=render_indent, sort_keys=True))


if __name__ == "__main__":
    main()
