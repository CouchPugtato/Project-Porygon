from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_defaults import float_default, int_default

DEFAULT_WARN_MOVE_SLOT_CONCENTRATION = float_default("warn_move_slot_concentration")
DEFAULT_HARD_MOVE_SLOT_COLLAPSE = float_default("hard_move_slot_collapse")
DEFAULT_WARN_SWITCH_SLOT6_CONCENTRATION = float_default("warn_switch_slot_6_concentration")
DEFAULT_WARN_TERA_BASELINE_RATIO = float_default("warn_tera_baseline_ratio")
DEFAULT_FAIL_FAST_EARNED_WIN_RATE = float_default("fail_fast_earned_win_rate")
DEFAULT_FAIL_FAST_MIN_GAMES = int_default("fail_fast_min_games")


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def load_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read summary {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid summary {path}: expected top-level object")
    return payload


def collapse_flags_for_group(
    group: dict[str, object],
    baseline_group: dict[str, object] | None,
    *,
    warn_move_slot_concentration: float,
    hard_move_slot_collapse: float,
    warn_switch_slot6_concentration: float,
    warn_tera_baseline_ratio: float,
    fail_fast_earned_win_rate: float,
    fail_fast_min_games: int,
) -> list[str]:
    flags: list[str] = []
    move_rates = group.get("move_slot_rates", {})
    if isinstance(move_rates, dict) and move_rates:
        dominant_slot, dominant_rate = max(
            ((str(slot), float(rate)) for slot, rate in move_rates.items()),
            key=lambda item: item[1],
        )
        if dominant_rate >= hard_move_slot_collapse:
            flags.append(f"hard_move_slot_collapse:{dominant_slot}:{dominant_rate:.3f}")
        elif dominant_rate >= warn_move_slot_concentration:
            flags.append(f"warn_move_slot_concentration:{dominant_slot}:{dominant_rate:.3f}")

    switch_rates = group.get("switch_slot_rates", {})
    if isinstance(switch_rates, dict):
        slot6_rate = float(switch_rates.get("slot_6", 0.0))
        if slot6_rate >= warn_switch_slot6_concentration:
            flags.append(f"warn_switch_slot_6_concentration:{slot6_rate:.3f}")

    tera_rate = float(group.get("tera_battle_rate", group.get("tera_rate", 0.0)))
    baseline_tera_rate = float(baseline_group.get("tera_battle_rate", baseline_group.get("tera_rate", 0.0))) if baseline_group else 0.0
    if baseline_tera_rate > 0.0 and tera_rate < (baseline_tera_rate * warn_tera_baseline_ratio):
        flags.append(f"warn_tera_rate_low_vs_baseline:{tera_rate:.3f}:{baseline_tera_rate:.3f}")

    matches_played = int(group.get("matches_played", 0))
    earned_win_rate = float(group.get("earned_win_rate", 0.0))
    if matches_played >= fail_fast_min_games and earned_win_rate < fail_fast_earned_win_rate:
        flags.append(f"fail_fast_low_earned_win_rate:{earned_win_rate:.3f}")
    return flags


def infer_group_stats(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_group_stats = summary.get("group_stats")
    if isinstance(raw_group_stats, dict):
        normalized: dict[str, dict[str, object]] = {}
        for side in ("a", "b"):
            side_stats = raw_group_stats.get(side, {})
            if isinstance(side_stats, dict):
                normalized[side] = side_stats
        if normalized:
            return normalized
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--candidate-side", choices=["a", "b"], default="a")
    parser.add_argument("--output", default="")
    parser.add_argument("--warn-move-slot-concentration", type=float, default=DEFAULT_WARN_MOVE_SLOT_CONCENTRATION)
    parser.add_argument("--hard-move-slot-collapse", type=float, default=DEFAULT_HARD_MOVE_SLOT_COLLAPSE)
    parser.add_argument("--warn-switch-slot6-concentration", type=float, default=DEFAULT_WARN_SWITCH_SLOT6_CONCENTRATION)
    parser.add_argument("--warn-tera-baseline-ratio", type=float, default=DEFAULT_WARN_TERA_BASELINE_RATIO)
    parser.add_argument("--fail-fast-earned-win-rate", type=float, default=DEFAULT_FAIL_FAST_EARNED_WIN_RATE)
    parser.add_argument("--fail-fast-min-games", type=int, default=DEFAULT_FAIL_FAST_MIN_GAMES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.summary).resolve()
    summary = load_summary(summary_path)
    group_stats = infer_group_stats(summary)
    if args.candidate_side not in group_stats:
        raise SystemExit(f"summary missing group_stats for side {args.candidate_side}")
    baseline_side = "b" if args.candidate_side == "a" else "a"
    candidate_group = group_stats[args.candidate_side]
    baseline_group = group_stats.get(baseline_side)
    flags = collapse_flags_for_group(
        candidate_group,
        baseline_group,
        warn_move_slot_concentration=args.warn_move_slot_concentration,
        hard_move_slot_collapse=args.hard_move_slot_collapse,
        warn_switch_slot6_concentration=args.warn_switch_slot6_concentration,
        warn_tera_baseline_ratio=args.warn_tera_baseline_ratio,
        fail_fast_earned_win_rate=args.fail_fast_earned_win_rate,
        fail_fast_min_games=args.fail_fast_min_games,
    )
    payload = {
        "summary_path": str(summary_path),
        "candidate_side": args.candidate_side,
        "candidate_checkpoint": summary.get("candidate_checkpoint", ""),
        "parent_checkpoint": summary.get("parent_checkpoint", ""),
        "matches_played": int(candidate_group.get("matches_played", 0)),
        "earned_win_rate": float(candidate_group.get("earned_win_rate", 0.0)),
        "tera_battle_rate": float(candidate_group.get("tera_battle_rate", candidate_group.get("tera_rate", 0.0))),
        "tera_rate": float(candidate_group.get("tera_battle_rate", candidate_group.get("tera_rate", 0.0))),
        "dominant_move_slot_rate": max(
            (float(rate) for rate in candidate_group.get("move_slot_rates", {}).values()),
            default=0.0,
        ),
        "switch_slot_6_rate": float(candidate_group.get("switch_slot_rates", {}).get("slot_6", 0.0)),
        "collapse_flags": flags,
        "collapse_detected": bool(flags),
    }
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
