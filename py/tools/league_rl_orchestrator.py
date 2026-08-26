from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from league_manage import (
    DEFAULT_REGISTRY_PATH,
    LeagueMember,
    LeagueRegistry,
    OpponentStats,
    find_member,
    league_member_from_json,
    load_registry,
    member_to_json_dict,
    save_registry,
)
from opponent_sampling import ADAPTIVE_STRATEGY, matchup_difficulty_weight as calculate_matchup_difficulty_weight
from live_rl_orchestrator import (
    BaseWorkflowReporter,
    DashboardProgressWriter,
    WorkflowDashboardState,
    run_reported_command,
    select_workflow_reporter,
)
from ppo_search import (
    aggregate_collapse_flags,
    aggregate_group_stats,
    evaluation_artifacts_match,
    valid_outcome_counts,
    wilson_interval,
)
from rl_defaults import bool_default, float_default, int_default


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_LEAGUE_ROOT = Path("models") / "league"
DEFAULT_MATCHUP_TARGET_WIN_RATE = float_default("league_matchup_target_win_rate")
DEFAULT_MATCHUP_MIN_WEIGHT = float_default("league_matchup_min_weight")
DEFAULT_MATCHUP_CONFIDENCE_GAMES = int_default("league_matchup_confidence_games")
DEFAULT_MIN_CATEGORY_STARTS = int_default("league_min_category_starts")
DEFAULT_PROMOTION_CONFIDENCE_THRESHOLD = float_default("promotion_confidence_threshold")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and at most 1")
    return parsed


def target_win_rate_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
    return parsed


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


def load_default_args(path: Path) -> list[str]:
    if not path.exists():
        return []
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
        flag = "--" + key.replace("_", "-")
        lowered = value_text.lower()
        if lowered in {"true", "false"}:
            args.extend([flag, "1" if lowered == "true" else "0"])
            continue
        if len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        args.extend([flag, value])
    return args


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(f"[league-rl] {message}", flush=True)


def registry_has_member(registry: LeagueRegistry, member_id: str) -> bool:
    return any(member.id == member_id for member in registry.members)


def utc_now_unix() -> float:
    return time.time()


def matchup_reference(
    learner: LeagueMember,
    opponent: LeagueMember,
    champion_id: str,
    confidence_games: int,
) -> tuple[float | None, int]:
    stats = learner.opponent_stats.get(opponent.id)
    if stats is not None and stats.matches_played > 0:
        return min(1.0, max(0.0, stats.recent_win_rate)), stats.matches_played
    if opponent.id == champion_id and learner.eval.vs_champion_win_rate is not None:
        return min(1.0, max(0.0, learner.eval.vs_champion_win_rate)), confidence_games
    return None, 0


def matchup_difficulty_weight(
    win_rate: float | None,
    matches_played: int,
    *,
    target_win_rate: float = DEFAULT_MATCHUP_TARGET_WIN_RATE,
    min_weight: float = DEFAULT_MATCHUP_MIN_WEIGHT,
    confidence_games: int = DEFAULT_MATCHUP_CONFIDENCE_GAMES,
) -> float:
    return calculate_matchup_difficulty_weight(
        win_rate,
        matches_played,
        target_win_rate=target_win_rate,
        min_weight=min_weight,
        confidence_games=confidence_games,
    )


def sort_recent(members: list[LeagueMember]) -> list[LeagueMember]:
    return sorted(members, key=lambda item: (item.generation, item.promoted_at or item.created_at, item.id), reverse=True)


def choose_parent_member(registry: LeagueRegistry, parent_id: str, learner_role: str) -> LeagueMember:
    if parent_id:
        return find_member(registry, parent_id)
    if learner_role == "main":
        if registry.champion_id:
            return find_member(registry, registry.champion_id)
    candidates = [member for member in registry.members if member.role == learner_role and member.status in {"active", "candidate"}]
    if candidates:
        return sort_recent(candidates)[0]
    if registry.champion_id:
        return find_member(registry, registry.champion_id)
    raise SystemExit("could not infer parent member; provide --parent-id or initialize the registry with a champion")


def build_weighted_pool(
    registry: LeagueRegistry,
    learner_role: str,
    parent_id: str,
    include_random_weight: float,
    *,
    target_win_rate: float = DEFAULT_MATCHUP_TARGET_WIN_RATE,
    min_difficulty_weight: float = DEFAULT_MATCHUP_MIN_WEIGHT,
    confidence_games: int = DEFAULT_MATCHUP_CONFIDENCE_GAMES,
    min_category_starts: int = DEFAULT_MIN_CATEGORY_STARTS,
) -> dict[str, object]:
    members: list[dict[str, object]] = []
    parent_member = find_member(registry, parent_id)
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    historical = [m for m in registry.members if m.role == "historical_snapshot" and m.status == "active"]
    recent_main = [
        m for m in registry.members
        if m.role == "main"
        and m.status in {"active", "candidate"}
        and m.id not in {parent_id, registry.champion_id}
    ]
    main_exploiters = [m for m in registry.members if m.role == "main_exploiter" and m.status in {"active", "candidate"}]
    league_exploiters = [m for m in registry.members if m.role == "league_exploiter" and m.status in {"active", "candidate"}]

    bucket_members: list[tuple[float, list[LeagueMember], str]] = []
    if learner_role == "main":
        bucket_members.extend([
            (0.40, [champion] if champion else [], "champion"),
            (0.25, sort_recent(recent_main)[:4], "recent_main"),
            (0.20, historical, "historical"),
            (0.10, sort_recent(main_exploiters)[:3], "main_exploiter"),
            (0.05, sort_recent(league_exploiters)[:3], "league_exploiter"),
        ])
    elif learner_role == "main_exploiter":
        targets = [parent_member]
        bucket_members.extend([
            (0.60, targets, "exploit_target"),
            (0.20, sort_recent(recent_main)[:4], "recent_main"),
            (0.20, [champion] if champion else [], "champion"),
        ])
    elif learner_role == "league_exploiter":
        bucket_members.extend([
            (0.50, [champion] if champion else [], "champion"),
            (0.30, historical, "historical"),
            (0.20, sort_recent(recent_main)[:4], "recent_main"),
        ])
    else:
        bucket_members.append((1.0, [champion] if champion else [parent_member], "champion"))

    combined: dict[str, dict[str, object]] = {}
    for bucket_weight, bucket, category in bucket_members:
        filtered = [member for member in bucket if member is not None]
        if not filtered or bucket_weight <= 0.0:
            continue
        raw_weights: list[float] = []
        matchup_metadata: list[tuple[float | None, int, float]] = []
        for member in filtered:
            reference_rate, matchup_games = matchup_reference(
                parent_member,
                member,
                registry.champion_id,
                confidence_games,
            )
            difficulty_weight = matchup_difficulty_weight(
                reference_rate,
                matchup_games,
                target_win_rate=target_win_rate,
                min_weight=min_difficulty_weight,
                confidence_games=confidence_games,
            )
            raw_weights.append(member.collection_weight * difficulty_weight)
            matchup_metadata.append((reference_rate, matchup_games, difficulty_weight))
        total = sum(raw_weights)
        if total <= 0.0:
            continue
        for member, raw_weight, metadata in zip(filtered, raw_weights, matchup_metadata):
            reference_rate, matchup_games, difficulty_weight = metadata
            weight = bucket_weight * (raw_weight / total)
            existing = combined.get(member.id)
            if existing is None:
                combined[member.id] = {
                    "name": member.id,
                    "kind": "checkpoint",
                    "path": member.path,
                    "weight": weight,
                    "category": category,
                    "bucket_weight": bucket_weight,
                    "learner_win_rate": reference_rate,
                    "matchup_games": matchup_games,
                    "difficulty_weight": difficulty_weight,
                    "base_weight": member.collection_weight,
                }
            else:
                existing["weight"] = float(existing["weight"]) + weight

    members = sorted(combined.values(), key=lambda item: (-float(item["weight"]), str(item["name"])))
    if include_random_weight > 0.0:
        members.append({
            "name": "random",
            "kind": "random",
            "weight": include_random_weight,
            "category": "random",
        })
    if not members:
        members.append({
            "name": champion.id if champion else parent_member.id,
            "kind": "checkpoint",
            "path": (champion or parent_member).path,
            "weight": 1.0,
            "category": "fallback",
            "bucket_weight": 1.0,
            "difficulty_weight": 1.0,
            "base_weight": 1.0,
            "learner_win_rate": None,
            "matchup_games": 0,
        })
    return {
        "coverage": {
            "enabled": min_category_starts > 0,
            "min_category_starts": min_category_starts,
            "prefer_under_sampled_members": True,
        },
        "sampling": {
            "strategy": ADAPTIVE_STRATEGY,
            "target_win_rate": target_win_rate,
            "min_difficulty_weight": min_difficulty_weight,
            "confidence_games": confidence_games,
        },
        "members": members,
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def opponent_stats_payload(stats_by_opponent: dict[str, OpponentStats]) -> dict[str, dict[str, object]]:
    return {
        opponent_id: {
            "matches_played": stats.matches_played,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "recent_win_rate": stats.recent_win_rate,
            "source_run": stats.source_run,
        }
        for opponent_id, stats in sorted(stats_by_opponent.items())
    }


def collect_recent_opponent_stats(
    repo_root: Path,
    round_manifest_paths: list[Path],
    source_run: str,
) -> dict[str, OpponentStats]:
    aggregate: dict[str, OpponentStats] = {}
    for manifest_path in round_manifest_paths:
        resolved_manifest = resolve_path(repo_root, manifest_path)
        if not resolved_manifest.exists():
            continue
        round_manifest = load_json(resolved_manifest)
        collection_run = str(round_manifest.get("collection_run", "")).strip()
        if not collection_run:
            continue
        summary_path = (
            repo_root
            / "matches"
            / "runs"
            / collection_run
            / f"{collection_run}_summary.json"
        ).resolve()
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        group_member_stats = summary.get("group_member_stats", {}) or {}
        opponent_members = group_member_stats.get("b", {}) if isinstance(group_member_stats, dict) else {}
        if not isinstance(opponent_members, dict):
            continue
        for opponent_id, raw_stats in opponent_members.items():
            if not isinstance(raw_stats, dict) or not str(opponent_id).strip():
                continue
            opponent_wins = max(0, int(raw_stats.get("wins", 0)))
            opponent_losses = max(0, int(raw_stats.get("losses", 0)))
            draws = max(0, int(raw_stats.get("draws", 0)))
            matches_played = opponent_wins + opponent_losses + draws
            if matches_played <= 0:
                continue
            learner_wins = opponent_losses
            learner_losses = opponent_wins
            opponent_key = str(opponent_id)
            stats = aggregate.setdefault(opponent_key, OpponentStats(source_run=source_run))
            stats.matches_played += matches_played
            stats.wins += learner_wins
            stats.losses += learner_losses
            stats.draws += draws
            stats.recent_win_rate = learner_wins / matches_played
            stats.source_run = source_run
    return aggregate


def build_live_command(
    args: argparse.Namespace,
    repo_root: Path,
    pool_path: Path,
    parent_checkpoint: Path,
    *,
    run_name: str | None = None,
    rounds: int | None = None,
    pool_seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str((repo_root / "py" / "tools" / "live_rl_orchestrator.py").resolve()),
        "--run-name",
        run_name or args.run_name,
        "--training-mode",
        args.training_mode,
        "--init-checkpoint",
        str(parent_checkpoint),
        "--rounds",
        str(rounds if rounds is not None else args.rounds),
        "--games",
        str(args.games),
        "--concurrent-games",
        str(args.concurrent_games),
        "--worker-pairs",
        str(args.worker_pairs),
        "--ensure-shard-count",
        "true" if args.ensure_shard_count else "false",
        "--model-b-pool",
        str(pool_path),
        "--model-b",
        "",
        "--pool-seed",
        str(pool_seed if pool_seed is not None else args.pool_seed),
        "--format",
        args.format,
        "--learning-rate",
        str(args.learning_rate),
        "--episode-limit",
        str(getattr(args, "episode_limit", 0)),
        "--gamma",
        str(args.gamma),
        "--entropy-coef",
        str(args.entropy_coef),
        "--advantage-norm",
        "1" if args.advantage_norm else "0",
        "--gae-lambda",
        str(args.gae_lambda),
        "--ppo-clip-epsilon",
        str(args.ppo_clip_epsilon),
        "--ppo-value-clip-epsilon",
        str(args.ppo_value_clip_epsilon),
        "--target-kl",
        str(args.target_kl),
        "--target-kl-min-episodes",
        str(args.target_kl_min_episodes),
        "--target-kl-min-labels",
        str(args.target_kl_min_labels),
        "--target-kl-hard-multiplier",
        str(args.target_kl_hard_multiplier),
        "--target-kl-hard-consecutive-updates",
        str(args.target_kl_hard_consecutive_updates),
        "--shuffle-seed",
        str(args.shuffle_seed),
        "--ppo-minibatch-episodes",
        str(args.ppo_minibatch_episodes),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-epsilon",
        str(args.adam_epsilon),
        "--anchor-checkpoint",
        str(args.anchor_checkpoint),
        "--anchor-kl-coef",
        str(args.anchor_kl_coef),
        "--reward-mode",
        args.reward_mode,
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
        "--resource-check-seconds",
        str(args.resource_check_seconds),
        "--min-available-memory-gb",
        str(args.min_available_memory_gb),
        "--min-available-pagefile-gb",
        str(args.min_available_pagefile_gb),
        "--stop-on-collapse",
        "true" if args.stop_on_collapse else "false",
        "--omp-threads",
        str(args.omp_threads),
        "--resume",
        "true" if args.resume else "false",
        "--dashboard",
        "false",
        "--dashboard-write-raw-logs",
        "true" if args.dashboard_write_raw_logs else "false",
    ]
    return command


def evaluation_summary_path(repo_root: Path, run_name: str) -> Path:
    return (repo_root / "matches" / "runs" / run_name / f"{run_name}_summary.json").resolve()


def build_eval_command(
    args: argparse.Namespace,
    repo_root: Path,
    run_name: str,
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    candidate_side: str,
    games: int,
    pool_seed: int,
) -> list[str]:
    model_a = candidate_checkpoint if candidate_side == "a" else champion_checkpoint
    model_b = champion_checkpoint if candidate_side == "a" else candidate_checkpoint
    # A replacement block may only need a handful of games. Do not launch more
    # worker pairs than the block can use; doing so adds model-loading latency
    # and allows a small shortfall to overshoot substantially during draining.
    worker_pairs = min(args.eval_worker_pairs, max(1, games))
    concurrent_games = min(args.eval_concurrent_games, worker_pairs)
    return [
        sys.executable,
        str((repo_root / "py" / "tools" / "selfplay_server.py").resolve()),
        "--run-name",
        run_name,
        "--games",
        str(games),
        "--concurrent-games",
        str(concurrent_games),
        "--worker-pairs",
        str(worker_pairs),
        "--worker-games",
        "0",
        "--ensure-shard-count",
        "true",
        "--model-a-pool",
        "",
        "--model-a",
        str(model_a),
        "--model-b-pool",
        "",
        "--model-b",
        str(model_b),
        "--pool-seed",
        str(pool_seed),
        "--format",
        args.format,
        "--worker-think-mode",
        "live",
        "--serve-client",
        "0",
        "--worker-log-stdout",
        "0",
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
        "--resource-check-seconds",
        str(args.resource_check_seconds),
        "--min-available-memory-gb",
        str(args.min_available_memory_gb),
        "--min-available-pagefile-gb",
        str(args.min_available_pagefile_gb),
        "--startup-timeout-seconds",
        str(args.startup_timeout_seconds),
    ]


def balanced_evaluation_artifacts_match(
    summary: dict[str, object],
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    valid_games_per_side: int,
) -> bool:
    models = summary.get("model_specs", {}) or {}
    candidate_path = str((models.get("candidate", {}) or {}).get("path", "")).strip()
    champion_path = str((models.get("champion", {}) or {}).get("path", "")).strip()
    try:
        return (
            summary.get("status") == "completed"
            and summary.get("evaluation_mode") == "balanced_valid_games"
            and int(summary.get("target_valid_games_per_side", 0) or 0) == valid_games_per_side
            and bool(candidate_path)
            and bool(champion_path)
            and resolve_path(resolve_repo_root(), candidate_path) == candidate_checkpoint.resolve()
            and resolve_path(resolve_repo_root(), champion_path) == champion_checkpoint.resolve()
        )
    except (OSError, TypeError, ValueError):
        return False


def run_balanced_valid_evaluation(
    args: argparse.Namespace,
    repo_root: Path,
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    reporter: BaseWorkflowReporter,
    logs_dir: Path,
) -> tuple[Path, dict[str, object]]:
    combined_path = evaluation_summary_path(repo_root, args.eval_run_name)
    if args.resume and combined_path.exists():
        existing = load_json(combined_path)
        if balanced_evaluation_artifacts_match(
            existing, candidate_checkpoint, champion_checkpoint, args.eval_games,
        ):
            for side in ("a", "b"):
                side_result = ((existing.get("side_results", {}) or {}).get(side, {}) or {})
                reporter.state.update_evaluation(
                    side,
                    int(side_result.get("valid_games", 0) or 0),
                    int(side_result.get("invalid_games", 0) or 0),
                )
            reporter.state.finish_evaluation(existing)
            reporter.notice(f"reusing balanced valid-game evaluation: {combined_path}")
            return combined_path, existing

    candidate_stats: dict[str, object] = {}
    champion_stats: dict[str, object] = {}
    run_names: list[str] = []
    side_results: dict[str, dict[str, int | float | list[str]]] = {}
    for side_index, candidate_side in enumerate(("a", "b")):
        side_candidate_stats: dict[str, object] = {}
        side_champion_stats: dict[str, object] = {}
        side_runs: list[str] = []
        for attempt in range(1, args.eval_max_replacement_attempts + 1):
            before = valid_outcome_counts(side_candidate_stats, side_champion_stats)
            shortfall = args.eval_games - int(before["valid_games"])
            if shortfall <= 0:
                break
            run_name = f"{args.eval_run_name}_side_{candidate_side}_attempt_{attempt:02d}"
            summary_path = evaluation_summary_path(repo_root, run_name)
            model_a = candidate_checkpoint if candidate_side == "a" else champion_checkpoint
            model_b = champion_checkpoint if candidate_side == "a" else candidate_checkpoint
            can_resume = (
                args.resume
                and summary_path.exists()
                and evaluation_artifacts_match(summary_path, model_a, model_b, shortfall)
            )
            reporter.state.begin_evaluation(
                candidate_side,
                int(before["valid_games"]),
                int(before["invalid_games"]),
                shortfall,
            )
            reporter.notice(
                f"evaluation side {candidate_side.upper()} attempt {attempt}: "
                f"need {shortfall} valid games"
            )
            if can_resume:
                reporter.notice(f"reusing evaluation block: {run_name}")
            else:
                run_reported_command(
                    build_eval_command(
                        args,
                        repo_root,
                        run_name,
                        candidate_checkpoint,
                        champion_checkpoint,
                        candidate_side,
                        shortfall,
                        args.pool_seed + 100000 + side_index * 1000 + attempt,
                    ),
                    repo_root,
                    reporter,
                    logs_dir / f"{run_name}.log" if args.dashboard_write_raw_logs else None,
                )
            block = load_json(summary_path)
            if block.get("status") != "completed":
                raise SystemExit(f"evaluation block did not complete: {run_name}")
            groups = block.get("group_stats", {}) or {}
            candidate_group = groups.get(candidate_side, {}) or {}
            champion_side = "b" if candidate_side == "a" else "a"
            champion_group = groups.get(champion_side, {}) or {}
            side_candidate_stats = aggregate_group_stats(side_candidate_stats, candidate_group)
            side_champion_stats = aggregate_group_stats(side_champion_stats, champion_group)
            side_runs.append(run_name)
            run_names.append(run_name)
            after = valid_outcome_counts(side_candidate_stats, side_champion_stats)
            reporter.state.update_evaluation(
                candidate_side, int(after["valid_games"]), int(after["invalid_games"]),
            )
            reporter.notice(
                f"evaluation side={candidate_side} valid={int(after['valid_games'])}/{args.eval_games} "
                f"invalid={int(after['invalid_games'])} raw={int(after['raw_games'])}"
            )
        side_outcomes = valid_outcome_counts(side_candidate_stats, side_champion_stats)
        if int(side_outcomes["valid_games"]) < args.eval_games:
            raise SystemExit(
                f"could not collect {args.eval_games} valid games with candidate on side {candidate_side}; "
                f"collected {int(side_outcomes['valid_games'])} valid and "
                f"{int(side_outcomes['invalid_games'])} invalid after "
                f"{args.eval_max_replacement_attempts} attempts"
            )
        side_results[candidate_side] = {**side_outcomes, "run_names": side_runs}
        candidate_stats = aggregate_group_stats(candidate_stats, side_candidate_stats)
        champion_stats = aggregate_group_stats(champion_stats, side_champion_stats)

    valid = valid_outcome_counts(candidate_stats, champion_stats)
    confidence_low, confidence_high = wilson_interval(
        float(valid["valid_score"]), int(valid["valid_games"]),
    )
    collapse_flags = aggregate_collapse_flags(candidate_stats, champion_stats)
    summary: dict[str, object] = {
        "status": "completed",
        "run_name": args.eval_run_name,
        "evaluation_mode": "balanced_valid_games",
        "target_valid_games_per_side": args.eval_games,
        "target_valid_games": 2 * args.eval_games,
        "raw_games": int(valid["raw_games"]),
        "valid_games": int(valid["valid_games"]),
        "invalid_games": int(valid["invalid_games"]),
        "valid_wins": int(valid["valid_wins"]),
        "valid_draws": int(valid["valid_draws"]),
        "valid_score": float(valid["valid_score"]),
        "valid_win_rate": float(valid["valid_win_rate"]),
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "confidence_level": 0.95,
        "run_names": run_names,
        "side_results": side_results,
        "group_stats": {
            "candidate": candidate_stats,
            "champion": champion_stats,
        },
        "candidate_collapse_flags": collapse_flags,
        "model_specs": {
            "candidate": {"kind": "checkpoint", "path": str(candidate_checkpoint)},
            "champion": {"kind": "checkpoint", "path": str(champion_checkpoint)},
        },
        "metric_definitions": {
            "valid_game": "normal earned decision or draw; disconnect/forfeit outcomes excluded",
            "valid_win_rate": "candidate earned wins plus half of valid draws / valid games",
        },
    }
    write_json(combined_path, summary)
    reporter.state.finish_evaluation(summary)
    reporter.notice(
        f"evaluation complete: score={float(valid['valid_win_rate']):.2%} "
        f"95% CI=[{confidence_low:.2%}, {confidence_high:.2%}]"
    )
    return combined_path, summary


def create_member_from_run(
    args: argparse.Namespace,
    registry: LeagueRegistry,
    candidate_checkpoint: Path,
    parent_member: LeagueMember,
    eval_summary_path: Path | None,
    eval_summary: dict[str, object] | None,
    collapse_flags: list[str],
    opponent_stats: dict[str, OpponentStats],
) -> LeagueMember:
    generation = registry.current_generation + 1
    vs_champion_win_rate = None
    if eval_summary is not None:
        vs_champion_win_rate = float(eval_summary.get("valid_win_rate", 0.0) or 0.0)
    return LeagueMember(
        id=args.member_id or args.run_name,
        path=str(candidate_checkpoint),
        generation=generation,
        status="candidate",
        collection_weight=1.0,
        role=args.learner_role,
        parent_id=parent_member.id,
        exploit_target_id=args.exploit_target_id,
        regime="rl",
        source_run=args.run_name,
        experiment_id=args.experiment_id,
        training_config_id=args.training_config_id,
        snapshot_eligible=True,
        notes=args.notes,
        opponent_stats=opponent_stats,
        eval=league_member_from_json({
            "id": "tmp",
            "path": str(candidate_checkpoint),
            "generation": generation,
            "status": "candidate",
            "collection_weight": 1.0,
            "eval": {
                "vs_champion_win_rate": vs_champion_win_rate,
                "summary_path": str(eval_summary_path) if eval_summary_path else "",
                "collapse_flags": collapse_flags,
            },
        }).eval,
    )


def league_promotion_assessment(
    args: argparse.Namespace,
    eval_summary: dict[str, object],
    collapse_flags: list[str],
) -> dict[str, object]:
    group_stats = eval_summary.get("group_stats", {}) or {}
    candidate_stats = group_stats.get("candidate", {}) or {}
    champion_stats = group_stats.get("champion", {}) or {}
    valid_win_rate = float(eval_summary.get("valid_win_rate", 0.0) or 0.0)
    confidence_low = float(eval_summary.get("confidence_low", 0.0) or 0.0)
    confidence_high = float(eval_summary.get("confidence_high", 1.0) or 1.0)
    candidate_tera_rate = float(candidate_stats.get("tera_battle_rate", candidate_stats.get("tera_rate", 0.0)) or 0.0)
    champion_tera_rate = float(champion_stats.get("tera_battle_rate", champion_stats.get("tera_rate", 0.0)) or 0.0)
    required_tera_rate = args.min_promotion_tera_ratio * champion_tera_rate
    clears_point_gate = valid_win_rate >= args.promote_threshold
    clears_confidence_gate = confidence_low > args.promotion_confidence_threshold
    clears_tera_gate = candidate_tera_rate >= required_tera_rate
    collapse_free = not collapse_flags
    promotion_confident = collapse_free and clears_point_gate and clears_confidence_gate and clears_tera_gate
    if not collapse_free:
        status = "collapse_rejected"
    elif not clears_tera_gate:
        status = "tera_rejected"
    elif not clears_point_gate:
        status = "no_winner"
    elif not clears_confidence_gate:
        status = "tentative_winner"
    else:
        status = "confident_winner"
    return {
        "status": status,
        "valid_win_rate": valid_win_rate,
        "valid_games": int(eval_summary.get("valid_games", 0) or 0),
        "invalid_games": int(eval_summary.get("invalid_games", 0) or 0),
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "minimum_win_rate": args.promote_threshold,
        "confidence_threshold": args.promotion_confidence_threshold,
        "candidate_tera_rate": candidate_tera_rate,
        "champion_tera_rate": champion_tera_rate,
        "minimum_tera_ratio": args.min_promotion_tera_ratio,
        "collapse_free": collapse_free,
        "clears_point_gate": clears_point_gate,
        "clears_confidence_gate": clears_confidence_gate,
        "clears_tera_gate": clears_tera_gate,
        "promotion_confident": promotion_confident,
    }


def maybe_promote_candidate(
    args: argparse.Namespace,
    registry: LeagueRegistry,
    candidate: LeagueMember,
    eval_summary: dict[str, object],
    collapse_flags: list[str],
) -> bool:
    assessment = league_promotion_assessment(args, eval_summary, collapse_flags)
    if not assessment["promotion_confident"]:
        return False
    candidate.status = "active"
    candidate.role = "champion" if args.learner_role == "main" else candidate.role
    candidate.promoted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    registry.champion_id = candidate.id
    return True


def round_screen_should_expand(
    eval_summary: dict[str, object],
    collapse_flags: list[str],
    minimum_win_rate: float,
) -> bool:
    return (
        not collapse_flags
        and float(eval_summary.get("valid_win_rate", 0.0) or 0.0) >= minimum_win_rate
    )


def round_candidate_rank(
    eval_summary: dict[str, object],
    collapse_flags: list[str],
) -> tuple[int, float, float, int]:
    """Prefer safe candidates, then evidence strength and sample size."""
    return (
        0 if collapse_flags else 1,
        float(eval_summary.get("confidence_low", 0.0) or 0.0),
        float(eval_summary.get("valid_win_rate", 0.0) or 0.0),
        int(eval_summary.get("valid_games", 0) or 0),
    )


def evaluation_args_for_round(
    args: argparse.Namespace,
    run_name: str,
    games_per_side: int,
) -> argparse.Namespace:
    values = dict(vars(args))
    values["eval_run_name"] = run_name
    values["eval_games"] = games_per_side
    return argparse.Namespace(**values)


class RoundMappedWorkflowReporter:
    """Map a one-round child workflow onto the outer league round index."""

    def __init__(self, reporter: BaseWorkflowReporter, round_index: int, rounds_total: int) -> None:
        self.reporter = reporter
        self.round_index = round_index
        self.rounds_total = rounds_total
        self.state = reporter.state

    def command_started(self, command: list[str]) -> None:
        self.reporter.command_started(command)

    def child_line(self, line: str) -> None:
        mapped = line.replace(
            "round=1/1", f"round={self.round_index}/{self.rounds_total}",
        ).replace(
            "round_completed=1/1",
            f"round_completed={self.round_index}/{self.rounds_total}",
        )
        self.reporter.child_line(mapped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--parent-id", default="")
    parser.add_argument("--learner-role", choices=["main", "main_exploiter", "league_exploiter"], default="main")
    parser.add_argument("--member-id", default="")
    parser.add_argument("--exploit-target-id", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--training-config-id", default="live_ppo_default")
    parser.add_argument("--notes", default="")
    parser.add_argument("--rounds", type=positive_int, default=1)
    parser.add_argument("--games", type=nonnegative_int, required=True)
    parser.add_argument("--concurrent-games", type=positive_int, required=True)
    parser.add_argument("--worker-pairs", type=positive_int, default=60)
    parser.add_argument("--ensure-shard-count", type=parse_bool, default=True)
    parser.add_argument("--pool-seed", type=int, default=1)
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--training-mode", choices=["rl", "ppo"], default="ppo")
    parser.add_argument("--learning-rate", type=float, default=float_default("league_ppo_learning_rate"))
    parser.add_argument("--episode-limit", type=nonnegative_int, default=256, help="Maximum episodes per PPO update; 0 uses the full collected batch")
    parser.add_argument("--gamma", type=float, default=float_default("ppo_gamma"))
    parser.add_argument("--entropy-coef", type=float, default=float_default("league_ppo_entropy_coef"))
    parser.add_argument("--advantage-norm", type=parse_bool, default=bool_default("advantage_norm"))
    parser.add_argument("--gae-lambda", type=float, default=float_default("gae_lambda"))
    parser.add_argument("--ppo-clip-epsilon", type=float, default=float_default("ppo_clip_epsilon"))
    parser.add_argument("--ppo-value-clip-epsilon", type=float, default=float_default("ppo_value_clip_epsilon"))
    parser.add_argument("--target-kl", type=float, default=float_default("ppo_target_kl"))
    parser.add_argument("--target-kl-min-episodes", type=positive_int, default=int_default("ppo_target_kl_min_episodes"))
    parser.add_argument("--target-kl-min-labels", type=positive_int, default=int_default("ppo_target_kl_min_labels"))
    parser.add_argument("--target-kl-hard-multiplier", type=positive_float, default=float_default("ppo_target_kl_hard_multiplier"))
    parser.add_argument("--target-kl-hard-consecutive-updates", type=positive_int, default=int_default("ppo_target_kl_hard_consecutive_updates"))
    parser.add_argument("--shuffle-seed", type=int, default=int_default("ppo_shuffle_seed"))
    parser.add_argument("--ppo-minibatch-episodes", type=positive_int, default=int_default("ppo_minibatch_episodes"))
    parser.add_argument("--adam-beta1", type=float, default=float_default("adam_beta1"))
    parser.add_argument("--adam-beta2", type=float, default=float_default("adam_beta2"))
    parser.add_argument("--adam-epsilon", type=positive_float, default=float_default("adam_epsilon"))
    parser.add_argument("--anchor-checkpoint", default="", help="Fixed PPO anchor; defaults to the selected parent checkpoint")
    parser.add_argument("--anchor-kl-coef", type=float, default=float_default("anchor_kl_coef"))
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.35)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=3.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=6.0)
    parser.add_argument("--stop-on-collapse", type=parse_bool, default=True)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--dashboard", type=parse_bool, default=True)
    parser.add_argument("--dashboard-refresh-per-second", type=positive_float, default=8.0)
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool, default=True)
    parser.add_argument("--include-random-weight", type=float, default=0.0)
    parser.add_argument("--matchup-target-win-rate", type=target_win_rate_float, default=DEFAULT_MATCHUP_TARGET_WIN_RATE)
    parser.add_argument("--matchup-min-weight", type=positive_unit_float, default=DEFAULT_MATCHUP_MIN_WEIGHT)
    parser.add_argument("--matchup-confidence-games", type=positive_int, default=DEFAULT_MATCHUP_CONFIDENCE_GAMES)
    parser.add_argument("--min-category-starts", type=nonnegative_int, default=DEFAULT_MIN_CATEGORY_STARTS)
    parser.add_argument("--eval-games", type=positive_int, default=500, help="Required valid evaluation games per candidate side")
    parser.add_argument("--round-gating", type=parse_bool, default=True, help="Evaluate each round before allowing its checkpoint to become the next parent")
    parser.add_argument("--round-screen-games", type=positive_int, default=100, help="Valid screening games per side after each training round")
    parser.add_argument("--round-expand-threshold", type=float, default=0.50, help="Minimum screening score that advances to full evaluation")
    parser.add_argument("--round-early-stop-patience", type=positive_int, default=1, help="Stop after this many consecutive rejected rounds")
    parser.add_argument("--eval-concurrent-games", type=positive_int, default=40)
    parser.add_argument("--eval-worker-pairs", type=positive_int, default=120)
    parser.add_argument("--eval-max-replacement-attempts", type=positive_int, default=5)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--promote-threshold", type=float, default=float_default("promotion_earned_win_rate"))
    parser.add_argument("--promotion-confidence-threshold", type=float, default=DEFAULT_PROMOTION_CONFIDENCE_THRESHOLD)
    parser.add_argument("--min-promotion-tera-ratio", type=float, default=float_default("promotion_min_tera_baseline_ratio"))
    parser.add_argument("--snapshot-cadence", type=positive_int, default=5)
    parser.add_argument("--max-active-historical", type=positive_int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    for label, value in (
        ("promote-threshold", args.promote_threshold),
        ("promotion-confidence-threshold", args.promotion_confidence_threshold),
        ("min-promotion-tera-ratio", args.min_promotion_tera_ratio),
        ("round-expand-threshold", args.round_expand_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be between 0 and 1")
    repo_root = resolve_repo_root()
    registry_path = resolve_path(repo_root, args.registry)
    registry = load_registry(registry_path)
    parent_member = choose_parent_member(registry, args.parent_id, args.learner_role)
    parent_checkpoint = resolve_path(repo_root, parent_member.path)
    if not parent_checkpoint.exists():
        raise SystemExit(f"parent checkpoint not found: {parent_checkpoint}")
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    champion_checkpoint = resolve_path(repo_root, champion.path) if champion is not None else None
    if champion_checkpoint is not None and not champion_checkpoint.exists():
        raise SystemExit(f"champion checkpoint not found: {champion_checkpoint}")
    if args.anchor_checkpoint:
        anchor_checkpoint = resolve_path(repo_root, args.anchor_checkpoint)
        if not anchor_checkpoint.exists():
            raise SystemExit(f"anchor checkpoint not found: {anchor_checkpoint}")
    else:
        anchor_checkpoint = parent_checkpoint
    args.anchor_checkpoint = str(anchor_checkpoint)

    workflow_dir = (repo_root / DEFAULT_LEAGUE_ROOT / args.run_name).resolve()
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_manifest_path = workflow_dir / f"{args.run_name}_league_manifest.json"
    pool_path = workflow_dir / f"{args.run_name}_opponent_pool.json"
    args.eval_run_name = f"{args.run_name}_vs_champion_balanced_{args.eval_games}_per_side"

    pool_payload = build_weighted_pool(
        registry,
        args.learner_role,
        parent_member.id,
        args.include_random_weight,
        target_win_rate=args.matchup_target_win_rate,
        min_difficulty_weight=args.matchup_min_weight,
        confidence_games=args.matchup_confidence_games,
        min_category_starts=args.min_category_starts,
    )
    write_json(pool_path, pool_payload)

    if args.resume and workflow_manifest_path.exists():
        workflow_manifest = load_json(workflow_manifest_path)
        workflow_manifest["status"] = "running"
        workflow_manifest["resumed_at_unix"] = utc_now_unix()
        workflow_manifest["parent_id"] = parent_member.id
        workflow_manifest["parent_checkpoint"] = parent_member.path
        workflow_manifest["pool_path"] = str(pool_path)
        workflow_manifest["pool_payload"] = pool_payload
        log(f"resuming league workflow from {workflow_manifest_path}")
    else:
        workflow_manifest = {
            "run_name": args.run_name,
            "status": "running",
            "started_at_unix": utc_now_unix(),
            "registry": str(registry_path),
            "learner_role": args.learner_role,
            "parent_id": parent_member.id,
            "parent_checkpoint": parent_member.path,
            "pool_path": str(pool_path),
            "pool_payload": pool_payload,
        }
    workflow_manifest["evaluation_config"] = {
        "mode": "balanced_valid_games",
        "valid_games_per_side": args.eval_games,
        "max_replacement_attempts": args.eval_max_replacement_attempts,
        "promotion_min_win_rate": args.promote_threshold,
        "promotion_confidence_threshold": args.promotion_confidence_threshold,
        "min_promotion_tera_ratio": args.min_promotion_tera_ratio,
        "champion_id": champion.id if champion is not None else "",
        "champion_checkpoint": str(champion_checkpoint) if champion_checkpoint is not None else "",
        "round_gating": args.round_gating,
        "round_screen_games_per_side": args.round_screen_games,
        "round_expand_threshold": args.round_expand_threshold,
        "round_early_stop_patience": args.round_early_stop_patience,
    }
    workflow_manifest["training_config"] = {
        "mode": args.training_mode,
        "learning_rate": args.learning_rate,
        "episode_limit": args.episode_limit,
        "entropy_coef": args.entropy_coef,
        "gae_lambda": args.gae_lambda,
        "ppo_clip_epsilon": args.ppo_clip_epsilon,
        "ppo_value_clip_epsilon": args.ppo_value_clip_epsilon,
        "target_kl": args.target_kl,
        "target_kl_min_episodes": args.target_kl_min_episodes,
        "target_kl_min_labels": args.target_kl_min_labels,
        "target_kl_hard_multiplier": args.target_kl_hard_multiplier,
        "target_kl_hard_consecutive_updates": args.target_kl_hard_consecutive_updates,
        "shuffle_seed": args.shuffle_seed,
        "ppo_minibatch_episodes": args.ppo_minibatch_episodes,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "anchor_checkpoint": args.anchor_checkpoint,
        "anchor_kl_coef": args.anchor_kl_coef,
    }
    workflow_manifest["dashboard_config"] = {
        "enabled": args.dashboard,
        "refresh_per_second": args.dashboard_refresh_per_second,
        "write_raw_logs": args.dashboard_write_raw_logs,
    }
    write_json(workflow_manifest_path, workflow_manifest)

    dashboard_state = WorkflowDashboardState(
        run_name=args.run_name,
        rounds_total=args.rounds,
        games_per_round=args.games,
        evaluation_games_per_side=args.eval_games,
        title="League PPO" if args.training_mode == "ppo" else "League RL",
    )
    dashboard_state.metrics.update({
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "anchor_kl_coef": args.anchor_kl_coef,
        "ppo_clip_epsilon": args.ppo_clip_epsilon,
    })
    progress_writer = DashboardProgressWriter(workflow_manifest_path, workflow_manifest, dashboard_state)
    reporter = select_workflow_reporter(
        args.dashboard, args.dashboard_refresh_per_second, dashboard_state, progress_writer,
    )
    logs_dir = workflow_dir / "logs"
    reporter.start()
    accepted_parent_checkpoint = parent_checkpoint
    round_manifests: list[Path] = []
    round_records: list[dict[str, object]] = []
    round_commands: list[list[str]] = []
    best_round: dict[str, object] | None = None
    consecutive_rejections = 0

    for round_index in range(1, args.rounds + 1):
        round_run_name = f"{args.run_name}_round{round_index:02d}"
        round_parent_checkpoint = accepted_parent_checkpoint
        live_command = build_live_command(
            args,
            repo_root,
            pool_path,
            round_parent_checkpoint,
            run_name=round_run_name,
            rounds=1,
            pool_seed=args.pool_seed + round_index - 1,
        )
        round_commands.append(live_command)
        workflow_manifest["round_commands"] = round_commands
        workflow_manifest["active_round"] = round_index
        write_json(workflow_manifest_path, workflow_manifest)
        reporter.notice(
            f"training gated round {round_index}/{args.rounds} from {round_parent_checkpoint.name}"
        )
        mapped_reporter = RoundMappedWorkflowReporter(reporter, round_index, args.rounds)
        run_reported_command(
            live_command,
            repo_root,
            mapped_reporter,
            logs_dir / f"{round_run_name}_live_rl.log" if args.dashboard_write_raw_logs else None,
        )

        live_manifest_path = (
            repo_root / DEFAULT_RUNS_ROOT / round_run_name / f"{round_run_name}_live_rl_manifest.json"
        ).resolve()
        live_manifest = load_json(live_manifest_path)
        round_candidate_checkpoint = Path(str(live_manifest.get("latest_checkpoint", ""))).resolve()
        if not round_candidate_checkpoint.exists():
            raise SystemExit(
                f"candidate checkpoint not found after round {round_index}: {round_candidate_checkpoint}"
            )
        child_round_manifests = [Path(path) for path in live_manifest.get("round_manifests", [])]
        round_manifests.extend(child_round_manifests)
        child_round_manifest = load_json(child_round_manifests[-1]) if child_round_manifests else {}
        training_collapse_flags = [
            str(flag) for flag in child_round_manifest.get("training_round_collapse_flags", [])
        ]

        screen_summary: dict[str, object] | None = None
        screen_summary_path: Path | None = None
        confirmation_summary: dict[str, object] | None = None
        confirmation_summary_path: Path | None = None
        comparison_summary: dict[str, object] | None = None
        comparison_summary_path: Path | None = None
        comparison_collapse_flags = list(training_collapse_flags)
        gate_assessment: dict[str, object] | None = None
        accepted = not args.round_gating or champion_checkpoint is None
        expanded = False

        if args.round_gating and champion_checkpoint is not None:
            screen_run_name = (
                f"{args.run_name}_round{round_index:02d}_screen_balanced_"
                f"{args.round_screen_games}_per_side"
            )
            screen_args = evaluation_args_for_round(
                args, screen_run_name, args.round_screen_games,
            )
            screen_summary_path, screen_summary = run_balanced_valid_evaluation(
                screen_args,
                repo_root,
                round_candidate_checkpoint,
                round_parent_checkpoint,
                reporter,
                logs_dir,
            )
            screen_flags = sorted(set(
                training_collapse_flags
                + [str(flag) for flag in (screen_summary.get("candidate_collapse_flags", []) or [])]
            ))
            comparison_summary = screen_summary
            comparison_summary_path = screen_summary_path
            comparison_collapse_flags = screen_flags
            expanded = round_screen_should_expand(
                screen_summary, screen_flags, args.round_expand_threshold,
            )
            if expanded:
                confirmation_run_name = (
                    f"{args.run_name}_round{round_index:02d}_confirm_balanced_"
                    f"{args.eval_games}_per_side"
                )
                confirmation_args = evaluation_args_for_round(
                    args, confirmation_run_name, args.eval_games,
                )
                confirmation_summary_path, confirmation_summary = run_balanced_valid_evaluation(
                    confirmation_args,
                    repo_root,
                    round_candidate_checkpoint,
                    round_parent_checkpoint,
                    reporter,
                    logs_dir,
                )
                confirmation_flags = sorted(set(
                    training_collapse_flags
                    + [str(flag) for flag in (
                        confirmation_summary.get("candidate_collapse_flags", []) or []
                    )]
                ))
                comparison_summary = confirmation_summary
                comparison_summary_path = confirmation_summary_path
                comparison_collapse_flags = confirmation_flags
                gate_assessment = league_promotion_assessment(
                    confirmation_args, confirmation_summary, confirmation_flags,
                )
                accepted = bool(gate_assessment["promotion_confident"])
            else:
                gate_assessment = league_promotion_assessment(
                    screen_args, screen_summary, screen_flags,
                )

        if comparison_summary is None:
            comparison_summary = {
                "valid_win_rate": 1.0,
                "confidence_low": 1.0,
                "confidence_high": 1.0,
                "valid_games": 0,
                "invalid_games": 0,
            }

        round_record: dict[str, object] = {
            "round": round_index,
            "run_name": round_run_name,
            "parent_checkpoint": str(round_parent_checkpoint),
            "candidate_checkpoint": str(round_candidate_checkpoint),
            "live_manifest_path": str(live_manifest_path),
            "training_collapse_flags": training_collapse_flags,
            "screen_summary_path": str(screen_summary_path) if screen_summary_path else "",
            "confirmation_summary_path": (
                str(confirmation_summary_path) if confirmation_summary_path else ""
            ),
            "screen_expanded": expanded,
            "accepted_as_next_parent": accepted,
            "gate_assessment": gate_assessment,
            "comparison_valid_win_rate": float(
                comparison_summary.get("valid_win_rate", 0.0) or 0.0
            ),
            "comparison_confidence_low": float(
                comparison_summary.get("confidence_low", 0.0) or 0.0
            ),
            "comparison_collapse_flags": comparison_collapse_flags,
        }
        round_records.append(round_record)
        workflow_manifest["round_records"] = round_records

        ranked_round = {
            "round": round_index,
            "candidate_checkpoint": round_candidate_checkpoint,
            "parent_checkpoint": round_parent_checkpoint,
            "summary": comparison_summary,
            "summary_path": comparison_summary_path,
            "collapse_flags": comparison_collapse_flags,
            "accepted": accepted,
        }
        if accepted or best_round is None or (
            not bool(best_round.get("accepted"))
            and round_candidate_rank(comparison_summary, comparison_collapse_flags)
            > round_candidate_rank(
                best_round["summary"], best_round["collapse_flags"],  # type: ignore[arg-type]
            )
        ):
            best_round = ranked_round

        if accepted:
            accepted_parent_checkpoint = round_candidate_checkpoint
            consecutive_rejections = 0
            reporter.notice(f"round {round_index} accepted as the next parent")
        else:
            consecutive_rejections += 1
            reporter.notice(
                f"round {round_index} rejected; retaining {round_parent_checkpoint.name}"
            )
        reporter.state.rounds_completed = round_index
        reporter.state.round_index = round_index
        reporter.refresh()
        workflow_manifest["accepted_parent_checkpoint"] = str(accepted_parent_checkpoint)
        workflow_manifest["best_round"] = int(best_round["round"])
        write_json(workflow_manifest_path, workflow_manifest)

        if (
            args.round_gating
            and not accepted
            and consecutive_rejections >= args.round_early_stop_patience
        ):
            workflow_manifest["early_stopped"] = True
            workflow_manifest["early_stop_round"] = round_index
            reporter.notice(
                f"early stopping after {consecutive_rejections} consecutive rejected round(s)"
            )
            break

    if best_round is None:
        raise SystemExit("no training round completed")

    candidate_checkpoint = Path(best_round["candidate_checkpoint"])
    collapse_flags = [str(flag) for flag in best_round["collapse_flags"]]
    eval_summary: dict[str, object] | None = best_round["summary"]  # type: ignore[assignment]
    eval_summary_path = best_round["summary_path"]
    promotion_assessment = None

    # A later accepted parent was evaluated against its predecessor. Promotion
    # still requires a direct, fully sized comparison with the registry champion.
    best_baseline = Path(best_round["parent_checkpoint"]).resolve()
    best_has_full_eval = (
        eval_summary is not None
        and int(eval_summary.get("target_valid_games_per_side", 0) or 0) == args.eval_games
    )
    if champion is not None and champion_checkpoint is not None:
        if not args.round_gating or (
            best_baseline != champion_checkpoint.resolve() and best_has_full_eval
        ):
            final_args = evaluation_args_for_round(
                args,
                f"{args.run_name}_best_vs_champion_balanced_{args.eval_games}_per_side",
                args.eval_games,
            )
            eval_summary_path, eval_summary = run_balanced_valid_evaluation(
                final_args,
                repo_root,
                candidate_checkpoint,
                champion_checkpoint,
                reporter,
                logs_dir,
            )
            collapse_flags = sorted(set(
                collapse_flags
                + [str(flag) for flag in (eval_summary.get("candidate_collapse_flags", []) or [])]
            ))
        assert eval_summary is not None
        promotion_assessment = league_promotion_assessment(args, eval_summary, collapse_flags)
        reporter.state.collapse_flags = list(collapse_flags)
        reporter.state.set_promotion(promotion_assessment)
        reporter.notice(f"promotion assessment: {promotion_assessment['status']}")
    else:
        eval_summary = None
        eval_summary_path = None

    recent_opponent_stats = collect_recent_opponent_stats(repo_root, round_manifests, args.run_name)
    if recent_opponent_stats:
        parent_member.opponent_stats = recent_opponent_stats

    candidate_id = args.member_id or args.run_name
    if registry_has_member(registry, candidate_id):
        candidate = find_member(registry, candidate_id)
        candidate.path = str(candidate_checkpoint)
        candidate.parent_id = parent_member.id
        if recent_opponent_stats:
            candidate.opponent_stats = recent_opponent_stats
    else:
        candidate = create_member_from_run(
            args,
            registry,
            candidate_checkpoint,
            parent_member,
            eval_summary_path,
            eval_summary,
            collapse_flags,
            recent_opponent_stats,
        )
        registry.members.append(candidate)
        registry.current_generation = max(registry.current_generation, candidate.generation)
    if eval_summary is not None:
        candidate.eval.vs_champion_win_rate = float(eval_summary.get("valid_win_rate", 0.0) or 0.0)
        candidate.eval.summary_path = str(eval_summary_path) if eval_summary_path else ""
        candidate.eval.collapse_flags = list(collapse_flags)

    if args.learner_role == "main" and len(round_records) >= args.snapshot_cadence:
        snapshot_member = LeagueMember(
            id=f"{candidate.id}_snapshot",
            path=str(candidate_checkpoint),
            generation=candidate.generation,
            status="active",
            collection_weight=1.0,
            role="historical_snapshot",
            parent_id=candidate.id,
            regime="rl",
            source_run=args.run_name,
            experiment_id=args.experiment_id,
            training_config_id=args.training_config_id,
            snapshot_eligible=False,
            opponent_stats=dict(recent_opponent_stats),
        )
        registry.members.append(snapshot_member)

    if eval_summary is not None:
        maybe_promote_candidate(args, registry, candidate, eval_summary, collapse_flags)

    historical_active = [member for member in registry.members if member.role == "historical_snapshot" and member.status == "active"]
    for member in sort_recent(historical_active)[args.max_active_historical:]:
        member.status = "inactive"

    save_registry(registry_path, registry)

    workflow_manifest["status"] = "completed"
    workflow_manifest["completed_at_unix"] = utc_now_unix()
    workflow_manifest["candidate_id"] = candidate.id
    workflow_manifest["candidate_checkpoint"] = str(candidate_checkpoint)
    workflow_manifest["best_round"] = int(best_round["round"])
    workflow_manifest["rounds_completed"] = len(round_records)
    workflow_manifest["accepted_parent_checkpoint"] = str(accepted_parent_checkpoint)
    workflow_manifest["eval_summary_path"] = str(eval_summary_path) if eval_summary_path else ""
    workflow_manifest["post_eval_collapse_flags"] = collapse_flags
    workflow_manifest["promotion_assessment"] = promotion_assessment
    workflow_manifest["recent_opponent_stats"] = opponent_stats_payload(recent_opponent_stats)
    workflow_manifest["promoted"] = bool(registry.champion_id == candidate.id)
    write_json(workflow_manifest_path, workflow_manifest)
    reporter.close()


if __name__ == "__main__":
    main()
