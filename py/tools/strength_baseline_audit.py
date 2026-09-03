from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkModel:
    id: str
    label: str
    checkpoint: str
    provenance: str
    legacy: bool = False


MODEL_FIELDS = (
    ("g4", "Original g4", "g4_checkpoint", "g4_provenance", False),
    (
        "current_arch_full",
        "Current architecture supervised",
        "current_arch_checkpoint",
        "current_arch_provenance",
        False,
    ),
    (
        "current_arch_ep10_legacy",
        "Current architecture ep10 (legacy)",
        "legacy_ep10_checkpoint",
        "legacy_ep10_provenance",
        True,
    ),
    (
        "current_arch_ep20_legacy",
        "Current architecture ep20 (legacy)",
        "legacy_ep20_checkpoint",
        "legacy_ep20_provenance",
        True,
    ),
    ("run_0096_anchored_ppo", "run_0096 anchored PPO", "run_0096_checkpoint", "run_0096_provenance", False),
    (
        "run_0111_top_observed",
        "run_0111 top observed candidate",
        "run_0111_checkpoint",
        "run_0111_provenance",
        False,
    ),
)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def benchmark_models(args: argparse.Namespace, repo_root: Path) -> list[BenchmarkModel]:
    models: list[BenchmarkModel] = []
    for model_id, label, checkpoint_field, provenance_field, legacy in MODEL_FIELDS:
        checkpoint = resolve_path(repo_root, getattr(args, checkpoint_field))
        if not checkpoint.exists():
            raise SystemExit(f"benchmark checkpoint not found for {model_id}: {checkpoint}")
        models.append(
            BenchmarkModel(
                id=model_id,
                label=label,
                checkpoint=str(checkpoint),
                provenance=str(getattr(args, provenance_field)),
                legacy=legacy,
            )
        )
    if len({model.checkpoint for model in models}) != len(models):
        raise SystemExit("benchmark checkpoints must be six distinct artifact paths")
    return models


def evaluation_record(
    model: BenchmarkModel,
    stage: str,
    summary_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    candidate_stats = ((summary.get("group_stats", {}) or {}).get("candidate", {}) or {})
    model_specs = summary.get("model_specs", {}) or {}
    return {
        "model": asdict(model),
        "stage": stage,
        "summary_path": str(summary_path),
        "baseline_spec": model_specs.get("champion", {}),
        "raw_games": int(summary.get("raw_games", 0) or 0),
        "valid_games": int(summary.get("valid_games", 0) or 0),
        "invalid_games": int(summary.get("invalid_games", 0) or 0),
        "valid_wins": int(summary.get("valid_wins", 0) or 0),
        "valid_draws": int(summary.get("valid_draws", 0) or 0),
        "valid_score": float(summary.get("valid_score", 0.0) or 0.0),
        "valid_win_rate": float(summary.get("valid_win_rate", 0.0) or 0.0),
        "confidence_low": float(summary.get("confidence_low", 0.0) or 0.0),
        "confidence_high": float(summary.get("confidence_high", 0.0) or 0.0),
        "side_results": summary.get("side_results", {}),
        "tera_battle_rate": float(candidate_stats.get("tera_battle_rate", 0.0) or 0.0),
        "tera_rate": float(candidate_stats.get("tera_rate", 0.0) or 0.0),
        "collapse_flags": list(summary.get("candidate_collapse_flags", []) or []),
    }


def rank_evaluations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(
        records,
        key=lambda record: (
            float(record.get("confidence_low", 0.0)),
            float(record.get("valid_win_rate", 0.0)),
            str((record.get("model", {}) or {}).get("id", "")),
        ),
        reverse=True,
    )
    return [{**record, "rank": index} for index, record in enumerate(ranked, start=1)]


def head_to_head_record(
    candidate: BenchmarkModel,
    baseline: BenchmarkModel,
    summary_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    record = evaluation_record(candidate, "head_to_head", summary_path, summary)
    record["baseline_model"] = asdict(baseline)
    return record


def benchmark_contract(args: argparse.Namespace, models: list[BenchmarkModel]) -> dict[str, object]:
    return {
        "models": [asdict(model) for model in models],
        "screen_games_per_side": args.screen_games_per_side,
        "final_games_per_side": args.final_games_per_side,
        "head_to_head_games_per_side": args.head_to_head_games_per_side,
        "finalist_count": args.finalist_count,
        "screen_pool_seed": args.screen_pool_seed,
        "final_pool_seed": args.final_pool_seed,
        "head_to_head_pool_seed": args.head_to_head_pool_seed,
        "format": args.format,
    }


def build_learning_audit(
    run_name: str,
    trainer_tests: dict[str, object],
    screening_ranking: list[dict[str, object]],
    final_ranking: list[dict[str, object]],
    head_to_head: dict[str, object],
) -> dict[str, object]:
    strongest = final_ranking[0]
    strongest_model = strongest["model"]
    verified = float(strongest["confidence_low"]) > 0.5
    return {
        "run_name": run_name,
        "status": "completed",
        "pipeline_ready": bool(trainer_tests.get("passed")),
        "strength_baseline_verified": verified,
        "strongest_checkpoint": strongest_model["checkpoint"],
        "strongest_checkpoint_provenance": strongest_model,
        "confidence_interval_against_random": {
            "point_estimate": strongest["valid_win_rate"],
            "lower_95": strongest["confidence_low"],
            "upper_95": strongest["confidence_high"],
            "valid_games": strongest["valid_games"],
            "invalid_games": strongest["invalid_games"],
        },
        "strongest_collapse_flags": strongest["collapse_flags"],
        "trainer_test_results": trainer_tests,
        "recommended_starting_checkpoint": strongest_model["checkpoint"] if verified else "",
        "recommendation": (
            "Use the strongest verified checkpoint as the starting point for iterative self-play."
            if verified
            else "Do not start iterative self-play; no checkpoint has a lower 95% bound above random."
        ),
        "screening_ranking": screening_ranking,
        "final_random_ranking": final_ranking,
        "head_to_head": head_to_head,
    }
