from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from league_rl_orchestrator import balanced_evaluation_artifacts_match
from ppo_search import evaluation_artifacts_match
from strength_baseline_benchmark import (
    DEFAULT_CONFIG_PATH,
    BenchmarkModel,
    balanced_eval_command,
    build_parser,
    build_learning_audit,
    evaluation_record,
    load_config_args,
    rank_evaluations,
)


def model(model_id: str) -> BenchmarkModel:
    return BenchmarkModel(
        id=model_id,
        label=model_id,
        checkpoint=str(Path(f"{model_id}.chk").resolve()),
        provenance=f"{model_id} provenance",
    )


def evaluation(model_id: str, low: float, point: float) -> dict[str, object]:
    return {
        "model": model(model_id).__dict__,
        "confidence_low": low,
        "confidence_high": min(1.0, point + 0.04),
        "valid_win_rate": point,
        "valid_games": 2000,
        "invalid_games": 5,
        "collapse_flags": [],
    }


class RandomBaselineTests(unittest.TestCase):
    def test_canonical_config_names_all_six_models(self) -> None:
        args = build_parser().parse_args(load_config_args(DEFAULT_CONFIG_PATH))

        self.assertEqual(args.screen_games_per_side, 250)
        self.assertEqual(args.final_games_per_side, 1000)
        self.assertEqual(args.head_to_head_games_per_side, 1000)
        self.assertIn("run_0111_ppo_search_multi_batch_full_03", args.run_0111_checkpoint)

    def test_balanced_resume_accepts_random_model_spec(self) -> None:
        candidate = Path("candidate.chk").resolve()
        summary = {
            "status": "completed",
            "evaluation_mode": "balanced_valid_games",
            "target_valid_games_per_side": 250,
            "battle_seed_base": 12001,
            "model_specs": {
                "candidate": {"kind": "checkpoint", "path": str(candidate)},
                "champion": {"kind": "random"},
            },
        }

        self.assertTrue(
            balanced_evaluation_artifacts_match(summary, candidate, "random", 250, 12001)
        )
        self.assertFalse(
            balanced_evaluation_artifacts_match(summary, candidate, "random", 250, 12002)
        )
        self.assertFalse(balanced_evaluation_artifacts_match(summary, candidate, "other.chk", 250))

    def test_selfplay_block_resume_accepts_random_model_spec(self) -> None:
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir, "candidate.chk").resolve()
            summary_path = Path(temp_dir, "summary.json")
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "target_games": 250,
                        "model_specs": {
                            "a": {"kind": "checkpoint", "path": str(candidate)},
                            "b": {"kind": "random"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(evaluation_artifacts_match(summary_path, candidate, "random", 250))

    def test_driver_passes_random_to_balanced_evaluator(self) -> None:
        args = SimpleNamespace(
            concurrent_games=50,
            worker_pairs=50,
            max_replacement_attempts=5,
            format="gen9randomdoublesbattle",
            launch_stagger_seconds=0.25,
            resource_check_seconds=2.0,
            min_available_memory_gb=2.0,
            min_available_pagefile_gb=4.0,
            startup_timeout_seconds=120,
            dashboard=True,
            dashboard_refresh_per_second=8.0,
            dashboard_write_raw_logs=True,
            resume=True,
        )

        command = balanced_eval_command(
            args,
            Path.cwd(),
            "audit_screen_g4",
            str(Path("g4.chk").resolve()),
            "random",
            250,
            12001,
        )

        self.assertEqual(command[command.index("--baseline") + 1], "random")
        self.assertEqual(command[command.index("--games-per-side") + 1], "250")
        self.assertEqual(command[command.index("--battle-seed-base") + 1], "12001")


class AuditRankingTests(unittest.TestCase):
    def test_ranking_uses_lower_wilson_bound_before_point_estimate(self) -> None:
        ranked = rank_evaluations(
            [
                evaluation("higher_point", 0.51, 0.60),
                evaluation("higher_bound", 0.52, 0.55),
            ]
        )

        self.assertEqual(ranked[0]["model"]["id"], "higher_bound")
        self.assertEqual([record["rank"] for record in ranked], [1, 2])

    def test_audit_verifies_only_a_strictly_above_random_lower_bound(self) -> None:
        trainer_tests = {"passed": True}
        screen = rank_evaluations([evaluation("best", 0.53, 0.57)])
        final = rank_evaluations([evaluation("best", 0.5001, 0.54)])

        audit = build_learning_audit("audit", trainer_tests, screen, final, {})

        self.assertTrue(audit["pipeline_ready"])
        self.assertTrue(audit["strength_baseline_verified"])
        self.assertEqual(audit["recommended_starting_checkpoint"], model("best").checkpoint)

        final[0]["confidence_low"] = 0.5
        unverified = build_learning_audit("audit", trainer_tests, screen, final, {})
        self.assertFalse(unverified["strength_baseline_verified"])
        self.assertEqual(unverified["recommended_starting_checkpoint"], "")

    def test_evaluation_record_retains_audit_diagnostics(self) -> None:
        candidate = model("candidate")
        summary = {
            "valid_games": 500,
            "invalid_games": 7,
            "valid_wins": 280,
            "valid_draws": 2,
            "valid_score": 281.0,
            "valid_win_rate": 0.562,
            "confidence_low": 0.518,
            "confidence_high": 0.605,
            "side_results": {"a": {"valid_games": 250}, "b": {"valid_games": 250}},
            "group_stats": {"candidate": {"tera_battle_rate": 0.8, "tera_rate": 0.8}},
            "candidate_collapse_flags": ["move_slot_1_hard_collapse"],
        }

        record = evaluation_record(candidate, "screen", Path("summary.json"), summary)

        self.assertEqual(record["invalid_games"], 7)
        self.assertEqual(record["side_results"]["a"]["valid_games"], 250)
        self.assertEqual(record["tera_battle_rate"], 0.8)
        self.assertEqual(record["collapse_flags"], ["move_slot_1_hard_collapse"])


if __name__ == "__main__":
    unittest.main()
