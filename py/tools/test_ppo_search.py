from __future__ import annotations

import unittest
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ppo_search import (
    EvaluationResult,
    BaseSearchReporter,
    DEFAULT_TRAINER_EXE,
    Hyperparameters,
    ManifestProgressWriter,
    SearchDisplayState,
    aggregate_group_stats,
    adaptive_screen_contenders,
    build_data_scale_summary,
    build_eval_command,
    build_train_command,
    evaluation_artifacts_match,
    load_config_args,
    merge_evaluation_results,
    parse_evaluation_progress,
    parse_search_args,
    parse_training_progress,
    promotion_assessment,
    run_command,
    training_safety_flags,
    training_artifacts_match,
    valid_outcome_counts,
    wilson_interval,
)


def train_args() -> SimpleNamespace:
    return SimpleNamespace(
        policy_tag_expected="parent-tag",
        epochs=1,
        gamma=0.99,
        advantage_norm=True,
        gae_lambda=0.95,
        ppo_value_clip_epsilon=0.2,
        target_kl=0.02,
        target_kl_min_episodes=20,
        target_kl_min_labels=500,
        target_kl_hard_multiplier=4.0,
        target_kl_hard_consecutive_updates=2,
        ppo_minibatch_episodes=8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        reward_mode="terminal",
        dense_additive_hp_swing_weight=0.1,
        dense_additive_faint_swing_weight=0.25,
        dense_additive_reward_clip=0.4,
    )


def eval_args() -> SimpleNamespace:
    return SimpleNamespace(
        eval_concurrent_games=70,
        eval_worker_pairs=125,
        format="gen9randomdoublesbattle",
        launch_stagger_seconds=0.25,
        resource_check_seconds=2.0,
        min_available_memory_gb=2.0,
        min_available_pagefile_gb=4.0,
        startup_timeout_seconds=120,
    )


def safety_args() -> SimpleNamespace:
    return SimpleNamespace(
        target_kl_min_episodes=20,
        target_kl_min_labels=500,
        max_approx_kl=0.02,
        max_anchor_kl_mean=0.05,
        max_anchor_kl_max=0.20,
        max_clip_fraction=0.25,
    )


class PpoSearchTests(unittest.TestCase):
    @staticmethod
    def evaluation(stage: str, wins: int, games: int, low: float, high: float) -> EvaluationResult:
        return EvaluationResult(
            stage=stage,
            games=games,
            wins=wins,
            earned_wins=wins,
            win_rate=wins / games,
            earned_win_rate=wins / games,
            confidence_low=low,
            confidence_high=high,
            collapse_flags=[],
            run_names=[f"{stage}_{games}"],
            valid_games=games,
            invalid_games=0,
            valid_wins=wins,
            valid_score=float(wins),
            valid_win_rate=wins / games,
            candidate_stats=aggregate_group_stats({}, {
                "matches_played": games, "wins": wins, "earned_wins": wins,
                "losses": games - wins,
            }),
            baseline_stats=aggregate_group_stats({}, {
                "matches_played": games, "wins": games - wins, "earned_wins": games - wins,
                "losses": wins,
            }),
        )

    def test_search_config_loads_and_cli_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "search.toml"
            config.write_text(
                'max_trials = 5\nscreen_games_per_side = 40\nresume = false\n',
                encoding="utf-8",
            )
            config_args = load_config_args(config)
            self.assertEqual(config_args[:2], ["--max-trials", "5"])
            args = parse_search_args([
                "--config", str(config),
                "--run-prefix", "test",
                "--init-checkpoint", "parent.chk",
                "--episode-batch", "batch.jsonl",
                "--max-trials", "2",
            ])
            self.assertEqual(args.max_trials, 2)
            self.assertEqual(args.screen_games_per_side, 40)
            self.assertFalse(args.resume)
            self.assertTrue(args.dashboard)
            self.assertEqual(args.episode_limits, [0])

    def test_trainer_executable_is_fixed_and_rejected_in_config(self) -> None:
        self.assertEqual(DEFAULT_TRAINER_EXE.as_posix(), "build-fresh/showdown_client.exe")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "search.toml"
            config.write_text('trainer_exe = "custom.exe"\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fixed internally"):
                load_config_args(config)

    def test_live_progress_parsers_extract_trainer_and_selfplay_state(self) -> None:
        training = parse_training_progress(
            "[train-ppo] epoch=1 episodes=24/128 step=3 approx_kl=0.0123 "
            "anchor_kl_mean=0.0042 anchor_kl_max=0.0190 clip_fraction=0.1250 "
            "hard_kl_breaches=1/2 labels=640\n"
        )
        self.assertIsNotNone(training)
        assert training is not None
        self.assertEqual(training["current"], 24)
        self.assertEqual(training["total"], 128)
        self.assertAlmostEqual(float(training["approx_kl"]), 0.0123)
        self.assertAlmostEqual(float(training["anchor_kl_mean"]), 0.0042)
        self.assertEqual(parse_evaluation_progress("[selfplay] completed_games=71/100\n"), (71, 100))

    def test_dashboard_progress_payload_tracks_active_operation(self) -> None:
        state = SearchDisplayState("search", 6, 4, 2, 50, 100)
        params = Hyperparameters(1e-5, 3e-4, 0.01, 0.2, 101)
        state.phase = "training"
        state.begin_operation("training", "trial", 1, params, 128)
        state.active_current = 24
        state.active_metrics = {"approx_kl": 0.0123}
        payload = state.progress_payload()
        self.assertEqual(payload["phase"], "training")
        active = payload["active"]
        assert isinstance(active, dict)
        self.assertEqual(active["current"], 24)
        self.assertEqual(active["total"], 128)

    def test_evaluation_blocks_merge_counts_and_recompute_confidence(self) -> None:
        first = self.evaluation("screen", 55, 100, 0.45, 0.64)
        second = self.evaluation("screen", 50, 100, 0.40, 0.60)
        merged = merge_evaluation_results(first, second)
        self.assertEqual(merged.games, 200)
        self.assertEqual(merged.wins, 105)
        self.assertAlmostEqual(merged.win_rate, 0.525)
        self.assertEqual(merged.valid_games, 200)
        self.assertAlmostEqual(merged.valid_win_rate, 0.525)
        self.assertLess(merged.confidence_low, merged.win_rate)
        self.assertGreater(merged.confidence_high, merged.win_rate)
        self.assertEqual(merged.run_names, ["screen_100", "screen_100"])

    def test_evaluation_merge_recomputes_collapse_from_aggregate_behavior(self) -> None:
        balanced_counts = {
            "matches_played": 100,
            "wins": 55,
            "earned_wins": 55,
            "losses": 45,
            "tera_battles": 80,
            "total_move_slot_1": 25,
            "total_move_slot_2": 25,
            "total_move_slot_3": 25,
            "total_move_slot_4": 25,
            "total_switch_slot_3": 25,
            "total_switch_slot_4": 25,
            "total_switch_slot_5": 25,
            "total_switch_slot_6": 25,
        }
        stats = aggregate_group_stats({}, balanced_counts)
        first = self.evaluation("screen", 55, 100, 0.45, 0.64)
        second = self.evaluation("screen", 55, 100, 0.45, 0.64)
        first.collapse_flags = ["transient_block_warning"]
        first.candidate_stats = stats
        first.baseline_stats = stats
        second.candidate_stats = stats
        second.baseline_stats = stats
        merged = merge_evaluation_results(first, second)
        self.assertNotIn("transient_block_warning", merged.collapse_flags)
        self.assertEqual(merged.candidate_stats["matches_played"], 200)

    def test_adaptive_screening_refines_only_an_unresolved_cutoff(self) -> None:
        first = SimpleNamespace(run_name="first", screen_evaluation=self.evaluation("screen", 56, 100, 0.47, 0.65))
        cutoff = SimpleNamespace(run_name="cutoff", screen_evaluation=self.evaluation("screen", 52, 100, 0.43, 0.61))
        challenger = SimpleNamespace(run_name="challenger", screen_evaluation=self.evaluation("screen", 48, 100, 0.39, 0.52))
        requested = {"first": 100, "cutoff": 100, "challenger": 100}
        contenders = adaptive_screen_contenders([first, cutoff, challenger], 2, requested, 200)
        self.assertEqual([trial.run_name for trial in contenders], ["first", "cutoff", "challenger"])
        challenger.screen_evaluation.confidence_high = 0.42
        self.assertEqual(adaptive_screen_contenders([first, cutoff, challenger], 2, requested, 200), [])

    def test_promotion_gate_distinguishes_no_tentative_and_confident_winner(self) -> None:
        trial = SimpleNamespace(run_name="candidate", final_evaluation=self.evaluation("final", 49, 100, 0.39, 0.59))
        winner, assessment = promotion_assessment([trial], 0.5, 0.5)
        self.assertIsNone(winner)
        self.assertEqual(assessment["status"], "no_winner")
        trial.final_evaluation = self.evaluation("final", 52, 100, 0.42, 0.62)
        winner, assessment = promotion_assessment([trial], 0.5, 0.5)
        self.assertIs(winner, trial)
        self.assertEqual(assessment["status"], "tentative_winner")
        trial.final_evaluation = self.evaluation("final", 550, 1000, 0.519, 0.581)
        winner, assessment = promotion_assessment([trial], 0.5, 0.5)
        self.assertIs(winner, trial)
        self.assertEqual(assessment["status"], "confident_winner")
        self.assertTrue(assessment["promotion_confident"])

    def test_streamed_command_updates_progress_and_writes_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = SearchDisplayState("search", 1, 1, 1, 10, 10)
            params = Hyperparameters(1e-5, 3e-4, 0.01, 0.2, 101)
            state.begin_operation("training", "trial", 1, params, 16)
            manifest: dict[str, object] = {}
            writer = ManifestProgressWriter(root / "manifest.json", manifest, state)
            reporter = BaseSearchReporter(SimpleNamespace(), state, writer)
            raw_log = root / "trial.log"
            run_command([
                sys.executable,
                "-c",
                "print('[train-ppo] epoch=1 episodes=8/16 approx_kl=0.01 anchor_kl_mean=0.002 clip_fraction=0.1')",
            ], root, reporter, raw_log)
            self.assertEqual(state.active_current, 8)
            self.assertEqual(state.active_total, 16)
            self.assertAlmostEqual(float(state.active_metrics["approx_kl"]), 0.01)
            self.assertIn("episodes=8/16", raw_log.read_text(encoding="utf-8"))

    def test_train_command_uses_fixed_batch_anchor_and_reproducible_guards(self) -> None:
        params = Hyperparameters(2.5e-5, 3e-4, 0.01, 0.2, 101)
        command = build_train_command(
            train_args(),
            Path("trainer.exe"),
            Path("fixed.jsonl"),
            Path("parent.chk"),
            Path("anchor.chk"),
            Path("candidate.chk"),
            Path("summary.json"),
            params,
        )
        self.assertEqual(command[2], "fixed.jsonl")
        self.assertEqual(command[command.index("--parent-checkpoint") + 1], "parent.chk")
        self.assertEqual(command[command.index("--anchor-checkpoint") + 1], "anchor.chk")
        self.assertEqual(command[command.index("--anchor-kl-coef") + 1], "0.01")
        self.assertEqual(command[command.index("--shuffle-seed") + 1], "101")
        self.assertEqual(command[command.index("--episode-limit") + 1], "0")
        self.assertEqual(command[command.index("--target-kl-min-labels") + 1], "500")
        self.assertEqual(command[command.index("--target-kl-hard-consecutive-updates") + 1], "2")
        self.assertEqual(command[command.index("--ppo-minibatch-episodes") + 1], "8")

    def test_eval_command_clears_both_pools_and_swaps_candidate_side(self) -> None:
        command = build_eval_command(
            eval_args(),
            Path("repo"),
            "eval-run",
            Path("candidate.chk"),
            Path("parent.chk"),
            "b",
            250,
            7,
        )
        self.assertEqual(command[command.index("--model-a-pool") + 1], "")
        self.assertEqual(command[command.index("--model-b-pool") + 1], "")
        self.assertEqual(command[command.index("--model-a") + 1], "parent.chk")
        self.assertEqual(command[command.index("--model-b") + 1], "candidate.chk")

    def test_safety_flags_reject_hard_stop_and_inactive_anchor(self) -> None:
        flags = training_safety_flags({
            "episode_count": 4,
            "labels": 82,
            "approx_kl": 0.005,
            "target_kl_exceeded": True,
            "target_kl_hard_stop": True,
            "anchor_kl_mean": 0.0,
            "anchor_kl_max": 0.0,
            "clip_fraction": 0.01,
            "tera_action_rate": 0.03,
            "move_slot_rates": {},
        }, safety_args(), 0.01)
        self.assertIn("target_kl_hard_stop", flags)
        self.assertIn("anchor_inactive", flags)
        self.assertTrue(any(flag.startswith("target_kl_before_min_episodes") for flag in flags))
        self.assertTrue(any(flag.startswith("target_kl_before_min_labels") for flag in flags))

    def test_wilson_interval_reflects_finite_sample_uncertainty(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

    def test_resume_requires_matching_training_provenance(self) -> None:
        args = train_args()
        params = Hyperparameters(2.5e-5, 3e-4, 0.01, 0.2, 101)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            batch = root / "batch.jsonl"
            parent = root / "parent.chk"
            anchor = root / "anchor.chk"
            candidate = root / "candidate.chk"
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "input_episode_batch": str(batch),
                "parent_checkpoint": str(parent),
                "anchor_checkpoint": str(anchor),
                "output_checkpoint": str(candidate),
                "learning_rate": params.learning_rate,
                "entropy_coef": params.entropy_coef,
                "anchor_kl_coef": params.anchor_kl_coef,
                "ppo_clip_epsilon": params.ppo_clip_epsilon,
                "shuffle_seed": params.shuffle_seed,
                "episode_limit": params.episode_limit,
                "minibatch_episodes": args.ppo_minibatch_episodes,
                "target_kl_hard_consecutive_updates": args.target_kl_hard_consecutive_updates,
            }), encoding="utf-8")
            self.assertTrue(training_artifacts_match(
                summary, batch, parent, anchor, candidate, params, args,
            ))
            mismatched = Hyperparameters(5e-5, 3e-4, 0.01, 0.2, 101)
            self.assertFalse(training_artifacts_match(
                summary, batch, parent, anchor, candidate, mismatched, args,
            ))

    def test_resume_requires_matching_evaluation_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            candidate = root / "candidate.chk"
            parent = root / "parent.chk"
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "status": "completed",
                "target_games": 250,
                "model_specs": {
                    "a": {"kind": "checkpoint", "path": str(candidate)},
                    "b": {"kind": "checkpoint", "path": str(parent)},
                },
            }), encoding="utf-8")
            self.assertTrue(evaluation_artifacts_match(summary, candidate, parent, 250))
            self.assertFalse(evaluation_artifacts_match(summary, parent, candidate, 250))

    def test_valid_outcomes_exclude_disconnect_and_forfeit_results(self) -> None:
        candidate = {"matches_played": 100, "wins": 53, "earned_wins": 49, "draws": 2}
        baseline = {"matches_played": 100, "wins": 47, "earned_wins": 44, "draws": 2}
        counts = valid_outcome_counts(candidate, baseline)
        self.assertEqual(counts["raw_games"], 100)
        self.assertEqual(counts["valid_games"], 95)
        self.assertEqual(counts["invalid_games"], 5)
        self.assertEqual(counts["valid_score"], 50.0)
        self.assertAlmostEqual(float(counts["valid_win_rate"]), 50.0 / 95.0)

    def test_data_scale_summary_pools_seed_results_by_episode_limit(self) -> None:
        trials = []
        for seed, wins in ((101, 55), (202, 45)):
            result = self.evaluation("final", wins, 100, 0.40, 0.60)
            result.candidate_stats = aggregate_group_stats({}, {
                "matches_played": 100, "wins": wins, "earned_wins": wins,
                "losses": 100 - wins,
            })
            result.baseline_stats = aggregate_group_stats({}, {
                "matches_played": 100, "wins": 100 - wins, "earned_wins": 100 - wins,
                "losses": wins,
            })
            trials.append(SimpleNamespace(
                hyperparameters=Hyperparameters(1e-5, 3e-4, 0.01, 0.2, seed, 128),
                safety_flags=[], final_evaluation=result, screen_evaluation=None,
            ))
        summary = build_data_scale_summary(trials)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["episode_limit"], 128)
        self.assertEqual(summary[0]["configured_seeds"], [101, 202])
        self.assertEqual(summary[0]["valid_games"], 200)
        self.assertAlmostEqual(float(summary[0]["pooled_valid_win_rate"]), 0.5)


if __name__ == "__main__":
    unittest.main()
