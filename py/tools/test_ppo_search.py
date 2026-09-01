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
    ConfirmationOpponent,
    DEFAULT_TRAINER_EXE,
    Hyperparameters,
    ManifestProgressWriter,
    SearchDisplayState,
    aggregate_group_stats,
    adaptive_screen_contenders,
    adaptive_screen_group_contenders,
    build_data_scale_summary,
    build_hyperparameter_group_summary,
    build_robust_confirmation_group_summary,
    build_eval_command,
    build_train_command,
    configured_boundary_diagnostics,
    discover_confirmation_opponents,
    evaluation_artifacts_match,
    fresh_confirmation_comparison,
    inferred_policy_tag,
    load_config_args,
    merge_evaluation_results,
    parse_evaluation_progress,
    paired_evaluation_seed,
    parse_search_args,
    parse_training_progress,
    promotion_assessment,
    robust_grouped_promotion_assessment,
    resolve_search_episode_batches,
    resolve_evaluation_seed_suites,
    grouped_promotion_assessment,
    generate_local_refinement_space,
    run_command,
    select_search_combinations,
    select_space_filling_settings,
    suggest_bayesian_setting,
    training_safety_flags,
    training_artifacts_match,
    trial_run_name,
    unresolved_confirmation_setting_opponents,
    valid_outcome_counts,
    wilson_interval,
)
from showdown_determinism import (
    LADDERS_MARKER,
    ROOM_BATTLE_MARKER,
    apply_deterministic_battle_seed_patch,
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
            self.assertEqual(args.additional_episode_batches, [])

    def test_default_search_reserves_a_local_refinement_budget(self) -> None:
        args = parse_search_args([
            "--run-prefix", "test",
            "--init-checkpoint", "parent.chk",
            "--episode-batch", "batch.jsonl",
        ])
        self.assertEqual(args.max_trials, 30)
        self.assertEqual(args.screen_candidates, 30)
        self.assertEqual(args.bayes_refine_settings, 4)
        self.assertEqual(args.bayes_refine_fraction, 0.5)
        self.assertEqual(args.confirmation_historical_opponents, 2)
        self.assertEqual(args.confirmation_protected_historical_opponents, 1)
        self.assertEqual(args.confirmation_max_opponents, 3)
        self.assertEqual(args.confirmation_games_per_side, 100)
        self.assertTrue(args.confirmation_adaptive)
        self.assertEqual(args.fresh_confirmation_episode_batch, "")
        self.assertEqual(args.additional_episode_batches, [])
        self.assertEqual(args.fresh_confirmation_shuffle_seeds, [404, 505, 606])
        self.assertEqual(args.eval_concurrent_games, 30)
        self.assertEqual(args.eval_worker_pairs, 40)

    def test_trainer_executable_is_fixed_and_rejected_in_config(self) -> None:
        self.assertEqual(DEFAULT_TRAINER_EXE.as_posix(), "build-fresh/showdown_client.exe")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "search.toml"
            config.write_text('trainer_exe = "custom.exe"\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fixed internally"):
                load_config_args(config)

    def test_policy_tag_uses_exact_batch_spelling_after_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            checkpoint = root / "models" / "parent.chk"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            batch = root / "batch.jsonl"
            relative_tag = ".\\models\\parent.chk"
            batch.write_text(json.dumps({
                "type": "episode_complete",
                "policy_tag": relative_tag,
                "observations": [0],
            }) + "\n", encoding="utf-8")
            self.assertEqual(inferred_policy_tag(root, batch, checkpoint), relative_tag)
            other = root / "models" / "other.chk"
            other.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(RuntimeError, "does not identify"):
                inferred_policy_tag(root, batch, other)

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

    def test_total_eta_starts_conservative_then_uses_measured_rates(self) -> None:
        state = SearchDisplayState(
            "search", 2, 1, 1, 10, 20,
            screen_games_budget=40,
            final_games_budget=80,
            conditional_training_trials=1,
            initial_training_seconds_per_trial=100.0,
            initial_evaluation_seconds_per_game=2.0,
        )
        self.assertEqual(state.estimated_remaining_seconds(), 540.0)

        state.training_durations = [50.0]
        state.evaluation_seconds = 20.0
        state.evaluation_games = 20
        self.assertEqual(state.estimated_remaining_seconds(), 270.0)

        state.trained_count = 2
        state.conditional_training_trials = 0
        state.screen_games_planned = 20
        state.final_games_planned = 40
        state.screen_estimate_complete = True
        state.final_estimate_complete = True
        self.assertEqual(state.estimated_remaining_seconds(), 60.0)

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

    def test_replicated_adaptive_screening_refines_whole_unresolved_settings(self) -> None:
        def group(lr: float, low: float, high: float) -> dict[str, object]:
            return {
                "hyperparameters": {
                    "learning_rate": lr,
                    "entropy_coef": 1e-4,
                    "anchor_kl_coef": 0.01,
                    "ppo_clip_epsilon": 0.2,
                    "episode_limit": 256,
                },
                "complete_seed_group": True,
                "collapse_flags": [],
                "rejected_trials": 0,
                "confidence_low": low,
                "confidence_high": high,
            }

        first = group(2.5e-6, 0.47, 0.65)
        cutoff = group(5e-6, 0.43, 0.61)
        challenger = group(1e-5, 0.39, 0.52)
        keys = [
            (2.5e-6, 1e-4, 0.01, 0.2, 256),
            (5e-6, 1e-4, 0.01, 0.2, 256),
            (1e-5, 1e-4, 0.01, 0.2, 256),
        ]
        requested = {key: 100 for key in keys}
        self.assertEqual(
            adaptive_screen_group_contenders(
                [first, cutoff, challenger], 2, requested, 200,
            ),
            keys,
        )
        challenger["confidence_high"] = 0.42
        self.assertEqual(
            adaptive_screen_group_contenders(
                [first, cutoff, challenger], 2, requested, 200,
            ),
            [],
        )

    def test_paired_evaluation_seed_is_shared_within_a_block(self) -> None:
        suites = resolve_evaluation_seed_suites("run-a", 2026, 0, 0)
        repeated = resolve_evaluation_seed_suites("run-a", 2026, 0, 0)
        rotated = resolve_evaluation_seed_suites("run-b", 2026, 0, 0)
        self.assertEqual(suites, repeated)
        self.assertNotEqual(suites["screen"], suites["confirmation"])
        self.assertNotEqual(suites["screen"], rotated["screen"])
        screen_seed = paired_evaluation_seed(int(suites["screen"]), 2)
        self.assertEqual(
            screen_seed,
            paired_evaluation_seed(int(suites["screen"]), 2),
        )
        self.assertNotEqual(
            screen_seed,
            paired_evaluation_seed(int(suites["screen"]), 3),
        )
        self.assertNotEqual(
            screen_seed,
            paired_evaluation_seed(int(suites["screen"]), 2, 1),
        )
        self.assertNotEqual(
            screen_seed,
            paired_evaluation_seed(int(suites["screen"]), 2, 0, 1),
        )
        self.assertNotEqual(
            screen_seed,
            paired_evaluation_seed(int(suites["confirmation"]), 2),
        )

    def test_explicit_confirmation_suite_must_be_held_out(self) -> None:
        suites = resolve_evaluation_seed_suites("run-a", 2026, 11, 22, 33)
        self.assertEqual(suites["screen"], 11)
        self.assertEqual(suites["confirmation"], 22)
        self.assertEqual(suites["fresh_confirmation"], 33)
        self.assertEqual(suites["screen_source"], "explicit")
        with self.assertRaisesRegex(ValueError, "must differ"):
            resolve_evaluation_seed_suites("run-a", 2026, 11, 22, 22)

    def test_fresh_confirmation_uses_a_third_derived_suite(self) -> None:
        suites = resolve_evaluation_seed_suites("run-a", 2026, 0, 0, 0)
        self.assertEqual(len({
            suites["screen"],
            suites["confirmation"],
            suites["fresh_confirmation"],
        }), 3)
        self.assertEqual(suites["fresh_confirmation_source"], "derived_from_run")

    def test_fresh_confirmation_comparison_reports_reproduction_strength(self) -> None:
        preliminary = {"valid_win_rate": 0.55}
        passing = {
            "valid_win_rate": 0.57,
            "clears_point_gate": True,
            "collapse_free": True,
            "complete_seed_group": True,
            "all_opponents_complete": True,
            "all_opponents_non_regression": True,
            "promotion_confident": True,
        }
        self.assertEqual(
            fresh_confirmation_comparison(preliminary, passing)["outcome"],
            "reproduced",
        )
        preliminary["primary_valid_win_rate"] = 0.51
        passing["primary_valid_win_rate"] = 0.52
        comparison = fresh_confirmation_comparison(preliminary, passing)
        self.assertEqual(
            comparison["comparison_metric"],
            "worst_training_batch_parent_valid_win_rate",
        )
        self.assertAlmostEqual(float(comparison["valid_win_rate_delta"]), 0.01)
        passing["valid_win_rate"] = 0.52
        passing.pop("primary_valid_win_rate")
        preliminary.pop("primary_valid_win_rate")
        self.assertEqual(
            fresh_confirmation_comparison(preliminary, passing)["outcome"],
            "weakened",
        )
        passing["all_opponents_non_regression"] = False
        self.assertEqual(
            fresh_confirmation_comparison(preliminary, passing)["outcome"],
            "reversed",
        )

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
        self.assertEqual(command[command.index("--pool-seed") + 1], "7")
        self.assertEqual(command[command.index("--battle-seed-base") + 1], "7")
        self.assertEqual(command[command.index("--serve-client") + 1], "0")

    def test_eval_command_does_not_create_more_worker_pairs_than_games(self) -> None:
        command = build_eval_command(
            eval_args(), Path("repo"), "small-eval", Path("candidate.chk"),
            Path("parent.chk"), "a", 40, 7,
        )
        self.assertEqual(command[command.index("--worker-pairs") + 1], "40")

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
                "battle_seed_base": 7,
                "model_specs": {
                    "a": {"kind": "checkpoint", "path": str(candidate)},
                    "b": {"kind": "checkpoint", "path": str(parent)},
                },
            }), encoding="utf-8")
            self.assertTrue(evaluation_artifacts_match(summary, candidate, parent, 250))
            self.assertTrue(evaluation_artifacts_match(summary, candidate, parent, 250, 7))
            self.assertFalse(evaluation_artifacts_match(summary, candidate, parent, 250, 8))
            self.assertFalse(evaluation_artifacts_match(summary, parent, candidate, 250))

    def test_confirmation_opponents_deduplicate_parent_champion_and_cap_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "parent.chk"
            history_a = root / "history_a.chk"
            history_b = root / "history_b.chk"
            explicit = root / "explicit.chk"
            for checkpoint in (parent, history_a, history_b, explicit):
                checkpoint.write_bytes(b"checkpoint")
            registry = root / "registry.json"
            registry.write_text(json.dumps({
                "champion_id": "champion",
                "members": [
                    {"id": "champion", "path": str(parent), "role": "champion", "status": "active", "generation": 3, "opponent_stats": {"history_a": {"recent_win_rate": 0.7}, "history_b": {"recent_win_rate": 0.4}}},
                    {"id": "history_a", "path": str(history_a), "role": "historical_snapshot", "status": "active", "generation": 2},
                    {"id": "history_b", "path": str(history_b), "role": "historical_snapshot", "status": "active", "generation": 1},
                ],
            }), encoding="utf-8")
            opponents, discovery = discover_confirmation_opponents(
                root, parent, str(registry), [str(explicit)], 2, 4,
            )
            self.assertEqual(
                [item.role for item in opponents],
                ["parent", "explicit", "historical", "historical"],
            )
            self.assertEqual(opponents[2].id, "history_b")
            self.assertTrue(opponents[2].protected)
            self.assertEqual(opponents[3].id, "history_a")
            self.assertFalse(opponents[3].protected)
            self.assertTrue(discovery["registry_loaded"])
            self.assertEqual(discovery["protected_historical_limit"], 1)
            self.assertEqual(discovery["duplicates"], [{"id": "champion", "same_as": "parent"}])

    def test_showdown_seed_patch_is_opt_in_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = root / "server"
            server.mkdir()
            (server / "ladders.ts").write_text("""const PERIODIC_MATCH_INTERVAL = 60 * SECONDS;
\t\tconst delayedStart = format.playerCount > players.length ? 'multi' : false;
\t\treturn Rooms.createBattle({
\t\t\tformat: formatid,
\t\t\tplayers,
\t\t\trated: minRating,
\t\t\tchallengeType: readies[0].challengeType,
\t\t\tdelayedStart,
\t\t});
""", encoding="utf-8")
            (server / "room-battle.ts").write_text("""\tseed?: PRNGSeed;
\troomid?: RoomID;
\t\t\tconst options = {
\t\t\t\tname: player.name,
\t\t\t\tavatar: user.avatar,
\t\t\t\tteam: playerOpts?.team,
\t\t\t};
""", encoding="utf-8")
            self.assertTrue(apply_deterministic_battle_seed_patch(root))
            self.assertFalse(apply_deterministic_battle_seed_patch(root))
            self.assertIn(LADDERS_MARKER, (server / "ladders.ts").read_text(encoding="utf-8"))
            room_source = (server / "room-battle.ts").read_text(encoding="utf-8")
            self.assertIn(ROOM_BATTLE_MARKER, room_source)
            self.assertIn("seed: this.options.playerSeeds?.[player.num - 1]", room_source)

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

    def test_replicated_search_selection_keeps_complete_seed_groups(self) -> None:
        combinations = select_search_combinations(
            [2.5e-6, 5e-6, 1e-5], [1e-4], [0.01, 0.03], [0.1, 0.2],
            [101, 202, 303], [256], 7, 2026, True,
        )
        self.assertEqual(len(combinations), 6)
        grouped: dict[tuple[float, float, float, float, int], set[int]] = {}
        for params in combinations:
            key = (
                params.learning_rate, params.entropy_coef, params.anchor_kl_coef,
                params.ppo_clip_epsilon, params.episode_limit,
            )
            grouped.setdefault(key, set()).add(params.shuffle_seed)
        self.assertEqual(len(grouped), 2)
        self.assertTrue(all(seeds == {101, 202, 303} for seeds in grouped.values()))

    def test_hyperparameter_group_ranking_pools_all_seed_results(self) -> None:
        trials = []
        for seed, wins in ((101, 54), (202, 53), (303, 52)):
            trials.append(SimpleNamespace(
                run_name=f"seed-{seed}",
                hyperparameters=Hyperparameters(5e-6, 1e-4, 0.03, 0.1, seed, 256),
                safety_flags=[],
                screen_evaluation=self.evaluation("screen", wins, 100, 0.4, 0.6),
                final_evaluation=None,
            ))
        summaries = build_hyperparameter_group_summary(
            trials, stage="screen", expected_seeds=[101, 202, 303],
        )
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["complete_seed_group"])
        self.assertEqual(summaries[0]["valid_games"], 300)
        self.assertAlmostEqual(float(summaries[0]["valid_win_rate"]), 0.53)
        winner, assessment = grouped_promotion_assessment(summaries, 0.50, 0.50)
        self.assertIsNotNone(winner)
        self.assertEqual(assessment["selection_scope"], "pooled_hyperparameter_setting")

    def test_multi_batch_group_requires_every_batch_seed_replicate(self) -> None:
        trials = []
        for batch_id in ("batch01", "batch02"):
            for seed in (101, 202):
                trials.append(SimpleNamespace(
                    run_name=f"{batch_id}-seed-{seed}",
                    training_batch_id=batch_id,
                    training_batch_path=f"{batch_id}.jsonl",
                    hyperparameters=Hyperparameters(5e-6, 1e-4, 0.03, 0.1, seed, 256),
                    safety_flags=[],
                    screen_evaluation=self.evaluation("screen", 52, 100, 0.4, 0.6),
                    final_evaluation=None,
                ))
        complete = build_hyperparameter_group_summary(
            trials,
            stage="screen",
            expected_seeds=[101, 202],
            expected_batch_ids=["batch01", "batch02"],
        )[0]
        self.assertTrue(complete["complete_replicate_group"])
        self.assertEqual(len(complete["evaluated_replicates"]), 4)

        incomplete = build_hyperparameter_group_summary(
            trials[:-1],
            stage="screen",
            expected_seeds=[101, 202],
            expected_batch_ids=["batch01", "batch02"],
        )[0]
        self.assertFalse(incomplete["complete_replicate_group"])
        self.assertFalse(incomplete["complete_seed_group"])

    def test_search_batch_resolution_assigns_stable_ids_and_rejects_duplicates(self) -> None:
        root = Path("C:/repo")
        batches = resolve_search_episode_batches(
            root, "matches/a.jsonl", ["matches/b.jsonl"],
        )
        self.assertEqual([batch_id for batch_id, _ in batches], ["batch01", "batch02"])
        self.assertNotEqual(batches[0][1], batches[1][1])
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            resolve_search_episode_batches(
                root, "matches/a.jsonl", ["matches/a.jsonl"],
            )

    def test_multi_batch_run_names_are_unique_without_changing_legacy_names(self) -> None:
        params = Hyperparameters(5e-6, 1e-4, 0.03, 0.1, 101, 256)
        legacy = trial_run_name("search", 1, params)
        first = trial_run_name("search", 1, params, "batch01")
        second = trial_run_name("search", 1, params, "batch02")
        self.assertNotIn("batch01", legacy)
        self.assertIn("batch01", first)
        self.assertIn("batch02", second)
        self.assertNotEqual(first, second)

    def test_multi_opponent_confirmation_detects_specific_regression(self) -> None:
        opponents = [
            ConfirmationOpponent("parent", "parent.chk", "test", "parent"),
            ConfirmationOpponent("history", "history.chk", "test", "historical"),
        ]
        trials = []
        for seed in (101, 202, 303):
            parent_result = self.evaluation("final", 40, 100, 0.31, 0.49)
            history_result = self.evaluation("final", 65, 100, 0.55, 0.74)
            trials.append(SimpleNamespace(
                run_name=f"seed-{seed}",
                hyperparameters=Hyperparameters(5e-6, 1e-4, 0.03, 0.1, seed, 256),
                safety_flags=[],
                final_evaluation=merge_evaluation_results(parent_result, history_result),
                screen_evaluation=None,
                confirmation_evaluations={
                    "parent": parent_result,
                    "history": history_result,
                },
            ))
        summaries = build_robust_confirmation_group_summary(
            trials,
            expected_seeds=[101, 202, 303],
            opponents=opponents,
            non_regression_threshold=0.5,
            dominance_spread=0.10,
        )
        self.assertEqual(len(summaries), 1)
        self.assertFalse(summaries[0]["all_opponents_non_regression"])
        self.assertEqual(summaries[0]["regression_opponents"], ["parent"])
        self.assertTrue(summaries[0]["favorable_matchup_dominance"])
        winner, assessment = robust_grouped_promotion_assessment(summaries, 0.5, 0.5)
        self.assertIsNone(winner)
        self.assertEqual(assessment["status"], "opponent_regression")

    def test_history_win_cannot_offset_a_parent_loss(self) -> None:
        opponents = [
            ConfirmationOpponent("parent", "parent.chk", "test", "parent"),
            ConfirmationOpponent("history", "history.chk", "test", "historical", False),
        ]
        trials = []
        for seed in (101, 202, 303):
            parent_result = self.evaluation("final", 49, 100, 0.39, 0.59)
            history_result = self.evaluation("final", 65, 100, 0.55, 0.74)
            trials.append(SimpleNamespace(
                run_name=f"seed-{seed}",
                hyperparameters=Hyperparameters(5e-6, 3e-4, 0.03, 0.2, seed, 256),
                safety_flags=[],
                final_evaluation=merge_evaluation_results(parent_result, history_result),
                screen_evaluation=None,
                confirmation_evaluations={
                    "parent": parent_result,
                    "history": history_result,
                },
            ))

        summaries = build_robust_confirmation_group_summary(
            trials,
            expected_seeds=[101, 202, 303],
            opponents=opponents,
            non_regression_threshold=0.5,
            dominance_spread=0.10,
        )
        summary = summaries[0]
        self.assertGreater(float(summary["valid_win_rate"]), 0.5)
        self.assertAlmostEqual(float(summary["primary_valid_win_rate"]), 0.49)

        winner, assessment = robust_grouped_promotion_assessment(summaries, 0.5, 0.5)
        self.assertIsNone(winner)
        self.assertEqual(assessment["status"], "no_winner")
        self.assertEqual(
            assessment["selection_metric"],
            "worst_training_batch_parent_valid_win_rate",
        )

    def test_strong_batch_cannot_hide_weak_batch_at_promotion(self) -> None:
        opponent = ConfirmationOpponent("parent", "parent.chk", "test", "parent")
        trials = []
        for batch_id, wins in (("batch01", 60), ("batch02", 49)):
            result = self.evaluation("final", wins, 100, 0.35, 0.65)
            trials.append(SimpleNamespace(
                run_name=batch_id,
                training_batch_id=batch_id,
                training_batch_path=f"{batch_id}.jsonl",
                hyperparameters=Hyperparameters(5e-6, 1e-4, 0.03, 0.1, 101, 256),
                safety_flags=[],
                final_evaluation=result,
                screen_evaluation=None,
                confirmation_evaluations={"parent": result},
            ))
        summaries = build_robust_confirmation_group_summary(
            trials,
            expected_seeds=[101],
            expected_batch_ids=["batch01", "batch02"],
            opponents=[opponent],
            non_regression_threshold=0.5,
            dominance_spread=0.1,
        )
        winner, assessment = robust_grouped_promotion_assessment(
            summaries, 0.5, 0.5,
        )
        self.assertGreater(assessment["primary_valid_win_rate"], 0.5)
        self.assertEqual(assessment["primary_worst_batch_valid_win_rate"], 0.49)
        self.assertIsNone(winner)
        self.assertFalse(assessment["clears_point_gate"])

    def test_adaptive_confirmation_expands_only_unresolved_matchups(self) -> None:
        opponents = [
            ConfirmationOpponent("parent", "parent.chk", "test", "parent"),
            ConfirmationOpponent("history", "history.chk", "test", "historical"),
        ]
        trials = []
        for seed in (101, 202, 303):
            trials.append(SimpleNamespace(
                run_name=f"seed-{seed}",
                hyperparameters=Hyperparameters(5e-6, 1e-4, 0.03, 0.1, seed, 256),
                safety_flags=[],
                final_evaluation=None,
                screen_evaluation=None,
                confirmation_evaluations={
                    "parent": self.evaluation("final", 50, 100, 0.40, 0.60),
                    "history": self.evaluation("final", 65, 100, 0.55, 0.74),
                },
            ))
        setting = (5e-6, 1e-4, 0.03, 0.1, 256)
        unresolved = unresolved_confirmation_setting_opponents(
            trials,
            expected_seeds=[101, 202, 303],
            opponents=opponents,
            requested_games_per_side={(setting, "parent"): 100, (setting, "history"): 100},
            threshold=0.5,
            max_games_per_side=300,
        )
        self.assertEqual(unresolved, {(setting, "parent")})

    def test_bayesian_suggestion_is_untried_and_records_acquisition(self) -> None:
        space = [
            (2.5e-6, 1e-4, 0.01, 0.1, 256),
            (5e-6, 1e-4, 0.03, 0.1, 256),
            (1e-5, 3e-4, 0.05, 0.2, 256),
            (5e-6, 3e-4, 0.01, 0.2, 256),
        ]
        initial = select_space_filling_settings(space, 2, 2026)
        observed = []
        for setting, rate in zip(initial, (0.49, 0.54)):
            observed.append({
                "hyperparameters": {
                    "learning_rate": setting[0],
                    "entropy_coef": setting[1],
                    "anchor_kl_coef": setting[2],
                    "ppo_clip_epsilon": setting[3],
                    "episode_limit": setting[4],
                },
                "complete_seed_group": True,
                "collapse_flags": [],
                "rejected_trials": 0,
                "valid_games": 600,
                "valid_win_rate": rate,
            })
        remaining = [setting for setting in space if setting not in initial]
        suggestion, acquisition = suggest_bayesian_setting(observed, remaining, space)
        self.assertIn(suggestion, remaining)
        self.assertGreaterEqual(acquisition["expected_improvement"], 0.0)
        self.assertGreater(acquisition["predicted_stddev"], 0.0)

    def test_local_refinement_interpolates_and_explores_past_a_boundary(self) -> None:
        center = (2.5e-6, 1e-4, 0.03, 0.1, 256)
        candidates = generate_local_refinement_space(
            center,
            [2.5e-6, 5e-6, 1e-5],
            [1e-4, 3e-4],
            [0.01, 0.03, 0.05],
            [0.1, 0.2],
            fraction=0.5,
        )
        self.assertEqual(len(candidates), 80)
        self.assertNotIn(center, candidates)
        self.assertTrue(any(setting[0] < 2.5e-6 for setting in candidates))
        self.assertTrue(any(2.5e-6 < setting[0] < 5e-6 for setting in candidates))
        self.assertTrue(any(setting[1] < 1e-4 for setting in candidates))
        self.assertTrue(any(setting[3] < 0.1 for setting in candidates))

        diagnostics = configured_boundary_diagnostics(
            center,
            [2.5e-6, 5e-6, 1e-5],
            [1e-4, 3e-4],
            [0.01, 0.03, 0.05],
            [0.1, 0.2],
        )
        self.assertEqual(diagnostics["learning_rate"], "at_configured_min")
        self.assertEqual(diagnostics["anchor_kl_coef"], "interior")


if __name__ == "__main__":
    unittest.main()
