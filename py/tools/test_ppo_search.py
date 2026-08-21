from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ppo_search import (
    Hyperparameters,
    build_eval_command,
    build_train_command,
    evaluation_artifacts_match,
    training_safety_flags,
    training_artifacts_match,
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
        self.assertEqual(command[command.index("--target-kl-min-labels") + 1], "500")
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
                "minibatch_episodes": args.ppo_minibatch_episodes,
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


if __name__ == "__main__":
    unittest.main()
