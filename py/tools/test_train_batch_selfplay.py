from __future__ import annotations

import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from train_batch_selfplay import (
    aggregate_validation_metrics,
    batch_label_counts,
    collect_training_stats,
    completed_records_for_epoch,
    completed_shard_map,
    copy_file_atomically,
    dataset_epoch_snapshot_path,
    dataset_epoch_validation_path,
    evaluate_fixed_holdout,
    epoch_work_checkpoint_path,
    is_validation_battle,
    load_default_env,
    main,
    parse_args,
    replay_path_manifest_path,
    trainer_command_for_file,
    trainer_command_for_manifest,
    validate_resume_manifest,
    validation_hash,
    write_replay_path_manifest,
    verify_prepared_dataset,
)


class TrainingStatsTests(unittest.TestCase):
    def test_selected_config_controls_arguments_and_environment_without_default_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "training.toml"
            config.write_text('run = "dataset"\ncheckpoint = "models/student.chk"\n'
                              'sample_files = 0\nepochs = 3\nsupervised_optimizer = "adam"\n'
                              'env_porygon_omp_threads = 1\nshuffle_seed = 27501\n', encoding="utf-8")
            args = parse_args(["--config", str(config), "--epochs", "2"])
            self.assertEqual(args.epochs, 2)
            self.assertEqual(args.sample_files, 0)
            self.assertEqual(args.supervised_optimizer, "adam")
            self.assertEqual(args.shuffle_seed, 27501)
            self.assertEqual(load_default_env(Path(args.config)), {"PORYGON_OMP_THREADS": "1"})

    def test_large_data_profile_uses_all_shards_and_one_thread(self) -> None:
        config = TOOLS_DIR.parents[1] / "config/experimental/tactical_student_train_1000.toml"
        args = parse_args(["--config", str(config)])
        self.assertEqual(args.sample_files, 0)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.validation_seed, 20261024)
        self.assertEqual(args.learning_rate, 0.0001)
        self.assertEqual(args.shuffle_seed, 27501)
        self.assertEqual(load_default_env(Path(args.config))["PORYGON_OMP_THREADS"], "1")

    def test_prepared_dataset_checks_hash_and_split_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "dataset"
            directory.mkdir()
            replay = directory / "shard_000_raw.jsonl"
            replay.write_bytes(b"replay fixture\n")
            manifest = {"dataset_kind": "supervised_tactical_demonstrations", "status": "completed",
                        "validation_seed": 1337, "shards": [{"file": replay.name,
                        "sha256": hashlib.sha256(replay.read_bytes()).hexdigest()}], "holdout_shards": []}
            (directory / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            verify_prepared_dataset(directory, [replay], 1337)
            with self.assertRaises(SystemExit):
                verify_prepared_dataset(directory, [replay], 1338)
            replay.write_bytes(b"changed")
            with self.assertRaises(SystemExit):
                verify_prepared_dataset(directory, [replay], 1337)

    def test_evaluation_only_does_not_initialize_or_publish_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "matches/runs/dataset"
            run.mkdir(parents=True)
            (run / "holdout_000_raw.jsonl").write_text("fixture", encoding="utf-8")
            executable = root / "client.exe"
            executable.touch()
            checkpoint = root / "parent.chk"
            checkpoint.write_bytes(b"unchanged model")
            output = root / "baseline.json"
            config = root / "config.toml"
            config.write_text(f'run = "dataset"\ncheckpoint = "{checkpoint.as_posix()}"\n'
                              f'trainer_exe = "{executable.as_posix()}"\npattern = "holdout_*_raw.jsonl"\n', encoding="utf-8")
            argv = ["trainer", "--config", str(config), "--evaluate-only", "1", "--evaluation-summary", str(output)]
            metrics = {"counts": {"turns": 1, "illegal_predictions": 0, "nonfinite_values": 0}, "metrics": {}}
            with patch("sys.argv", argv), patch("train_batch_selfplay.Path.cwd", return_value=root), \
                    patch("train_batch_selfplay.ensure_initialized_checkpoint") as initialize, \
                    patch("train_batch_selfplay.evaluate_fixed_holdout", return_value=metrics), \
                    patch("train_batch_selfplay.select_reporter", return_value=Mock()):
                main()
            initialize.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"unchanged model")
            self.assertEqual(json.loads(output.read_text())["counts"]["turns"], 1)
            self.assertFalse((root / "models").exists())

    def test_common_holdout_rejects_dropped_labels_or_changed_turn_counts(self) -> None:
        args = SimpleNamespace(trainer_exe="unused", validation_seed=1337, run="dataset")
        metrics = "[eval] validation metrics_version=2 turns=2 full_turn_nll=1 illegal_predictions=0 nonfinite_values=0"
        cases = [(["[train] ingest complete invalid=1", metrics], None),
                 (["[train] accepted_labels direct=1 reconstructed=0 failed=1", metrics], None),
                 ([metrics], {"dataset_kind": "supervised_tactical_demonstrations",
                              "counts": {"holdout_decision_turns": 3}})]
        for lines, dataset in cases:
            with self.subTest(lines=lines), patch("train_batch_selfplay.run_trainer_and_capture", return_value=(0, lines)), \
                    patch("train_batch_selfplay.load_json_if_exists", return_value=dataset):
                with self.assertRaises(RuntimeError):
                    evaluate_fixed_holdout(args, [Path("holdout.jsonl")], Path("model.chk"), Path.cwd(), {}, Mock(), SimpleNamespace())

    def test_collects_versioned_policy_metrics(self) -> None:
        stats = collect_training_stats(
            [
                "[train] epoch=1 validation action_loss=2.100000 value_loss=0.500000 accuracy=0.250000 labels=35 sessions=2",
                "[train] epoch=1 validation top3_accuracy=0.500000 action1_accuracy=0.400000 action2_accuracy=0.300000 labels=35 skipped=1",
                "[train] epoch=1 validation metrics_version=2 action_nll=1.500000 target_nll=0.600000 full_turn_nll=2.100000 full_turn_accuracy=0.250000 joint_pair_accuracy=0.200000 kind_accuracy=0.700000 move_accuracy=0.650000 switch_accuracy=0.500000 tera_accuracy=0.900000 target_accuracy=0.600000 turns=20 joint_pairs=15 target_labels=10 illegal_predictions=0 nonfinite_values=0",
            ]
        )

        metrics = stats["validation_policy_metrics"]
        self.assertEqual(metrics["metrics_version"], 2)
        self.assertEqual(metrics["turns"], 20)
        self.assertEqual(metrics["joint_pairs"], 15)
        self.assertEqual(metrics["illegal_predictions"], 0)
        self.assertAlmostEqual(metrics["full_turn_nll"], 2.1)
        self.assertAlmostEqual(metrics["target_accuracy"], 0.6)

    def test_supervised_command_forwards_optimizer_and_learning_rate(self) -> None:
        args = SimpleNamespace(
            trainer_exe="build-fresh/showdown_client.exe",
            mode="supervised",
            resolved_checkpoint=Path("models/test.chk"),
            epochs_per_file=1,
            supervised_profile=False,
            supervised_optimizer="adam",
            validation_seed=20260902,
            learning_rate=0.001,
        )

        command = trainer_command_for_file(args, Path("matches/worker_1_raw.jsonl"))

        self.assertIn("--supervised-optimizer", command)
        self.assertEqual(command[command.index("--supervised-optimizer") + 1], "adam")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.001")
        self.assertEqual(command[command.index("--validation-seed") + 1], "20260902")
        self.assertEqual(command[command.index("--aux-checkpoints") + 1], "0")

    def test_eval_output_uses_the_same_versioned_metric_parser(self) -> None:
        stats = collect_training_stats(
            [
                "[eval] validation metrics_version=2 action_nll=1.25 full_turn_accuracy=0.75 "
                "turns=12 target_labels=3 illegal_predictions=0 nonfinite_values=0"
            ]
        )

        self.assertEqual(stats["validation_policy_metrics"]["metrics_version"], 2)
        self.assertEqual(stats["validation_policy_metrics"]["turns"], 12)

    def test_supervised_manifest_command_keeps_shards_in_one_process(self) -> None:
        args = SimpleNamespace(
            trainer_exe="build-fresh/showdown_client.exe",
            epochs_per_file=1,
            supervised_profile=False,
            supervised_optimizer="adam",
            validation_seed=20260903,
            learning_rate=0.001,
        )

        command = trainer_command_for_manifest(
            args, Path("models/replays.txt"), Path("models/work.chk")
        )

        self.assertEqual(command[1], "--train-supervised-manifest")
        self.assertEqual(command[2:4], [str(Path("models/replays.txt")), str(Path("models/work.chk"))])
        self.assertEqual(command[command.index("--supervised-optimizer") + 1], "adam")

    def test_collects_per_file_batch_results(self) -> None:
        stats = collect_training_stats(
            [
                "[train-batch] file_complete index=1/2 train_sessions=12 labels=99 elapsed=1.2s",
                "[train-batch] file_complete index=2/2 train_sessions=8 labels=51 elapsed=0.8s",
                "trained mode=supervised step=150 labels=150 sessions=20",
            ]
        )

        self.assertEqual([entry["labels"] for entry in stats["batch_files"]], [99, 51])
        self.assertEqual([entry["index"] for entry in stats["batch_files"]], [1, 2])
        self.assertEqual([entry["file_count"] for entry in stats["batch_files"]], [2, 2])
        self.assertEqual(stats["final_train"]["labels"], 150)

    def test_recovers_label_counts_from_pre_fix_batch_stats(self) -> None:
        parsed_stats = {
            "batch_files": [
                {"index": "1/2", "labels": 99},
                {"index": "2/2", "labels": 51},
            ],
            "final_train": {"labels": 150},
        }

        self.assertEqual(batch_label_counts(parsed_stats, 2), {1: 99, 2: 51})


class ValidationProvenanceTests(unittest.TestCase):
    def test_atomic_checkpoint_copy_replaces_destination_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.chk"
            destination = root / "snapshots" / "epoch.chk"
            source.write_bytes(b"new checkpoint")
            destination.parent.mkdir()
            destination.write_bytes(b"old checkpoint")

            copy_file_atomically(source, destination)

            self.assertEqual(destination.read_bytes(), b"new checkpoint")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_validation_split_is_stable_and_seeded(self) -> None:
        self.assertEqual(validation_hash("battle-4", 1337), 5406646322143240280)
        self.assertTrue(is_validation_battle("battle-4", 1337))
        self.assertFalse(is_validation_battle("battle-alpha", 1337))
        self.assertNotEqual(validation_hash("battle-4", 1337), validation_hash("battle-4", 1338))

    def test_validation_metrics_are_weighted_by_their_label_counts(self) -> None:
        summary = aggregate_validation_metrics(
            [
                {
                    "validation_policy_metrics": {
                        "turns": 10,
                        "target_labels": 2,
                        "action_nll": 2.0,
                        "full_turn_accuracy": 0.5,
                        "target_accuracy": 0.5,
                    }
                },
                {
                    "validation_policy_metrics": {
                        "turns": 30,
                        "target_labels": 8,
                        "action_nll": 1.0,
                        "full_turn_accuracy": 0.75,
                        "target_accuracy": 0.75,
                    }
                },
            ],
            source_shard_count=2,
            validation_seed=1337,
        )

        self.assertEqual(summary["counts"]["turns"], 40)
        self.assertAlmostEqual(summary["metrics"]["action_nll"], 1.25)
        self.assertAlmostEqual(summary["metrics"]["full_turn_accuracy"], 0.6875)
        self.assertAlmostEqual(summary["metrics"]["target_accuracy"], 0.7)

    def test_dataset_epoch_artifact_names_are_unambiguous(self) -> None:
        checkpoint = Path("models/run/model.chk")

        self.assertEqual(
            dataset_epoch_snapshot_path(checkpoint, 3),
            Path("models/run/model_dataset_epoch003.chk"),
        )
        self.assertEqual(
            dataset_epoch_validation_path(checkpoint, 3),
            Path("models/run/model_dataset_epoch003_validation.json"),
        )
        self.assertEqual(
            replay_path_manifest_path(checkpoint, 3),
            Path("models/run/model_epoch003_replays.txt"),
        )
        self.assertEqual(
            epoch_work_checkpoint_path(checkpoint, 3),
            Path("models/run/.model_epoch003_work.chk"),
        )

    def test_replay_path_manifest_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "worker 2.jsonl", root / "worker_1.jsonl"]
            manifest = root / "batch" / "replays.txt"

            write_replay_path_manifest(manifest, paths)

            self.assertEqual(
                manifest.read_text(encoding="utf-8").splitlines(),
                [str(path.resolve()) for path in paths],
            )

    def test_resume_uses_exact_completed_shard_identity(self) -> None:
        checkpoint = Path("models/run/model.chk")
        sources = [Path("matches/run/worker_1.jsonl"), Path("matches/run/worker_2.jsonl")]
        manifest = {
            "run": "run",
            "mode": "supervised",
            "output_checkpoint": str(checkpoint.resolve()),
            "source_shards": [str(path.resolve()) for path in sources],
            "validation_seed": 1337,
            "epochs": 2,
            "epochs_per_file": 1,
            "sample_files": 0,
            "shuffle": True,
            "epoch_plans": {"1": [str(path.resolve()) for path in sources]},
            "completed_shards": [
                {
                    "outer_epoch": 1,
                    "shard_index": 1,
                    "path": str(sources[0].resolve()),
                    "label_count": 42,
                }
            ],
        }

        validate_resume_manifest(
            manifest,
            run="run",
            mode="supervised",
            checkpoint=checkpoint,
            source_files=sources,
            validation_seed=1337,
            epochs=2,
            epochs_per_file=1,
            sample_files=0,
            shuffle=True,
        )
        completed = completed_shard_map(manifest)
        resumed_epoch = completed_records_for_epoch(completed, 1, sources)

        self.assertIn((1, 1, str(sources[0].resolve())), completed)
        self.assertEqual(completed[(1, 1, str(sources[0].resolve()))]["label_count"], 42)
        self.assertEqual(list(resumed_epoch), [(1, 1, str(sources[0].resolve()))])


if __name__ == "__main__":
    unittest.main()
