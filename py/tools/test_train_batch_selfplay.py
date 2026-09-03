from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from train_batch_selfplay import collect_training_stats, trainer_command_for_file


class TrainingStatsTests(unittest.TestCase):
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
            learning_rate=0.001,
        )

        command = trainer_command_for_file(args, Path("matches/worker_1_raw.jsonl"))

        self.assertIn("--supervised-optimizer", command)
        self.assertEqual(command[command.index("--supervised-optimizer") + 1], "adam")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.001")


if __name__ == "__main__":
    unittest.main()
