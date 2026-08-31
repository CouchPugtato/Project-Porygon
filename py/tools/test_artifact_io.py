from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from artifact_io import write_json_atomically


class AtomicJsonWriteTests(unittest.TestCase):
    def test_replaces_existing_json_without_leaving_a_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text('{"generation": 1}\n', encoding="utf-8")

            write_json_atomically(path, {"generation": 2, "status": "running"})

            self.assertEqual(
                {"generation": 2, "status": "running"},
                json.loads(path.read_text(encoding="utf-8")),
            )
            self.assertEqual([], list(root.glob(".manifest.json.*.tmp")))

    def test_failed_write_preserves_the_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            previous = '{"status": "completed"}\n'
            path.write_text(previous, encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomically(path, {"not_json": object()})

            self.assertEqual(previous, path.read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".manifest.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
