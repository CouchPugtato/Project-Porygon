from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_retries_a_temporary_windows_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text('{"generation": 1}\n', encoding="utf-8")
            real_replace = os.replace
            attempts = 0

            def flaky_replace(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(13, "file is temporarily in use", str(destination))
                real_replace(source, destination)

            with patch("artifact_io.os.replace", side_effect=flaky_replace), \
                    patch("artifact_io.time.sleep") as sleep:
                write_json_atomically(path, {"generation": 2})

            self.assertEqual(3, attempts)
            self.assertEqual(2, sleep.call_count)
            self.assertEqual({"generation": 2}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([], list(root.glob(".manifest.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
