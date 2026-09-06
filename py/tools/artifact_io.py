from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


REPLACE_RETRY_DELAYS_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4)


def _replace_with_retry(source: Path, destination: Path) -> None:
    for delay in (*REPLACE_RETRY_DELAYS_SECONDS, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
