"""Write evaluation results with enough provenance to reproduce them."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_result_record(
    run_name: str,
    retriever: str,
    split: str,
    config: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    commit: str | None = None,
) -> dict[str, Any]:
    """Attach configuration, source revision, and runtime details to metrics."""
    return {
        "run_name": run_name,
        "retriever": retriever,
        "split": split,
        "commit": commit or current_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "metrics": evaluation["metrics"],
        "timing": evaluation["timing"],
        "counts": evaluation["counts"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def write_result(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
