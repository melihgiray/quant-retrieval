"""Write evaluation results with enough provenance to reproduce them."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_record(contents: bytes, path: Path) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        record = json.loads(contents, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise SystemExit(f"{path} is not a valid result record: {error}") from error
    if not isinstance(record, dict):
        raise SystemExit(f"{path} result record must be an object")
    return record


def build_result_record(
    run_name: str,
    retriever: str,
    split: str,
    config: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    commit: str | None = None,
    include_rankings: bool = False,
) -> dict[str, Any]:
    """Attach configuration, source revision, and runtime details to metrics."""
    record = {
        "run_name": run_name,
        "retriever": retriever,
        "split": split,
        "commit": commit or current_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "config": deepcopy(config),
        "metrics": deepcopy(evaluation["metrics"]),
        "timing": deepcopy(evaluation["timing"]),
        "counts": deepcopy(evaluation["counts"]),
        "dataset_sha256": deepcopy(evaluation.get("dataset_sha256")),
        "diagnostics": {
            str(question_id): deepcopy(details)
            for question_id, details in sorted(evaluation.get("diagnostics", {}).items())
        },
        # Kept so two runs can be compared question by question. Averages alone
        # cannot say whether a difference is larger than the spread.
        "per_query": {
            str(query_id): {name: round(value, 6) for name, value in scores.items()}
            for query_id, scores in sorted(evaluation.get("per_query", {}).items())
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    if include_rankings:
        record["rankings"] = {
            str(question_id): [int(answer_id) for answer_id in ranking]
            for question_id, ranking in sorted(evaluation["rankings"].items())
        }
    return record


def write_result(record: dict[str, Any], path: Path, *, overwrite: bool = True) -> None:
    serialized = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        if overwrite:
            temporary.replace(path)
        else:
            # Linking publishes the complete file only if the destination is still absent.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
