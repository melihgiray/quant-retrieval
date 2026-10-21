"""Audit saved evaluation evidence without loading data or model runtimes."""

import math

from quant_retrieval.eval.metrics import METRIC_NAMES


def validated_scores(record: dict) -> dict[int, dict[str, float]]:
    if not isinstance(record, dict):
        raise ValueError("evaluation record must be an object")
    scores = record.get("per_query")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("evaluation needs nonempty per-query scores")
    checked = {}
    for key, row in scores.items():
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                or not 0 < int(key) < 2**63 or str(int(key)) != key):
            raise ValueError("evaluation question IDs must be canonical positive int64 strings")
        if not isinstance(row, dict) or set(row) != set(METRIC_NAMES):
            raise ValueError("every query must contain exactly the supported metrics")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in row.values()):
            raise ValueError("evaluation scores must be finite numbers between zero and one")
        checked[int(key)] = {name: float(row[name]) for name in METRIC_NAMES}
    return dict(sorted(checked.items()))
