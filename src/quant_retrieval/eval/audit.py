"""Audit saved evaluation evidence without loading data or model runtimes."""

import math

from quant_retrieval.eval.metrics import METRIC_NAMES

SCORE_ROUNDING_TOLERANCE = 1e-6

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


def audit_metrics(record: dict) -> dict:
    scores = validated_scores(record)
    saved = record.get("metrics")
    if not isinstance(saved, dict) or set(saved) != set(METRIC_NAMES):
        raise ValueError("aggregate metrics must contain exactly the supported metrics")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in saved.values()):
        raise ValueError("aggregate metrics must be finite numbers between zero and one")
    recomputed = {name: math.fsum(row[name] for row in scores.values()) / len(scores)
                  for name in METRIC_NAMES}
    errors = {name: abs(saved[name] - recomputed[name]) for name in METRIC_NAMES}
    if any(error > SCORE_ROUNDING_TOLERANCE for error in errors.values()):
        raise ValueError("aggregate metrics disagree with saved per-query scores")
    return {"recomputed": recomputed, "absolute_error": errors,
            "serialization_tolerance": SCORE_ROUNDING_TOLERANCE}


def audit_protocol(record: dict) -> dict:
    scores = validated_scores(record)
    if record.get("split") not in ("train", "val"):
        raise ValueError("offline development audits accept only train or val runs")
    for name in ("run_name", "retriever"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"evaluation {name} must be a nonempty string")
    counts = record.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("evaluation counts must be an object")
    for name in ("queries", "corpus_documents", "max_results"):
        if type(counts.get(name)) is not int or counts[name] < 1:
            raise ValueError(f"evaluation {name} must be a positive integer")
    if counts["queries"] != len(scores):
        raise ValueError("evaluation query count differs from score coverage")
    if counts["max_results"] < 100:
        raise ValueError("Recall@100 requires a retrieval depth of at least 100")
    return {"split": record["split"], "run_name": record["run_name"],
            "retriever": record["retriever"], "counts": dict(counts)}
