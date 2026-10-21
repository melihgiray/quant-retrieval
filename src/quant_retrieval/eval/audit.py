"""Audit saved evaluation evidence without loading data or model runtimes."""

import math
import re
from collections import Counter
from datetime import datetime

from quant_retrieval.eval.analysis import validate_diagnostic
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
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdecimal()
            or not 0 < int(key) < 2**63
            or str(int(key)) != key
        ):
            raise ValueError("evaluation question IDs must be canonical positive int64 strings")
        if not isinstance(row, dict) or set(row) != set(METRIC_NAMES):
            raise ValueError("every query must contain exactly the supported metrics")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in row.values()
        ):
            raise ValueError("evaluation scores must be finite numbers between zero and one")
        checked[int(key)] = {name: float(row[name]) for name in METRIC_NAMES}
    return dict(sorted(checked.items()))


def audit_metrics(record: dict) -> dict:
    scores = validated_scores(record)
    saved = record.get("metrics")
    if not isinstance(saved, dict) or set(saved) != set(METRIC_NAMES):
        raise ValueError("aggregate metrics must contain exactly the supported metrics")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in saved.values()
    ):
        raise ValueError("aggregate metrics must be finite numbers between zero and one")
    recomputed = {
        name: math.fsum(row[name] for row in scores.values()) / len(scores) for name in METRIC_NAMES
    }
    errors = {name: abs(saved[name] - recomputed[name]) for name in METRIC_NAMES}
    if any(error > SCORE_ROUNDING_TOLERANCE for error in errors.values()):
        raise ValueError("aggregate metrics disagree with saved per-query scores")
    return {
        "recomputed": recomputed,
        "absolute_error": errors,
        "serialization_tolerance": SCORE_ROUNDING_TOLERANCE,
    }


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
    return {
        "split": record["split"],
        "run_name": record["run_name"],
        "retriever": record["retriever"],
        "counts": dict(counts),
    }


def audit_provenance(record: dict) -> dict:
    hashes = record.get("dataset_sha256")
    if hashes is not None and (
        not isinstance(hashes, dict)
        or set(hashes) != {"corpus", "queries", "qrels"}
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes.values()
        )
    ):
        raise ValueError("dataset fingerprints must contain all three SHA-256 identities")
    commit = record.get("commit")
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{7,40}", commit) is None
    ):
        raise ValueError("evaluation commit must be a hexadecimal revision")
    created = record.get("created_at")
    if created is not None:
        try:
            if not isinstance(created, str) or datetime.fromisoformat(created).utcoffset() is None:
                raise ValueError("missing timezone")
        except ValueError as error:
            raise ValueError("evaluation creation time must include an ISO timezone") from error
    return {
        "dataset_sha256": None if hashes is None else dict(hashes),
        "dataset_identity_recorded": hashes is not None,
        "commit": commit,
        "created_at": created,
        "original_data_reloaded": False,
    }


def audit_diagnostics(record: dict) -> dict:
    protocol = audit_protocol(record)
    scores = validated_scores(record)
    details = record.get("diagnostics")
    if details is None or details == {}:
        return {"recorded": False}
    if not isinstance(details, dict) or set(details) != set(record["per_query"]):
        raise ValueError("diagnostics must cover exactly the scored questions")
    counts = protocol["counts"]
    for key, detail in details.items():
        validate_diagnostic(detail)
        if (
            detail["returned"] > min(counts["max_results"], counts["corpus_documents"])
            or detail["primary_labels"] > counts["corpus_documents"]
        ):
            raise ValueError("diagnostic counts exceed the evaluation corpus or depth")
        first = detail["first_primary_rank"]
        expected_mrr = 1 / first if first is not None and first <= 10 else 0.0
        row = scores[int(key)]
        if abs(row["mrr_at_10"] - expected_mrr) > SCORE_ROUNDING_TOLERANCE:
            raise ValueError("saved MRR disagrees with the diagnostic first-primary rank")
        if detail["returned"] <= 100:
            labels = detail["primary_labels"]
            recall = detail["primary_retrieved"] / labels if labels else 0.0
            if abs(row["recall_at_100"] - recall) > SCORE_ROUNDING_TOLERANCE:
                raise ValueError("saved Recall@100 disagrees with diagnostic primary counts")
    statuses = Counter(row["status"] for row in details.values())
    return {
        "recorded": True,
        "queries": len(details),
        "status_counts": dict(sorted(statuses.items())),
    }
