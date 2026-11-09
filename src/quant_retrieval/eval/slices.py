"""Descriptive, identity-preserving summaries of saved query-level scores."""

import math
from collections import defaultdict

import pandas as pd

from quant_retrieval.eval.audit import audit_diagnostics, audit_result, validated_scores
from quant_retrieval.eval.fingerprints import query_fingerprint
from quant_retrieval.eval.metrics import METRIC_NAMES


def grouped_summary(
    record: dict, labels: dict[int, str], *, group_by: str, min_queries: int = 5
) -> dict:
    audit = audit_result(record)
    scores = validated_scores(record)
    if type(min_queries) is not int or min_queries < 1:
        raise ValueError("minimum group size must be a positive integer")
    if not isinstance(group_by, str) or not group_by.strip():
        raise ValueError("grouping must have a nonempty name")
    if (
        not isinstance(labels, dict)
        or set(labels) != set(scores)
        or any(type(key) is not int for key in labels)
    ):
        raise ValueError("group labels must cover exactly the scored query IDs")
    if any(
        not isinstance(label, str) or not label or label != label.strip()
        for label in labels.values()
    ):
        raise ValueError("group labels must be nonempty trimmed strings")
    groups = defaultdict(list)
    for key in scores:
        groups[labels[key]].append(key)
    rows = []
    for label, ids in sorted(groups.items()):
        rows.append(
            {
                "label": label,
                "queries": len(ids),
                "question_ids": ids,
                "fraction": len(ids) / len(scores),
                "small_group": len(ids) < min_queries,
                "metrics": {
                    name: math.fsum(scores[key][name] for key in ids) / len(ids)
                    for name in METRIC_NAMES
                },
            }
        )
    return {
        "schema_version": 1,
        "scope": "descriptive_query_slices",
        "group_by": group_by,
        "run_name": record["run_name"],
        "split": record["split"],
        "queries": len(scores),
        "min_queries": min_queries,
        "overall_metrics": audit["metrics"]["recomputed"],
        "groups": rows,
        "interpretation": "Descriptive groups, not significance tests.",
    }


def diagnostic_labels(record: dict, group_by: str) -> dict[int, str]:
    choices = ("diagnostic_status", "primary_labels", "first_primary_rank")
    if group_by not in choices:
        raise ValueError(f"diagnostic grouping must be one of {choices}")
    if not audit_diagnostics(record)["recorded"]:
        raise ValueError("this grouping requires saved retrieval diagnostics")
    labels = {}
    for key, row in record["diagnostics"].items():
        if group_by == "diagnostic_status":
            label = row["status"]
        elif group_by == "primary_labels":
            label = str(row["primary_labels"]) if row["primary_labels"] < 2 else "2+"
        else:
            first = row["first_primary_rank"]
            label = (
                "no_hit"
                if first is None
                else "rank_1"
                if first == 1
                else "ranks_2_to_10"
                if first <= 10
                else "below_10"
            )
        labels[int(key)] = label
    return labels


def count_buckets(counts: dict[int, int], boundaries: tuple[int, ...]) -> dict[int, str]:
    """Use inclusive lower bounds and exclusive upper bounds for integer counts."""
    if (
        not isinstance(boundaries, (tuple, list))
        or not boundaries
        or any(type(edge) is not int or edge < 1 for edge in boundaries)
        or any(left >= right for left, right in zip(boundaries, boundaries[1:], strict=False))
    ):
        raise ValueError("bucket boundaries must be strictly increasing positive integers")
    if (
        not isinstance(counts, dict)
        or not counts
        or any(type(key) is not int or not 0 < key < 2**63 for key in counts)
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("bucket counts require positive int64 IDs and nonnegative integer counts")
    labels = {}
    for key, value in counts.items():
        lower = 0
        for upper in boundaries:
            if value < upper:
                labels[key] = f"{lower}_to_{upper - 1}"
                break
            lower = upper
        else:
            labels[key] = f"{lower}_plus"
    return labels


def query_length_labels(
    record: dict, queries: pd.DataFrame, boundaries: tuple[int, ...] = (10, 25, 50, 100)
) -> dict[int, str]:
    """Group exact evaluation text by whitespace words, not model-token counts."""
    audit = audit_result(record)
    if not audit["provenance"]["dataset_identity_recorded"]:
        raise ValueError("query-text grouping requires recorded dataset fingerprints")
    if (
        not isinstance(queries, pd.DataFrame)
        or not queries.columns.is_unique
        or not {"question_id", "text", "split"} <= set(queries.columns)
    ):
        raise ValueError("query table requires unique question_id, text and split columns")
    selected = queries.loc[queries["split"] == record["split"]]
    ids = selected["question_id"]
    if (
        not pd.api.types.is_integer_dtype(ids.dtype)
        or ids.isna().any()
        or (ids <= 0).any()
        or (ids >= 2**63).any()
        or ids.duplicated().any()
        or set(ids) != set(validated_scores(record))
    ):
        raise ValueError("query table IDs must match the evaluated questions exactly")
    if not all(isinstance(text, str) and text.strip() for text in selected["text"]):
        raise ValueError("query table text must be nonempty strings")
    if query_fingerprint(selected) != record["dataset_sha256"]["queries"]:
        raise ValueError("query text or row order differs from the evaluated dataset")
    return count_buckets(
        {int(row.question_id): len(row.text.split()) for row in selected.itertuples()}, boundaries
    )


def _paired_audits(baseline: dict, candidate: dict) -> tuple[dict, dict]:
    left_audit, right_audit = audit_result(baseline), audit_result(candidate)
    for field in ("split", "counts"):
        if left_audit["protocol"][field] != right_audit["protocol"][field]:
            raise ValueError("paired slices require identical split and evaluation counts")
    hashes = left_audit["provenance"]["dataset_sha256"]
    if hashes != right_audit["provenance"]["dataset_sha256"]:
        raise ValueError("paired slices require matching dataset identities or two legacy runs")
    if set(baseline["per_query"]) != set(candidate["per_query"]):
        raise ValueError("paired analyses require exactly the same query IDs")
    return left_audit, right_audit


def compare_groups(
    baseline: dict, candidate: dict, labels: dict[int, str], *, group_by: str, min_queries: int = 5
) -> dict:
    left_audit, _ = _paired_audits(baseline, candidate)
    hashes = left_audit["provenance"]["dataset_sha256"]
    left = grouped_summary(baseline, labels, group_by=group_by, min_queries=min_queries)
    right = grouped_summary(candidate, labels, group_by=group_by, min_queries=min_queries)
    rows = []
    for before, after in zip(left["groups"], right["groups"], strict=True):
        rows.append(
            {
                "label": before["label"],
                "queries": before["queries"],
                "question_ids": before["question_ids"],
                "small_group": before["small_group"],
                "baseline_metrics": before["metrics"],
                "candidate_metrics": after["metrics"],
                "delta": {
                    name: after["metrics"][name] - before["metrics"][name] for name in METRIC_NAMES
                },
            }
        )
    return {
        "schema_version": 1,
        "scope": "descriptive_paired_slices",
        "group_by": group_by,
        "baseline": baseline["run_name"],
        "candidate": candidate["run_name"],
        "queries": left["queries"],
        "min_queries": min_queries,
        "groups": rows,
        "dataset_identity_matched": hashes is not None,
        "group_assignment": "one fixed partition shared by both runs",
        "interpretation": "Observed group differences, not significance tests.",
    }


def status_transitions(baseline: dict, candidate: dict) -> dict:
    left, right = _paired_audits(baseline, candidate)
    if not left["diagnostics"]["recorded"] or not right["diagnostics"]["recorded"]:
        return {"recorded": False}
    cells = defaultdict(list)
    for key in sorted(baseline["per_query"], key=int):
        pair = (baseline["diagnostics"][key]["status"], candidate["diagnostics"][key]["status"])
        cells[pair].append(int(key))
    return {
        "recorded": True,
        "queries": len(baseline["per_query"]),
        "transitions": [
            {
                "baseline_status": before,
                "candidate_status": after,
                "queries": len(ids),
                "question_ids": ids,
            }
            for (before, after), ids in sorted(cells.items())
        ],
    }
