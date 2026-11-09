"""Descriptive, identity-preserving summaries of saved query-level scores."""

import math
from collections import defaultdict

from quant_retrieval.eval.audit import audit_diagnostics, audit_result, validated_scores
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
