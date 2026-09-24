"""Explain retrieval failures without treating unjudged answers as irrelevant."""


def ranking_diagnostics(rankings: dict, qrels: dict, cutoff: int = 10) -> dict:
    if cutoff <= 0:
        raise ValueError("diagnostic cutoff must be positive")
    diagnostics = {}
    for question_id, ranking in rankings.items():
        labels = qrels.get(question_id, {})
        primary = {answer_id for answer_id, grade in labels.items() if grade >= 2}
        positions = [rank for rank, answer_id in enumerate(ranking, 1) if answer_id in primary]
        first = min(positions, default=None)
        if not primary:
            status = "no_primary_label"
        elif first is None:
            status = "not_retrieved"
        elif first > cutoff:
            status = "below_cutoff"
        else:
            status = "top_k"
        diagnostics[question_id] = {
            "status": status, "cutoff": cutoff, "first_primary_rank": first,
            "primary_labels": len(primary), "primary_retrieved": len(positions),
            "sibling_retrieved": sum(labels.get(answer_id) == 1 for answer_id in ranking),
            "unjudged_retrieved": sum(answer_id not in labels for answer_id in ranking),
            "returned": len(ranking),
        }
    return diagnostics
